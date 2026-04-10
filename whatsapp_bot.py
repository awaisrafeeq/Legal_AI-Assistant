"""
WhatsApp Bot - Improved Version
GreenAPI webhook handler with:
- English-forced responses (French documents → English answers)
- Voice/Audio message handling via Whisper transcription
- STRICT group-only mode (zero private chat responses)
- Conversation history per sender
- Caption-based audio mention detection
"""

import os
import io
import logging
import tempfile
import unicodedata
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI

from logging.handlers import TimedRotatingFileHandler
os.makedirs("logs", exist_ok=True)

_log_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_file_handler = TimedRotatingFileHandler("logs/whatsapp_bot.log", when="midnight", backupCount=30, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GreenAPIConfig:
    api_url: str
    id_instance: str
    api_token_instance: str
    allowed_group_id: str       # ONLY this group gets responses
    bot_name: str               # e.g. "LegalBot"
    bot_phone: str              # Bot's own phone number (to avoid self-reply loops)
    allowed_dm_phones: List[str]  # Phone numbers allowed to DM bot directly

    @classmethod
    def from_env(cls) -> "GreenAPIConfig":
        dm_phones_raw = os.environ.get(
            "ALLOWED_DM_PHONES",
            "19548878885,15148621541,15149236664,923347525551,923322551960"
        )
        dm_phones = [p.strip().lstrip("+") for p in dm_phones_raw.split(",") if p.strip()]

        return cls(
            api_url=os.environ.get("GREENAPI_URL", "https://7101.api.greenapi.com"),
            id_instance=os.environ["GREENAPI_INSTANCE_ID"],
            api_token_instance=os.environ["GREENAPI_TOKEN"],
            allowed_group_id=os.environ["ALLOWED_GROUP_ID"],   # e.g. "12345678901@g.us"
            bot_name=os.environ.get("BOT_NAME", "LegalBot"),
            bot_phone=os.environ.get("BOT_PHONE", ""),         # e.g. "923001234567"
            allowed_dm_phones=dm_phones,
        )


# ============================================================================
# GreenAPI Client
# ============================================================================

class GreenAPIClient:
    def __init__(self, config: GreenAPIConfig):
        self.config = config
        self.base_url = f"{config.api_url}/waInstance{config.id_instance}"

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        url = f"{self.base_url}/{endpoint}/{self.config.api_token_instance}"
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GreenAPI request failed: {e}")
            raise

    def send_text_message(self, chat_id: str, message: str) -> Dict:
        """Send a text message, auto-splitting if too long for WhatsApp."""
        MAX_LEN = 4096  # Safe limit for GreenAPI
        if len(message) <= MAX_LEN:
            data = {"chatId": chat_id, "message": message}
            return self._make_request("POST", "sendMessage", data)

        # Split into multiple messages at line boundaries
        parts = []
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                if current:
                    parts.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            parts.append(current)

        result = None
        for i, part in enumerate(parts):
            if len(parts) > 1:
                header = f"_({i+1}/{len(parts)})_\n" if i > 0 else ""
                part = header + part
            data = {"chatId": chat_id, "message": part}
            result = self._make_request("POST", "sendMessage", data)
        return result

    def download_media(self, chat_id: str, id_message: str) -> Optional[bytes]:
        """Download media (voice/audio) by message ID via GreenAPI"""
        try:
            data = {"chatId": chat_id, "idMessage": id_message}
            result = self._make_request("POST", "downloadFile", data)
            download_url = result.get("downloadUrl", "")
            if not download_url:
                logger.error(f"No downloadUrl in response: {result}")
                return None
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.error(f"Media download failed: {e}")
            return None


# ============================================================================
# RAG Backend Client
# ============================================================================

class RAGBackendClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def chat(self, query: str, history: List[Dict] = None, conversation_id: str = None) -> Dict[str, Any]:
        """
        Send query to RAG backend.
        - target_language is always 'en' because client wants English responses
        - history is passed for conversation context
        """
        payload = {
            "query": query,
            "target_language": "auto",     # Respond in same language as user's query
            "conversation_id": conversation_id,
            "history": history or [],
            "source_mode": "all"
        }
        try:
            base_url = self.base_url.rstrip('/')
            response = requests.post(
                f"{base_url}/chat",
                json=payload,
                timeout=180    # 3 min timeout for complex queries with metadata filtering
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG backend request failed: {e}")
            raise


# ============================================================================
# Whisper Transcription
# ============================================================================

class WhisperTranscriber:
    """
    Transcribes voice messages using Azure OpenAI Whisper.
    Falls back to standard OpenAI Whisper if Azure deployment not configured.
    """
    def __init__(self):
        whisper_endpoint = os.environ.get("WHISPER_ENDPOINT", "")
        openai_endpoint = os.environ.get("OPENAI_ENDPOINT", "")
        azure_endpoint = whisper_endpoint if whisper_endpoint else openai_endpoint
        
        whisper_key = os.environ.get("WHISPER_KEY", "")
        openai_key = os.environ.get("OPENAI_KEY", "")
        api_key = whisper_key if whisper_key else openai_key
        
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
        )
        self.whisper_deployment = os.environ.get("OPENAI_WHISPER_DEPLOYMENT", "whisper")

    def transcribe(self, audio_bytes: bytes, file_extension: str = "ogg") -> Optional[str]:
        """
        Transcribe audio bytes using Whisper.
        WhatsApp voice notes come as .ogg (Opus codec).
        Returns transcribed text or None on failure.
        """
        try:
            # Write to temp file — Whisper API requires a file-like object
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model=self.whisper_deployment,
                    file=audio_file,
                    response_format="text"
                )

            os.unlink(tmp_path)   # Clean up temp file
            transcription = result.strip() if isinstance(result, str) else result
            logger.info(f"Transcription: '{str(transcription)[:100]}'")
            return str(transcription)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None


# ============================================================================
# Conversation Memory (per sender, in-memory)
# ============================================================================

# class ConversationMemory:
#     """
#     Stores last N messages per sender for context.
#     In production, replace with Redis or a database.
#     """
#     MAX_HISTORY = 6  # Last 3 turns (user + assistant × 3)

#     def __init__(self):
#         self._store: Dict[str, List[Dict]] = {}

#     def get(self, sender_id: str) -> List[Dict]:
#         return self._store.get(sender_id, [])

#     def add(self, sender_id: str, role: str, content: str):
#         if sender_id not in self._store:
#             self._store[sender_id] = []
#         self._store[sender_id].append({"role": role, "content": content})
#         # Keep only last MAX_HISTORY messages
#         if len(self._store[sender_id]) > self.MAX_HISTORY:
#             self._store[sender_id] = self._store[sender_id][-self.MAX_HISTORY:]

#     def clear(self, sender_id: str):
#         self._store[sender_id] = []

class ConversationMemory:
    """
    Stores last N messages per sender for context.
    Also stores bot messages by WhatsApp message ID so replies to old bot
    messages can be resolved correctly.
    """
    MAX_HISTORY = 6  # Last 3 turns (user + assistant × 3)

    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}
        self._bot_messages: Dict[str, Dict[str, Dict]] = {}

    def get(self, sender_id: str) -> List[Dict]:
        return self._store.get(sender_id, [])

    def add(self, sender_id: str, role: str, content: str):
        if sender_id not in self._store:
            self._store[sender_id] = []
        self._store[sender_id].append({"role": role, "content": content})
        if len(self._store[sender_id]) > self.MAX_HISTORY:
            self._store[sender_id] = self._store[sender_id][-self.MAX_HISTORY:]

    def clear(self, sender_id: str):
        self._store[sender_id] = []
        self._bot_messages[sender_id] = {}

    def save_bot_message(
        self,
        sender_id: str,
        whatsapp_message_id: str,
        answer: str,
        query: str,
        sources: Optional[List[Dict]] = None
    ):
        if not whatsapp_message_id:
            return

        if sender_id not in self._bot_messages:
            self._bot_messages[sender_id] = {}

        self._bot_messages[sender_id][whatsapp_message_id] = {
            "role": "assistant",
            "content": answer,
            "query": query,
            "sources": sources or []
        }

    def get_bot_message_by_id(self, sender_id: str, whatsapp_message_id: str) -> Optional[Dict]:
        return self._bot_messages.get(sender_id, {}).get(whatsapp_message_id)


# ============================================================================
# WhatsApp Handler
# ============================================================================

class WhatsAppHandler:
    def __init__(
        self,
        green_api: GreenAPIClient,
        rag_backend: RAGBackendClient,
        transcriber: WhisperTranscriber,
    ):
        self.green_api = green_api
        self.rag_backend = rag_backend
        self.transcriber = transcriber
        self.memory = ConversationMemory()

    # ------------------------------------------------------------------
    # Group & Mention Guards
    # ------------------------------------------------------------------

    def _is_from_allowed_group(self, chat_id: str) -> bool:
        """
        STRICT: Only the exact configured group passes.
        Private chats, other groups → all silently ignored.
        """
        is_group = chat_id.endswith("@g.us")
        is_allowed = chat_id == self.green_api.config.allowed_group_id

        if not is_group:
            logger.info(f"Blocked — private chat: {chat_id}")
            return False
        if not is_allowed:
            logger.info(f"Blocked — wrong group: {chat_id}")
            return False

        return True

    def _is_allowed_dm(self, chat_id: str) -> bool:
        """Check if this is a private chat from an allowed DM phone number."""
        if not chat_id.endswith("@c.us"):
            return False
        phone = chat_id.replace("@c.us", "")
        return phone in self.green_api.config.allowed_dm_phones

    def _is_bot_mentioned_in_text(self, text: str) -> bool:
        """
        Check if the bot is mentioned in a text message.
        Matches @BotName, @botname (case-insensitive).
        WhatsApp inserts invisible Unicode direction chars (\u2068 \u2069 etc.)
        around mentions — strip them before matching.
        """
        # Strip invisible Unicode formatting/direction characters
        cleaned = ''.join(c for c in text if unicodedata.category(c) not in ('Cf', 'Cc') or c in ('\n', '\t'))
        cleaned_lower = cleaned.lower()
        bot_name = self.green_api.config.bot_name.lower()

        mention_patterns = [
            f"@{bot_name}",
            f"@{bot_name.replace(' ', '')}",   # handle spaces in bot name
        ]

        for pattern in mention_patterns:
            if pattern in cleaned_lower:
                return True
        return False

    def _is_bot_mentioned_in_participants(self, message_data: Dict) -> bool:
        """
        GreenAPI includes 'mentionedJidList' when someone is @mentioned.
        Check if bot's phone is in that list.
        """
        bot_phone = self.green_api.config.bot_phone
        if not bot_phone:
            return False

        # mentionedJidList is in extended text message data
        extended = message_data.get("extendedTextMessageData", {})
        mentioned = extended.get("mentionedJidList", [])
        bot_jid = f"{bot_phone}@s.whatsapp.net"
        return bot_jid in mentioned

    def _is_bot_mentioned(self, text: str, message_data: Dict) -> bool:
        """Combined check: text mention OR JID mention"""
        return (
            self._is_bot_mentioned_in_text(text)
            or self._is_bot_mentioned_in_participants(message_data)
        )

    def _is_audio_for_bot(self, message_data: Dict) -> bool:
        """
        Voice/audio message is processed if EITHER:
        1. The caption contains @BotName (user typed caption when sending voice)
        2. There is no caption (auto-process all voice in allowed group — optional behavior)

        Change AUTO_PROCESS_ALL_AUDIO to False to require caption mention.
        """
        AUTO_PROCESS_ALL_AUDIO = os.environ.get("AUTO_PROCESS_AUDIO", "true").lower() == "true"

        file_data = message_data.get("fileMessageData", {})
        caption = file_data.get("caption", "").strip()

        if AUTO_PROCESS_ALL_AUDIO:
            return True   # Process all voice notes in allowed group

        # If auto off, require @BotName in caption
        return self._is_bot_mentioned_in_text(caption)

    # ------------------------------------------------------------------
    # Message Extraction
    # ------------------------------------------------------------------

    def _extract_text(self, message_data: Dict) -> Optional[str]:
        """Extract text from text-type messages"""
        type_msg = message_data.get("typeMessage", "")

        if type_msg == "textMessage":
            return message_data.get("textMessageData", {}).get("textMessage", "")

        elif type_msg == "extendedTextMessage":
            return message_data.get("extendedTextMessageData", {}).get("text", "")

        return None

    def _is_voice_or_audio(self, message_data: Dict) -> bool:
        type_msg = message_data.get("typeMessage", "")
        return type_msg in ("voiceMessage", "audioMessage")

    def _clean_query(self, text: str) -> str:
        """Remove @mentions and bot name from query before sending to RAG"""
        bot_name = self.green_api.config.bot_name.lower()
        patterns = [
            f"@{bot_name}",
            f"@{bot_name.replace(' ', '')}",
            "@ai", "@bot",
        ]
        cleaned = text
        for p in patterns:
            cleaned = cleaned.lower().replace(p, "").replace(p.replace("@", "@ "), "")
        # Restore original casing by removing only matched parts
        import re
        for p in patterns:
            cleaned = re.sub(re.escape(p), "", text, flags=re.IGNORECASE)
            text = cleaned
        return " ".join(cleaned.split()).strip()

    # ------------------------------------------------------------------
    # Reference Management
    # ------------------------------------------------------------------
    
    
    def _has_explicit_same_reference(self, query: str) -> bool:
        q = query.lower().strip()

        phrases = [
            "same email",
            "same document",
            "same report",
            "same source",
            "that same email",
            "that same document",
            "that same report",
            "that email",
            "that document",
            "that report",
            "this email",
            "this document",
            "this report",
            "same case",
            "that case"
        ]
        return any(p in q for p in phrases)

    def _has_new_identifier(self, query: str) -> bool:
        import re

        q = query.lower()

        keyword_identifiers = [
            "couvent",
            "brompton",
            "st-augustin",
            "st augustin",
            "st-paul",
            "8181772",
            "9301-7291",
            "financial report",
            "invoice register",
            "civil plans",
            "email",
            "courriel",
            "grands livres",
            "rapport financier",
            "mutation",
            "invoice",
            "register"
        ]

        if any(k in q for k in keyword_identifiers):
            return True

        if re.search(r"\b\d{4,}\b", q):
            return True

        if re.search(r"\b(20\d{2}|19\d{2})\b", q):
            return True

        return False

    def _has_dependent_language(self, query: str) -> bool:
        q = query.lower().strip()

        phrases = [
            "who sent it",
            "who received it",
            "what about",
            "and the date",
            "what is the date",
            "summarize it",
            "explain it",
            "its source",
            "its amount",
            "that one",
            "this one",
            "and this",
            "and that",
            "it",
            "this",
            "that"
        ]

        return any(p in q for p in phrases)

    def _classify_query_context_mode(self, query: str) -> str:
        """
        Returns:
        - 'reply_context'
        - 'follow_up'
        - 'fresh'
        """
        if self._has_explicit_same_reference(query):
            return "follow_up"

        if self._has_new_identifier(query):
            return "fresh"

        if self._has_dependent_language(query):
            return "follow_up"

        return "fresh"

    def _extract_quoted_message_id(self, webhook_data: Dict[str, Any]) -> Optional[str]:
        """
        Try to extract the replied/quoted message ID from GreenAPI webhook payload.
        Different payload shapes may exist, so we check multiple common locations.
        """
        message_data = webhook_data.get("messageData", {})

        candidates = [
            message_data.get("quotedMessage", {}).get("stanzaId"),
            message_data.get("quotedMessageData", {}).get("stanzaId"),
            message_data.get("extendedTextMessageData", {}).get("stanzaId"),
            message_data.get("extendedTextMessageData", {}).get("quotedMessage", {}).get("stanzaId"),
            message_data.get("textMessageData", {}).get("quotedMessage", {}).get("stanzaId"),
            webhook_data.get("quotedMessage", {}).get("stanzaId"),
            webhook_data.get("quotedMessageData", {}).get("stanzaId"),
        ]

        for c in candidates:
            if c:
                return c

        return None
    
    # ------------------------------------------------------------------
    # Response Formatting
    # ------------------------------------------------------------------

    def _format_response(self, answer: str, sources: List[Dict], query_was_voice: bool = False) -> str:
        """
        Format the AI answer for WhatsApp.
        WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```code```
        """
        prefix = "🎤 _Voice query processed_\n\n" if query_was_voice else ""

        header = "🤖 *Legal AI Assistant*\n" + "─" * 28 + "\n\n"
        body = answer.strip()

        # Sources section (all unique sources)
        sources_text = ""
        if sources:
            sources_text = "\n\n" + "─" * 28 + "\n📚 *Sources:*\n"
            seen = set()
            count = 0
            for source in sources:
                fname = source.get("file_name", "Unknown")
                if fname in seen:
                    continue
                seen.add(fname)
                count += 1

                container = source.get("source_container", "")
                # FIX: exact match instead of substring — "internal" substring could match wrong values
                # Now aligns with actual value written by index_internal.py: "legal-documents-internal"
                icon = "🔴" if container == "legal-documents-internal" else "📗"
                label = "Internal (Confidential)" if container == "legal-documents-internal" else "Legal Documents"
                source_url = source.get("source_url", "")

                folder = source.get("folder_path", "")
                full_path = f"{folder}/{fname}" if folder else fname
                sources_text += f"\n{count}. {icon} {full_path}\n"
                sources_text += f"   _{label}_\n"
                if source_url:
                    sources_text += f"   🔗 {source_url}\n"

        footer = "\n\n_Reply to this message or tag @" + self.green_api.config.bot_name + " to ask a follow-up._"

        return prefix + header + body + sources_text + footer

    # ------------------------------------------------------------------
    # Core Handler
    # ------------------------------------------------------------------

    def handle_incoming_message(self, webhook_data: Dict[str, Any]) -> None:
        """Main entry point for all incoming webhooks"""
        try:
            type_webhook = webhook_data.get("typeWebhook", "")

            # Only process incoming messages
            if type_webhook != "incomingMessageReceived":
                return

            sender_data = webhook_data.get("senderData", {})
            chat_id = sender_data.get("chatId", "")
            sender_name = sender_data.get("senderName", "User")
            sender_phone = sender_data.get("sender", "")  # e.g. "923001234567@s.whatsapp.net"
            id_message = webhook_data.get("idMessage", "")

            logger.info(f"Incoming from {sender_name} | chat: {chat_id}")

            # ── GUARD 1: Must be from allowed group OR allowed DM ──
            is_dm = self._is_allowed_dm(chat_id)
            if not is_dm and not self._is_from_allowed_group(chat_id):
                return

            # ── GUARD 2: Ignore bot's own messages (prevent loops) ──
            bot_phone = self.green_api.config.bot_phone
            if bot_phone and sender_phone.startswith(bot_phone):
                logger.info("Ignoring own message")
                return

            message_data = webhook_data.get("messageData", {})

            # ── VOICE / AUDIO HANDLING ──
            # if self._is_voice_or_audio(message_data):
            #     self._handle_voice_message(
            #         chat_id, sender_phone, sender_name, id_message, message_data
            #     )
            #     return
            if self._is_voice_or_audio(message_data):
                # DMs: always process voice. Groups: check _is_audio_for_bot.
                if not is_dm and not self._is_audio_for_bot(message_data):
                    logger.info("Voice note not addressed to bot (no caption mention)")
                    return

                quoted_message_id = self._extract_quoted_message_id(webhook_data)
                reply_context = None

                if quoted_message_id:
                    reply_context = self.memory.get_bot_message_by_id(sender_phone, quoted_message_id)
                    if reply_context:
                        logger.info(f"Voice reply context found for quoted bot message: {quoted_message_id}")
                    else:
                        logger.info(f"Voice quoted message ID found but no bot context stored: {quoted_message_id}")

                self._handle_voice_message(
                    chat_id, sender_phone, sender_name, id_message, message_data, reply_context
                )
                return

            # ── TEXT MESSAGE ──
            text = self._extract_text(message_data)
            if not text:
                return

            # ── GUARD 3: Bot must be mentioned in text (groups only, DMs skip this) ──
            if not is_dm and not self._is_bot_mentioned(text, message_data):
                logger.info(f"Bot not mentioned, skipping: '{text[:40]}'")
                return

            # Handle special commands
            cleaned = self._clean_query(text)
            lower = cleaned.lower().strip()

            quoted_message_id = self._extract_quoted_message_id(webhook_data)
            reply_context = None

            if quoted_message_id:
                reply_context = self.memory.get_bot_message_by_id(sender_phone, quoted_message_id)
                if reply_context:
                    logger.info(f"Reply context found for quoted bot message: {quoted_message_id}")
                else:
                    logger.info(f"Quoted message ID found but no bot context stored: {quoted_message_id}")

            if lower in ["hi", "hello", "help", "/start", "/help"]:
                self._send_welcome(chat_id)
                return

            if lower == "/clear":
                self.memory.clear(sender_phone)
                self.green_api.send_text_message(chat_id, "🗑️ Conversation history cleared.")
                return

            # ── QUERY RAG ──
            self._query_and_respond(
                chat_id=chat_id,
                sender_phone=sender_phone,
                sender_name=sender_name,
                query=cleaned,
                is_voice=False,
                reply_context=reply_context
            )

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    def _handle_voice_message(
        self,
        chat_id: str,
        sender_phone: str,
        sender_name: str,
        id_message: str,
        message_data: Dict,
        reply_context: Optional[Dict] = None
    ) -> None:
        """Download, transcribe, and process voice message"""

        # Check if this voice message is meant for the bot
        if not self._is_audio_for_bot(message_data):
            logger.info("Voice note not addressed to bot (no caption mention)")
            return

        logger.info(f"Processing voice message from {sender_name}")

        # Send acknowledgment immediately
        self.green_api.send_text_message(
            chat_id,
            f"🎤 _Voice message received. Transcribing..._\n⏳ Please wait a moment."
        )

        # Download audio
        audio_bytes = self.green_api.download_media(chat_id, id_message)
        if not audio_bytes:
            self.green_api.send_text_message(
                chat_id,
                "❌ Could not download your voice message. Please try sending it again."
            )
            return

        # Transcribe
        transcription = self.transcriber.transcribe(audio_bytes, file_extension="ogg")
        if not transcription or len(transcription.strip()) < 3:
            self.green_api.send_text_message(
                chat_id,
                "❌ Could not transcribe your voice message. Please send your question as text."
            )
            return

        logger.info(f"Transcription: '{transcription[:80]}'")

        # Show user what was understood
        self.green_api.send_text_message(
            chat_id,
            f"🎤 *Heard:* _{transcription}_\n\n⏳ Searching documents..."
        )

        # Query RAG with transcription
        self._query_and_respond(
            chat_id=chat_id,
            sender_phone=sender_phone,
            sender_name=sender_name,
            query=transcription,
            is_voice=True,
            reply_context=reply_context
        )

    # def _query_and_respond(
    #     self,
    #     chat_id: str,
    #     sender_phone: str,
    #     sender_name: str,
    #     query: str,
    #     is_voice: bool = False
    # ) -> None:
    #     """Send query to RAG backend and reply in WhatsApp"""
    #     if not query:
    #         return

    #     try:
    #         # Get conversation history for this sender
    #         history = self.memory.get(sender_phone)

    #         logger.info(f"Querying RAG: '{query[:60]}' | history: {len(history)} msgs")

    #         response = self.rag_backend.chat(
    #             query=query,
    #             history=history,
    #             conversation_id=sender_phone
    #         )

    #         answer = response.get("answer", "I could not find an answer. Please try rephrasing.")
    #         sources = response.get("sources", [])

    #         # Update conversation memory
    #         self.memory.add(sender_phone, "user", query)
    #         self.memory.add(sender_phone, "assistant", answer)

    #         # Format and send
    #         formatted = self._format_response(answer, sources, query_was_voice=is_voice)
    #         self.green_api.send_text_message(chat_id, formatted)

    #         logger.info(f"Response sent to {chat_id}")

    #     except Exception as e:
    #         logger.error(f"RAG error: {e}", exc_info=True)
    #         self.green_api.send_text_message(
    #             chat_id,
    #             "❌ An error occurred while processing your request. Please try again."
    #         )
    def _query_and_respond(
        self,
        chat_id: str,
        sender_phone: str,
        sender_name: str,
        query: str,
        is_voice: bool = False,
        reply_context: Optional[Dict] = None
    ) -> None:
        """Send query to RAG backend and reply in WhatsApp"""
        if not query:
            return

        try:
            full_history = self.memory.get(sender_phone)

            if reply_context:
                history = [
                    {"role": "user", "content": reply_context.get("query", "")},
                    {"role": "assistant", "content": reply_context.get("content", "")}
                ]
                context_mode = "reply_context"
                logger.info("Reply-to-message detected | using replied bot message as context")

            else:
                context_mode = self._classify_query_context_mode(query)

                if context_mode == "follow_up":
                    history = full_history[-2:]   # only last 1 turn
                    logger.info(f"Follow-up query detected | using short history: {len(history)} msgs")
                else:
                    history = []
                    logger.info("Fresh query detected | ignoring history")

            logger.info(f"Querying RAG: '{query[:60]}' | mode={context_mode} | history={len(history)} msgs")

            response = self.rag_backend.chat(
                query=query,
                history=history,
                conversation_id=sender_phone
            )

            answer = response.get("answer", "I could not find an answer. Please try rephrasing.")
            sources = response.get("sources", [])

            self.memory.add(sender_phone, "user", query)
            self.memory.add(sender_phone, "assistant", answer)

            formatted = self._format_response(answer, sources, query_was_voice=is_voice)
            send_result = self.green_api.send_text_message(chat_id, formatted)

            sent_message_id = send_result.get("idMessage", "")
            self.memory.save_bot_message(
                sender_id=sender_phone,
                whatsapp_message_id=sent_message_id,
                answer=answer,
                query=query,
                sources=sources
            )

            logger.info(f"Response sent to {chat_id}")

        except Exception as e:
            logger.error(f"RAG error: {e}", exc_info=True)
            self.green_api.send_text_message(
                chat_id,
                "❌ An error occurred while processing your request. Please try again."
            )

    def _send_welcome(self, chat_id: str) -> None:
        bot_name = self.green_api.config.bot_name
        msg = (
            f"👋 *Welcome to {bot_name}!*\n\n"
            "I'm your AI-powered legal document assistant.\n\n"
            "*How to use me:*\n"
            f"• Tag me: `@{bot_name} your question here`\n"
            f"• Voice notes: Record and send (I'll transcribe and answer)\n"
            f"• Voice with caption: Add `@{bot_name}` as caption\n\n"
            "*Example questions:*\n"
            "• `@{bot_name} What does Article 15 say about land rights?`\n"
            "• `@{bot_name} Find cases about contract disputes`\n\n"
            "*Commands:*\n"
            "• `/clear` — Clear your conversation history\n"
            "• `/help` — Show this message\n\n"
            "_All responses are in English. Sources are shown for every answer._"
        )
        self.green_api.send_text_message(chat_id, msg)


# ============================================================================
# FastAPI App
# ============================================================================

whatsapp_handler: Optional[WhatsAppHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whatsapp_handler

    green_config = GreenAPIConfig.from_env()
    green_api = GreenAPIClient(green_config)
    rag_backend = RAGBackendClient(
        base_url=os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")
    )
    transcriber = WhisperTranscriber()

    whatsapp_handler = WhatsAppHandler(green_api, rag_backend, transcriber)
    logger.info("WhatsApp handler initialized (Improved v2.0)")

    yield
    logger.info("Shutting down WhatsApp bot...")


app = FastAPI(
    title="Legal AI WhatsApp Bot",
    description="GreenAPI webhook — English responses, voice support, strict group mode",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "whatsapp-bot", "version": "2.0.0"}


@app.post("/webhook/greenapi")
async def greenapi_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive webhooks from GreenAPI"""
    try:
        payload = await request.json()
        webhook_type = payload.get("typeWebhook", "unknown")
        sender_data = payload.get("senderData", {})
        chat_id = sender_data.get("chatId", "unknown")

        logger.info(f"WEBHOOK | type={webhook_type} | chat={chat_id}")

        # Always respond 200 immediately, process in background
        background_tasks.add_task(whatsapp_handler.handle_incoming_message, payload)
        return {"status": "received"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/webhook/greenapi")
async def greenapi_webhook_get():
    return {"status": "webhook active"}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
