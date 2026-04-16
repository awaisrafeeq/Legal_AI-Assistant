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
import json

from logging.handlers import TimedRotatingFileHandler
import prompts

# Cosmos DB (optional, for persistent memory)
try:
    from azure.cosmos import CosmosClient, PartitionKey
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False
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

    def _split_message_parts(self, message: str, max_len: int = 4096) -> List[str]:
        """Split a message into WhatsApp-safe parts, preserving line boundaries."""
        if len(message) <= max_len:
            return [message]

        parts = []
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > max_len:
                if current:
                    parts.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            parts.append(current)
        return parts

    def send_text_message(self, chat_id: str, message: str) -> Dict:
        """Send a text message, auto-splitting if too long for WhatsApp."""
        MAX_LEN = 4096  # Safe limit for GreenAPI
        parts = self._split_message_parts(message, MAX_LEN)
        if len(parts) == 1:
            data = {"chatId": chat_id, "message": message}
            return self._make_request("POST", "sendMessage", data)

        all_results = []
        sent_parts = []
        for i, part in enumerate(parts):
            if len(parts) > 1:
                header = f"_({i+1}/{len(parts)})_\n" if i > 0 else ""
                part = header + part
            data = {"chatId": chat_id, "message": part}
            result = self._make_request("POST", "sendMessage", data)
            all_results.append(result)
            sent_parts.append(part)

        # Return combined result with all message IDs for multi-message tracking
        combined = all_results[-1] if all_results else {}
        if len(all_results) > 1:
            combined["allMessageIds"] = [r.get("idMessage", "") for r in all_results if r.get("idMessage")]
            combined["sentParts"] = sent_parts
        return combined

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

    def chat(
        self,
        query: str,
        history: List[Dict] = None,
        conversation_id: str = None,
        exclude_blob_paths: List[str] = None,
        allowed_files: List[str] = None,
        task_type: str = None,
    ) -> Dict[str, Any]:
        """
        Send query to RAG backend.
        - history: conversation context
        - exclude_blob_paths: documents to skip (already shown to user)
        - allowed_files: restrict search to specific files only
        - task_type: legal task type (chronology, contradictions, etc.)
        """
        payload = {
            "query": query,
            "target_language": "auto",
            "conversation_id": conversation_id,
            "history": history or [],
            "source_mode": "all",
            "exclude_blob_paths": exclude_blob_paths or [],
            "allowed_files": allowed_files or [],
        }
        if task_type:
            payload["task_type"] = task_type
        try:
            base_url = self.base_url.rstrip('/')
            response = requests.post(
                f"{base_url}/chat",
                json=payload,
                timeout=3600    # 3 min timeout for complex queries with metadata filtering
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG backend request failed: {e}")
            raise

    def send_email(
        self,
        recipient_email: str,
        answer: str,
        query: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        subject: Optional[str] = None,
        formatted_body: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "recipient_email": recipient_email,
            "answer": answer,
            "query": query,
            "sources": sources or [],
            "subject": subject,
            "formatted_body": formatted_body,
        }
        try:
            base_url = self.base_url.rstrip('/')
            response = requests.post(
                f"{base_url}/share/email",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Email share request failed: {e}")
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
    Full conversation state per sender:
    - history: last N messages for context
    - shown_blob_paths: documents already shown (for "give me more" exclusion)
    - last_query / last_sources / last_filters: previous turn's state
    - topic: GPT-summarized conversation topic
    - bot_messages: all bot message IDs → context (for reply handling)
    """
    MAX_HISTORY = 16  # Last 8 turns (user + assistant × 8)

    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}
        self._bot_messages: Dict[str, Dict[str, Dict]] = {}
        self._state: Dict[str, Dict] = {}

    # ── History ──

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
        self._state[sender_id] = {}

    # ── Conversation State ──

    def get_state(self, sender_id: str) -> Dict:
        if sender_id not in self._state:
            self._state[sender_id] = {
                "topic": "",
                "last_query": "",
                "last_filters": {},
                "shown_blob_paths": set(),
                "last_sources": [],
                "all_message_ids": [],
                "last_activity": None,
            }
        return self._state[sender_id]

    def update_state(self, sender_id: str, **kwargs):
        state = self.get_state(sender_id)
        state.update(kwargs)
        state["last_activity"] = datetime.now()

    def add_shown_blob_paths(self, sender_id: str, blob_paths: List[str]):
        state = self.get_state(sender_id)
        state["shown_blob_paths"].update(blob_paths)

    def clear_shown_blob_paths(self, sender_id: str):
        state = self.get_state(sender_id)
        state["shown_blob_paths"] = set()

    def get_shown_blob_paths(self, sender_id: str) -> List[str]:
        return list(self.get_state(sender_id).get("shown_blob_paths", set()))

    # ── Bot Message Tracking (all message IDs → same context) ──

    def save_bot_message(
        self,
        sender_id: str,
        whatsapp_message_ids: list,
        answer: str,
        query: str,
        sources: Optional[List[Dict]] = None,
        message_parts: Optional[List[str]] = None,
        formatted_answer: str = "",
    ):
        """Store context for ALL message IDs from a response (supports multi-message)."""
        if not whatsapp_message_ids:
            return

        if sender_id not in self._bot_messages:
            self._bot_messages[sender_id] = {}

        base_context = {
            "role": "assistant",
            "content": answer,
            "full_content": answer,
            "formatted_content": formatted_answer or answer,
            "full_formatted_content": formatted_answer or answer,
            "query": query,
            "sources": sources or [],
        }

        # Each split message keeps its own visible text while preserving the full answer.
        for idx, msg_id in enumerate(whatsapp_message_ids):
            if msg_id:
                part_text = ""
                if message_parts and idx < len(message_parts):
                    part_text = message_parts[idx]
                context = dict(base_context)
                context["content"] = part_text or answer
                context["formatted_content"] = part_text or formatted_answer or answer
                context["part_index"] = idx + 1
                context["part_count"] = len(whatsapp_message_ids)
                self._bot_messages[sender_id][msg_id] = context

        # Also update state with these message IDs
        state = self.get_state(sender_id)
        state["all_message_ids"] = whatsapp_message_ids
        state["last_sources"] = sources or []
        state["last_query"] = query

    def get_bot_message_by_id(self, sender_id: str, whatsapp_message_id: str) -> Optional[Dict]:
        return self._bot_messages.get(sender_id, {}).get(whatsapp_message_id)


# ============================================================================
# Cosmos DB Persistent Conversation Memory
# ============================================================================

class CosmosConversationMemory:
    """
    Drop-in replacement for ConversationMemory backed by Azure Cosmos DB.
    Same interface, but data survives restarts and deployments.

    Cosmos document types per sender (partition key: /sender_id):
      - {sender_id}:state       → conversation state (topic, last_query, shown_blob_paths, etc.)
      - {sender_id}:history     → message history array
      - {sender_id}:bot_msgs    → WhatsApp message ID → context mapping
    """
    MAX_HISTORY = 16
    MAX_BOT_MESSAGES = 100  # Keep last N bot messages to prevent unbounded growth
    TTL_SECONDS = 90 * 86400  # 90 days

    def __init__(self, cosmos_container):
        self._container = cosmos_container
        # Write-through cache: avoid repeated Cosmos reads within a single webhook
        self._cache_history: Dict[str, List[Dict]] = {}
        self._cache_state: Dict[str, Dict] = {}
        self._cache_bot_msgs: Dict[str, Dict[str, Dict]] = {}

    # ── Cosmos helpers ──

    def _read_doc(self, doc_id: str, sender_id: str) -> Optional[Dict]:
        """Read a document from Cosmos. Returns None if not found."""
        try:
            return self._container.read_item(item=doc_id, partition_key=sender_id)
        except Exception:
            return None

    def _upsert_doc(self, doc: Dict):
        """Create or replace a document in Cosmos."""
        try:
            self._container.upsert_item(doc)
        except Exception as e:
            logging.getLogger(__name__).error(f"Cosmos upsert failed for {doc.get('id')}: {e}")

    def _delete_doc(self, doc_id: str, sender_id: str):
        """Delete a document from Cosmos. Silently ignores if not found."""
        try:
            self._container.delete_item(item=doc_id, partition_key=sender_id)
        except Exception:
            pass

    # ── History ──

    def get(self, sender_id: str) -> List[Dict]:
        if sender_id in self._cache_history:
            return self._cache_history[sender_id]
        doc = self._read_doc(f"{sender_id}:history", sender_id)
        messages = doc.get("messages", []) if doc else []
        self._cache_history[sender_id] = messages
        return messages

    def add(self, sender_id: str, role: str, content: str):
        messages = self.get(sender_id)
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(messages) > self.MAX_HISTORY:
            messages = messages[-self.MAX_HISTORY:]
        self._cache_history[sender_id] = messages
        self._upsert_doc({
            "id": f"{sender_id}:history",
            "sender_id": sender_id,
            "doc_type": "history",
            "messages": messages,
            "ttl": self.TTL_SECONDS,
        })

    def clear(self, sender_id: str):
        self._cache_history.pop(sender_id, None)
        self._cache_state.pop(sender_id, None)
        self._cache_bot_msgs.pop(sender_id, None)
        self._delete_doc(f"{sender_id}:history", sender_id)
        self._delete_doc(f"{sender_id}:state", sender_id)
        self._delete_doc(f"{sender_id}:bot_msgs", sender_id)

    # ── Conversation State ──

    def _default_state(self) -> Dict:
        return {
            "topic": "",
            "last_query": "",
            "last_filters": {},
            "shown_blob_paths": [],
            "last_sources": [],
            "all_message_ids": [],
            "last_activity": None,
            "last_answer": "",
            "last_formatted_answer": "",
        }

    def get_state(self, sender_id: str) -> Dict:
        if sender_id in self._cache_state:
            return self._cache_state[sender_id]
        doc = self._read_doc(f"{sender_id}:state", sender_id)
        if doc:
            state = {k: v for k, v in doc.items() if k not in ("id", "sender_id", "doc_type", "ttl", "_rid", "_self", "_etag", "_attachments", "_ts")}
            # Convert shown_blob_paths list back to set
            state["shown_blob_paths"] = set(state.get("shown_blob_paths", []))
            if not state.get("last_activity"):
                state["last_activity"] = None
        else:
            state = self._default_state()
            state["shown_blob_paths"] = set()
        self._cache_state[sender_id] = state
        return state

    def _persist_state(self, sender_id: str):
        state = self._cache_state.get(sender_id)
        if state is None:
            return
        doc = dict(state)
        # Convert set to list for JSON serialization
        doc["shown_blob_paths"] = list(doc.get("shown_blob_paths", set()))
        # Convert datetime to string
        if isinstance(doc.get("last_activity"), datetime):
            doc["last_activity"] = doc["last_activity"].isoformat()
        # Truncate large fields to prevent Cosmos 2MB limit
        sources = doc.get("last_sources", [])
        if sources and len(sources) > 10:
            doc["last_sources"] = sources[:10]
        doc.update({
            "id": f"{sender_id}:state",
            "sender_id": sender_id,
            "doc_type": "state",
            "ttl": self.TTL_SECONDS,
        })
        self._upsert_doc(doc)

    def update_state(self, sender_id: str, **kwargs):
        state = self.get_state(sender_id)
        state.update(kwargs)
        state["last_activity"] = datetime.now()
        self._persist_state(sender_id)

    def add_shown_blob_paths(self, sender_id: str, blob_paths: List[str]):
        state = self.get_state(sender_id)
        state["shown_blob_paths"].update(blob_paths)
        self._persist_state(sender_id)

    def clear_shown_blob_paths(self, sender_id: str):
        state = self.get_state(sender_id)
        state["shown_blob_paths"] = set()
        self._persist_state(sender_id)

    def get_shown_blob_paths(self, sender_id: str) -> List[str]:
        return list(self.get_state(sender_id).get("shown_blob_paths", set()))

    # ── Bot Message Tracking ──

    def save_bot_message(
        self,
        sender_id: str,
        whatsapp_message_ids: list,
        answer: str,
        query: str,
        sources: Optional[List[Dict]] = None,
        message_parts: Optional[List[str]] = None,
        formatted_answer: str = "",
    ):
        if not whatsapp_message_ids:
            return

        bot_msgs = self._get_bot_messages(sender_id)

        base_context = {
            "role": "assistant",
            "content": answer,
            "full_content": answer,
            "formatted_content": formatted_answer or answer,
            "full_formatted_content": formatted_answer or answer,
            "query": query,
            "sources": sources or [],
        }

        for idx, msg_id in enumerate(whatsapp_message_ids):
            if msg_id:
                part_text = ""
                if message_parts and idx < len(message_parts):
                    part_text = message_parts[idx]
                context = dict(base_context)
                context["content"] = part_text or answer
                context["formatted_content"] = part_text or formatted_answer or answer
                context["part_index"] = idx + 1
                context["part_count"] = len(whatsapp_message_ids)
                bot_msgs[msg_id] = context

        # Trim old messages if too many
        if len(bot_msgs) > self.MAX_BOT_MESSAGES:
            keys = list(bot_msgs.keys())
            for old_key in keys[:-self.MAX_BOT_MESSAGES]:
                del bot_msgs[old_key]

        self._cache_bot_msgs[sender_id] = bot_msgs
        # Persist - strip sources content to reduce document size
        persist_msgs = {}
        for mid, ctx in bot_msgs.items():
            slim = dict(ctx)
            # Keep source metadata but drop content field to save space
            slim_sources = []
            for s in slim.get("sources", []):
                slim_sources.append({
                    "file_name": s.get("file_name", ""),
                    "blob_path": s.get("blob_path", ""),
                    "source_container": s.get("source_container", ""),
                    "source_url": s.get("source_url", ""),
                })
            slim["sources"] = slim_sources
            persist_msgs[mid] = slim

        self._upsert_doc({
            "id": f"{sender_id}:bot_msgs",
            "sender_id": sender_id,
            "doc_type": "bot_messages",
            "messages": persist_msgs,
            "ttl": self.TTL_SECONDS,
        })

        # Also update state
        state = self.get_state(sender_id)
        state["all_message_ids"] = whatsapp_message_ids
        state["last_sources"] = sources or []
        state["last_query"] = query
        self._persist_state(sender_id)

    def _get_bot_messages(self, sender_id: str) -> Dict[str, Dict]:
        if sender_id in self._cache_bot_msgs:
            return self._cache_bot_msgs[sender_id]
        doc = self._read_doc(f"{sender_id}:bot_msgs", sender_id)
        msgs = doc.get("messages", {}) if doc else {}
        self._cache_bot_msgs[sender_id] = msgs
        return msgs

    def get_bot_message_by_id(self, sender_id: str, whatsapp_message_id: str) -> Optional[Dict]:
        return self._get_bot_messages(sender_id).get(whatsapp_message_id)


# ============================================================================
# Case Memory Store (Persistent Case Facts & Findings)
# ============================================================================

class CaseMemoryStore:
    """
    Persistent store for extracted case knowledge that outlives individual conversations.
    Enables the lawyer to come back days later and the agent remembers:
    - What facts were discovered
    - What documents were discussed
    - What questions remain open
    - What contradictions were found
    - Timeline of events

    Each sender can have multiple cases. Cases are auto-resolved by topic matching.
    """
    TTL_SECONDS = 180 * 86400  # 180 days for case memory

    def __init__(self, cosmos_container):
        self._container = cosmos_container
        self._cache: Dict[str, Dict] = {}

    def _read_doc(self, doc_id: str, sender_id: str) -> Optional[Dict]:
        try:
            return self._container.read_item(item=doc_id, partition_key=sender_id)
        except Exception:
            return None

    def _upsert_doc(self, doc: Dict):
        try:
            self._container.upsert_item(doc)
        except Exception as e:
            logging.getLogger(__name__).error(f"Case memory upsert failed: {e}")

    @staticmethod
    def _normalize_text_item(value: Any) -> str:
        """Convert mixed GPT output into a stable text value for string-list fields."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            preferred_keys = (
                "fact", "name", "document", "question", "issue",
                "contradiction", "summary", "event", "text", "value"
            )
            for key in preferred_keys:
                raw = value.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value).strip()
        if isinstance(value, (list, tuple, set)):
            parts = [CaseMemoryStore._normalize_text_item(v) for v in value]
            return " | ".join(p for p in parts if p)
        return str(value).strip()

    @staticmethod
    def _normalize_timeline_event(value: Any) -> Optional[Dict[str, str]]:
        if not value:
            return None
        if isinstance(value, str):
            return {"date": "", "event": value.strip()}
        if not isinstance(value, dict):
            return {"date": "", "event": CaseMemoryStore._normalize_text_item(value)}

        date_value = CaseMemoryStore._normalize_text_item(
            value.get("date") or value.get("when") or value.get("time")
        )
        event_value = CaseMemoryStore._normalize_text_item(
            value.get("event") or value.get("description") or value.get("fact") or value.get("summary")
        )
        if not date_value and not event_value:
            return None
        return {"date": date_value, "event": event_value}

    @staticmethod
    def _normalize_action_item(value: Any) -> Optional[Dict[str, str]]:
        if not value:
            return None
        if isinstance(value, str):
            task = value.strip()
            if not task:
                return None
            return {"task": task, "priority": "medium", "status": "pending"}
        if not isinstance(value, dict):
            task = CaseMemoryStore._normalize_text_item(value)
            if not task:
                return None
            return {"task": task, "priority": "medium", "status": "pending"}

        task = CaseMemoryStore._normalize_text_item(
            value.get("task") or value.get("action") or value.get("next_step") or value.get("summary")
        )
        if not task:
            return None
        priority = CaseMemoryStore._normalize_text_item(value.get("priority")).lower() or "medium"
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        status = CaseMemoryStore._normalize_text_item(value.get("status")).lower() or "pending"
        return {"task": task, "priority": priority, "status": status}

    @staticmethod
    def _normalize_evidence_reference(value: Any) -> Optional[Dict[str, str]]:
        if not value:
            return None
        if isinstance(value, str):
            quote = value.strip()
            if not quote:
                return None
            return {"quote": quote, "source_file": "", "relevance": ""}
        if not isinstance(value, dict):
            quote = CaseMemoryStore._normalize_text_item(value)
            if not quote:
                return None
            return {"quote": quote, "source_file": "", "relevance": ""}

        quote = CaseMemoryStore._normalize_text_item(
            value.get("quote") or value.get("text") or value.get("evidence")
        )
        if not quote:
            return None
        return {
            "quote": quote,
            "source_file": CaseMemoryStore._normalize_text_item(
                value.get("source_file") or value.get("source") or value.get("document")
            ),
            "relevance": CaseMemoryStore._normalize_text_item(
                value.get("relevance") or value.get("why_it_matters") or value.get("summary")
            ),
        }

    def get_active_case(self, sender_id: str, case_id: str = "default") -> Dict:
        """Get or create a case memory document."""
        cache_key = f"{sender_id}:{case_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        doc_id = f"{sender_id}:case:{case_id}"
        doc = self._read_doc(doc_id, sender_id)
        if doc:
            case = {k: v for k, v in doc.items() if not k.startswith("_") and k not in ("id", "sender_id", "doc_type", "ttl")}
        else:
            case = {
                "case_id": case_id,
                "case_name": "",
                "case_facts": [],
                "persons_mentioned": [],
                "documents_discussed": [],
                "open_questions": [],
                "key_contradictions": [],
                "timeline_events": [],
                "topic_history": [],
                "action_items": [],       # Task/Follow-Up Store: lawyer's to-dos and next steps
                "evidence_references": [], # Evidence Store: key quotes/findings with source metadata
                "turn_count": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }
        self._cache[cache_key] = case
        return case

    def persist_case(self, sender_id: str, case_id: str):
        """Immediately persist the current case state to Cosmos DB."""
        cache_key = f"{sender_id}:{case_id}"
        case = self._cache.get(cache_key)
        if not case:
            return
        doc = dict(case)
        doc.update({
            "id": f"{sender_id}:case:{case_id}",
            "sender_id": sender_id,
            "doc_type": "case_memory",
            "ttl": self.TTL_SECONDS,
        })
        self._upsert_doc(doc)

    def update_case_from_extraction(self, sender_id: str, case_id: str, extracted: Dict):
        """Merge extracted facts from a conversation turn into the case memory."""
        case = self.get_active_case(sender_id, case_id)

        # Normalize legacy / malformed values before dedup logic runs.
        case["case_facts"] = [self._normalize_text_item(v) for v in case.get("case_facts", []) if self._normalize_text_item(v)]
        case["persons_mentioned"] = [self._normalize_text_item(v) for v in case.get("persons_mentioned", []) if self._normalize_text_item(v)]
        case["documents_discussed"] = [self._normalize_text_item(v) for v in case.get("documents_discussed", []) if self._normalize_text_item(v)]
        case["open_questions"] = [self._normalize_text_item(v) for v in case.get("open_questions", []) if self._normalize_text_item(v)]
        case["key_contradictions"] = [self._normalize_text_item(v) for v in case.get("key_contradictions", []) if self._normalize_text_item(v)]
        case["timeline_events"] = [ev for ev in (self._normalize_timeline_event(v) for v in case.get("timeline_events", [])) if ev]
        case["action_items"] = [item for item in (self._normalize_action_item(v) for v in case.get("action_items", [])) if item]
        case["evidence_references"] = [ev for ev in (self._normalize_evidence_reference(v) for v in case.get("evidence_references", [])) if ev]

        # Merge facts (deduplicate)
        existing_facts = set(case.get("case_facts", []))
        for fact in extracted.get("case_facts", []):
            fact = self._normalize_text_item(fact)
            if fact and fact not in existing_facts:
                case["case_facts"].append(fact)
                existing_facts.add(fact)
        # Cap at 200 facts
        if len(case["case_facts"]) > 200:
            case["case_facts"] = case["case_facts"][-200:]

        # Merge persons (deduplicate)
        existing_persons = set(p.lower() for p in case.get("persons_mentioned", []))
        for person in extracted.get("persons_mentioned", []):
            person = self._normalize_text_item(person)
            if person and person.lower() not in existing_persons:
                case["persons_mentioned"].append(person)
                existing_persons.add(person.lower())

        # Merge documents (deduplicate)
        existing_docs = set(d.lower() for d in case.get("documents_discussed", []))
        for doc in extracted.get("documents_discussed", []):
            doc = self._normalize_text_item(doc)
            if doc and doc.lower() not in existing_docs:
                case["documents_discussed"].append(doc)
                existing_docs.add(doc.lower())

        # Merge open questions (deduplicate, also remove answered ones)
        for q in extracted.get("open_questions", []):
            q = self._normalize_text_item(q)
            if q and q not in case.get("open_questions", []):
                case["open_questions"].append(q)
        # Cap open questions
        if len(case["open_questions"]) > 50:
            case["open_questions"] = case["open_questions"][-50:]

        # Merge contradictions
        for c in extracted.get("key_contradictions", []):
            c = self._normalize_text_item(c)
            if c and c not in case.get("key_contradictions", []):
                case["key_contradictions"].append(c)

        # Merge timeline events
        existing_events = set(json.dumps(e, sort_keys=True) for e in case.get("timeline_events", []))
        for event in extracted.get("timeline_events", []):
            event = self._normalize_timeline_event(event)
            if not event:
                continue
            event_key = json.dumps(event, sort_keys=True)
            if event_key not in existing_events:
                case["timeline_events"].append(event)
                existing_events.add(event_key)

        # Update topic history
        topic_summary = extracted.get("topic_summary", "")
        if topic_summary:
            case["topic_history"].append({
                "summary": topic_summary,
                "timestamp": datetime.now().isoformat(),
            })
            # Keep last 50 topic entries
            if len(case["topic_history"]) > 50:
                case["topic_history"] = case["topic_history"][-50:]

        # Merge action items / follow-ups (Task Store)
        if "action_items" not in case:
            case["action_items"] = []
        for item in extracted.get("action_items", []):
            item = self._normalize_action_item(item)
            if item and item.get("task"):
                # Deduplicate by task text
                existing_tasks = {a.get("task", "").lower() for a in case["action_items"]}
                if item["task"].lower() not in existing_tasks:
                    item["created"] = item.get("created") or datetime.now().isoformat()
                    case["action_items"].append(item)
        # Cap action items
        if len(case["action_items"]) > 50:
            case["action_items"] = case["action_items"][-50:]

        # Merge evidence references (Evidence Store)
        if "evidence_references" not in case:
            case["evidence_references"] = []
        for ev in extracted.get("evidence_references", []):
            ev = self._normalize_evidence_reference(ev)
            if ev and ev.get("quote"):
                # Deduplicate by quote text (first 80 chars)
                existing_quotes = {e.get("quote", "")[:80].lower() for e in case["evidence_references"]}
                if ev["quote"][:80].lower() not in existing_quotes:
                    case["evidence_references"].append(ev)
        # Cap evidence
        if len(case["evidence_references"]) > 100:
            case["evidence_references"] = case["evidence_references"][-100:]

        case["turn_count"] = case.get("turn_count", 0) + 1
        case["last_updated"] = datetime.now().isoformat()

        # Persist
        doc = dict(case)
        doc.update({
            "id": f"{sender_id}:case:{case_id}",
            "sender_id": sender_id,
            "doc_type": "case_memory",
            "ttl": self.TTL_SECONDS,
        })
        self._upsert_doc(doc)
        self._cache[f"{sender_id}:{case_id}"] = case

    def get_case_context_summary(self, sender_id: str, case_id: str = "default") -> str:
        """
        Case Workspace Loader: Build a full context summary combining all 4 memory stores
        (conversation, case facts, tasks/follow-ups, evidence) for injection into conversation.
        """
        case = self.get_active_case(sender_id, case_id)
        if not case.get("case_facts") and not case.get("topic_history"):
            return ""

        parts = []
        case_name = case.get("case_name", "")
        if case_name:
            parts.append(f"Active case: {case_name}")

        # ── Case Facts Store ──
        facts = case.get("case_facts", [])
        if facts:
            parts.append("Known facts from previous sessions:")
            for i, fact in enumerate(facts[-15:], 1):
                parts.append(f"  {i}. {fact}")

        persons = case.get("persons_mentioned", [])
        if persons:
            parts.append(f"Key persons: {', '.join(persons[-10:])}")

        docs = case.get("documents_discussed", [])
        if docs:
            parts.append(f"Documents previously discussed: {', '.join(docs[-10:])}")

        # ── Evidence Store ──
        evidence = case.get("evidence_references", [])
        if evidence:
            parts.append("Key evidence from previous sessions:")
            for ev in evidence[-8:]:
                quote = ev.get("quote", "")[:150]
                source = ev.get("source_file", "unknown")
                relevance = ev.get("relevance", "")
                parts.append(f"  - \"{quote}\" (from: {source}){' — ' + relevance if relevance else ''}")

        contradictions = case.get("key_contradictions", [])
        if contradictions:
            parts.append("Contradictions found:")
            for c in contradictions[-5:]:
                parts.append(f"  - {c}")

        timeline = case.get("timeline_events", [])
        if timeline:
            parts.append("Timeline of events:")
            for ev in timeline[-8:]:
                date = ev.get("date", "?")
                event = ev.get("event", "")
                parts.append(f"  - {date}: {event}")

        # ── Task/Follow-Up Store ──
        action_items = case.get("action_items", [])
        pending_items = [a for a in action_items if a.get("status", "pending") == "pending"]
        if pending_items:
            parts.append("Pending action items / follow-ups:")
            for item in pending_items[-8:]:
                task = item.get("task", "")
                priority = item.get("priority", "")
                parts.append(f"  - {task}{' [' + priority + ']' if priority else ''}")

        open_qs = case.get("open_questions", [])
        if open_qs:
            parts.append("Open questions from previous sessions:")
            for q in open_qs[-5:]:
                parts.append(f"  - {q}")

        # ── Conversation Topic History ──
        topics = case.get("topic_history", [])
        if topics:
            parts.append("Recent topics discussed:")
            for t in topics[-5:]:
                parts.append(f"  - {t.get('summary', '')} ({t.get('timestamp', '')[:10]})")

        return "\n".join(parts)

    def list_active_cases(self, sender_id: str) -> List[Dict]:
        """List all active cases for a sender. Used by session resolver."""
        try:
            query = "SELECT * FROM c WHERE c.sender_id = @sid AND c.doc_type = 'case_memory'"
            params = [{"name": "@sid", "value": sender_id}]
            items = list(self._container.query_items(query=query, parameters=params, partition_key=sender_id))
            cases = []
            for item in items:
                cases.append({
                    "case_id": item.get("case_id", ""),
                    "case_name": item.get("case_name", ""),
                    "turn_count": item.get("turn_count", 0),
                    "last_updated": item.get("last_updated", ""),
                    "persons": item.get("persons_mentioned", [])[:5],
                    "topic_summary": (item.get("topic_history", []) or [{}])[-1].get("summary", "") if item.get("topic_history") else "",
                })
            return sorted(cases, key=lambda c: c.get("last_updated", ""), reverse=True)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to list cases: {e}")
            return []


# ============================================================================
# WhatsApp Handler
# ============================================================================

class WhatsAppHandler:
    def __init__(
        self,
        green_api: GreenAPIClient,
        rag_backend: RAGBackendClient,
        transcriber: WhisperTranscriber,
        gpt_client: Optional[AzureOpenAI] = None,
        gpt_deployment: str = "",
        memory: Optional[Any] = None,
        case_memory: Optional[CaseMemoryStore] = None,
    ):
        self.green_api = green_api
        self.rag_backend = rag_backend
        self.transcriber = transcriber
        self.memory = memory or ConversationMemory()
        self.case_memory = case_memory
        self.gpt_client = gpt_client
        self.gpt_deployment = gpt_deployment

    def _extract_email_request(self, query: str) -> Optional[str]:
        import re

        email_match = re.search(r'[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}', query or "")
        if not email_match:
            return None

        lowered = (query or "").lower()
        email_terms = [
            "email", "mail", "send", "envoyer", "envoie",
            "bhejo", "bhej", "forward", "share"
        ]
        if any(term in lowered for term in email_terms):
            return email_match.group(0)
        return None

    def _wants_email_action(self, query: str) -> bool:
        lowered = (query or "").lower()
        email_terms = [
            "email", "mail", "send", "envoyer", "envoie",
            "bhejo", "bhej", "forward", "share"
        ]
        return any(term in lowered for term in email_terms)

    def _resolve_email_payload(self, sender_phone: str, reply_context: Optional[Dict]) -> Optional[Dict[str, Any]]:
        if reply_context:
            return {
                "answer": reply_context.get("content", "") or reply_context.get("full_content", ""),
                "formatted_body": reply_context.get("formatted_content", "") or reply_context.get("full_formatted_content", ""),
                "query": reply_context.get("query", ""),
                "sources": reply_context.get("sources", []) or [],
            }

        state = self.memory.get_state(sender_phone)
        answer = state.get("last_answer", "")
        query = state.get("last_query", "")
        sources = state.get("last_sources", []) or []
        if not answer:
            return None
        return {
            "answer": answer,
            "formatted_body": state.get("last_formatted_answer", "") or answer,
            "query": query,
            "sources": sources,
        }

    def _handle_email_request(
        self,
        chat_id: str,
        sender_phone: str,
        query: str,
        reply_context: Optional[Dict] = None,
    ) -> bool:
        recipient_email = self._extract_email_request(query)
        if not recipient_email and self._wants_email_action(query):
            self.green_api.send_text_message(
                chat_id,
                "📧 Please include the recipient email address, for example: send this to name@example.com"
            )
            return True
        if not recipient_email:
            return False

        payload = self._resolve_email_payload(sender_phone, reply_context)
        if not payload:
            self.green_api.send_text_message(
                chat_id,
                "🤖 I need an existing answer or source list before I can send an email. Reply to a previous bot message and try again."
            )
            return True

        selected_answer = payload.get("answer", "")
        if reply_context:
            selected_answer = self._extract_requested_reply_subset(query, reply_context)
            payload["answer"] = selected_answer
            payload["formatted_body"] = selected_answer

        try:
            self.rag_backend.send_email(
                recipient_email=recipient_email,
                answer=payload.get("answer", ""),
                query=payload.get("query", ""),
                sources=payload.get("sources", []),
                subject="Legal Assistant — Shared Answer",
                formatted_body=payload.get("formatted_body", ""),
            )
            self.green_api.send_text_message(
                chat_id,
                f"📧 The email has been sent to {recipient_email}."
            )
        except Exception:
            self.green_api.send_text_message(
                chat_id,
                "❌ An error occurred while sending the email. Check the backend email settings and ACS configuration."
            )
        return True

    def _extract_requested_reply_subset(self, query: str, reply_context: Dict[str, Any]) -> str:
        import re

        reply_text = (reply_context.get("content") or "").strip()
        full_text = (reply_context.get("full_content") or reply_text or "").strip()
        working_text = reply_text or full_text
        if not working_text:
            return ""

        lower = (query or "").lower()
        if any(phrase in lower for phrase in ["this message", "this msg", "is msg", "this paragraph", "this part"]):
            return working_text

        number_matches = re.findall(r"\b(?:statement|statements|paragraph|paragraphs|point|points|item|items)\s+((?:\d+\s*(?:,|and)?\s*)+)", lower)
        numbers: List[int] = []
        for match in number_matches:
            for num in re.findall(r"\d+", match):
                numbers.append(int(num))

        if not numbers:
            return working_text

        lines = [line.strip() for line in working_text.splitlines() if line.strip()]
        numbered_lines = []
        for line in lines:
            m = re.match(r"^(\d+)[\).\-\:]?\s+(.*)$", line)
            if m:
                numbered_lines.append((int(m.group(1)), line))

        if numbered_lines:
            selected = [line for idx, line in numbered_lines if idx in numbers]
            if selected:
                return "\n".join(selected)

        paragraphs = [p.strip() for p in working_text.split("\n\n") if p.strip()]
        selected_paragraphs = [paragraphs[i - 1] for i in numbers if 1 <= i <= len(paragraphs)]
        if selected_paragraphs:
            return "\n\n".join(selected_paragraphs)

        return working_text

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
    # GPT Conversation Router
    # ------------------------------------------------------------------

    def _route_conversation(
        self,
        query: str,
        sender_phone: str,
        reply_context: Optional[Dict] = None,
    ) -> Dict:
        """
        GPT-based conversation router. Classifies the user's message and decides
        how the system should handle it. Returns classification dict.
        """
        history = self.memory.get(sender_phone)
        state = self.memory.get_state(sender_phone)

        # Build history text for prompt
        history_text = ""
        if history:
            last_turns = history[-8:]  # last 4 turns
            history_text = "\n".join(
                f"{m['role'].upper()}: {m['content'][:300]}" for m in last_turns
            )

        # Build last sources text
        last_sources = state.get("last_sources", [])
        last_sources_text = ""
        if last_sources:
            for i, s in enumerate(last_sources, 1):
                fname = s.get("file_name", "unknown")
                doc_type = s.get("document_type", "")
                last_sources_text += f"{i}. {fname} (type: {doc_type})\n"

        # Build reply context text
        reply_context_text = ""
        if reply_context:
            reply_context_text = (
                f"Original query: {reply_context.get('query', '')}\n"
                f"Bot answer (first 500 chars): {reply_context.get('content', '')[:500]}"
            )

        prompt = prompts.get_conversation_router_prompt(
            query=query,
            history_text=history_text,
            last_sources_text=last_sources_text,
            reply_context_text=reply_context_text,
        )

        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            logger.info(f"Router: {result.get('classification')} | reason: {result.get('reasoning', '')[:80]}")
            return result
        except Exception as e:
            logger.error(f"Router GPT call failed: {e}")
            # Fallback: if history exists treat as follow-up deep, else fresh
            return {
                "classification": "FOLLOW_UP_DEEP" if history else "FRESH",
                "reasoning": "fallback due to router error",
                "effective_query": query,
                "topic": state.get("topic", ""),
            }

    def _extract_quoted_message_id(self, webhook_data: Dict[str, Any]) -> Optional[str]:
        """
        Try to extract the replied/quoted message ID from GreenAPI webhook payload.
        Different payload shapes may exist, so we check multiple common locations.
        """
        message_data = webhook_data.get("messageData", {})
        current_message_id = webhook_data.get("idMessage", "")

        def _collect_values(node: Any, keys: set[str]) -> List[str]:
            found: List[str] = []
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in keys and isinstance(value, str) and value:
                        found.append(value)
                    found.extend(_collect_values(value, keys))
            elif isinstance(node, list):
                for item in node:
                    found.extend(_collect_values(item, keys))
            return found

        preferred_nodes = [
            message_data.get("quotedMessage", {}),
            message_data.get("quotedMessageData", {}),
            message_data.get("extendedTextMessageData", {}).get("quotedMessage", {}),
            message_data.get("extendedTextMessageData", {}).get("contextInfo", {}),
            message_data.get("textMessageData", {}).get("quotedMessage", {}),
            message_data.get("textMessageData", {}).get("contextInfo", {}),
            webhook_data.get("quotedMessage", {}),
            webhook_data.get("quotedMessageData", {}),
        ]

        candidate_keys = {"quotedMessageId", "quotedStanzaId", "stanzaId"}
        candidates: List[str] = []

        for node in preferred_nodes:
            candidates.extend(_collect_values(node, candidate_keys))

        candidates.extend(
            [
                message_data.get("extendedTextMessageData", {}).get("quotedMessageId"),
                message_data.get("textMessageData", {}).get("quotedMessageId"),
                webhook_data.get("quotedMessageId"),
            ]
        )

        for candidate in candidates:
            if candidate and candidate != current_message_id:
                return candidate

        return None
    
    # ------------------------------------------------------------------
    # Response Formatting
    # ------------------------------------------------------------------

    def _format_source_label(self, source: Dict) -> str:
        container = source.get("source_container", "")
        icon = "🔴" if container == "legal-documents-internal" else "📗"
        label = "Internal (Confidential)" if container == "legal-documents-internal" else "Legal Documents"
        fname = source.get("file_name", "Unknown")
        return f"{icon} {fname} [{label}]"

    def _format_inline_sections(self, sections: List[Dict]) -> str:
        blocks = []
        for section in sections:
            text = (section.get("text") or "").strip()
            if not text:
                continue

            block = text
            source_url = section.get("source_url", "")
            if source_url:
                block += (
                    f"\nSource: {self._format_source_label(section)}"
                    f"\nLink: {source_url}"
                )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _format_response(
        self,
        answer: str,
        sources: List[Dict],
        sections: Optional[List[Dict]] = None,
        query_was_voice: bool = False
    ) -> str:
        """
        Format the AI answer for WhatsApp using plain text.
        """
        prefix = "Voice query processed\n\n" if query_was_voice else ""

        header = "Legal AI Assistant\n" + "─" * 28 + "\n\n"
        if sections:
            body = self._format_inline_sections(sections)
        else:
            body = answer.strip()

        # Sources section (all unique sources)
        sources_text = ""
        if sources and not sections:
            sources_text = "\n\n" + "─" * 28 + "\nSources:\n"
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
                sources_text += f"   {label}\n"
                if source_url:
                    sources_text += f"   Link: {source_url}\n"

        footer = "\n\nReply to this message or tag @" + self.green_api.config.bot_name + " to ask a follow-up."

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

            # ── Extract quoted/reply context BEFORE bot mention check ──
            quoted_message_id = self._extract_quoted_message_id(webhook_data)
            reply_context = None
            is_reply_to_bot = False

            if quoted_message_id:
                reply_context = self.memory.get_bot_message_by_id(sender_phone, quoted_message_id)
                if reply_context:
                    is_reply_to_bot = True
                    logger.info(f"Reply context found for quoted bot message: {quoted_message_id}")
                else:
                    logger.info(f"Quoted message ID found but no bot context stored: {quoted_message_id}")

            # ── GUARD 3: Bot must be mentioned in text (groups only, DMs skip this) ──
            # Exception: replies to bot messages always pass through
            if not is_dm and not is_reply_to_bot and not self._is_bot_mentioned(text, message_data):
                logger.info(f"Bot not mentioned, skipping: '{text[:40]}'")
                return

            # Handle special commands
            cleaned = self._clean_query(text)
            lower = cleaned.lower().strip()

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

    def _query_and_respond(
        self,
        chat_id: str,
        sender_phone: str,
        sender_name: str,
        query: str,
        is_voice: bool = False,
        reply_context: Optional[Dict] = None
    ) -> None:
        """Send query to RAG backend with GPT-routed conversation intelligence."""
        if not query:
            return

        try:
            if self._handle_email_request(chat_id, sender_phone, query, reply_context):
                return

            # ── Step 1: GPT Conversation Router ──
            routing = self._route_conversation(query, sender_phone, reply_context)
            classification = routing.get("classification", "FRESH")
            effective_query = routing.get("effective_query") or query
            topic = routing.get("topic", "")

            conv_state = self.memory.get_state(sender_phone)

            # ── Step 2: Handle special classifications ──

            # CHITCHAT — respond without querying RAG
            if classification == "CHITCHAT":
                chitchat_resp = routing.get("chitchat_response", "Hello! Ask me anything about the legal documents.")
                self.green_api.send_text_message(chat_id, f"🤖 {chitchat_resp}")
                return

            # VAGUE — ask for clarification
            if classification == "VAGUE":
                clarification = routing.get(
                    "clarification_message",
                    "Could you be more specific? For example:\n"
                    "• _Find all emails from Jean Tremblay_\n"
                    "• _What is the loan amount for DP-0372?_\n"
                    "• _Show me contracts for project Couvent_"
                )
                self.green_api.send_text_message(chat_id, f"🤖 {clarification}")
                return

            # ── Step 2.5: Load case memory context (Memory Agent) ──
            case_context = self._load_case_context(sender_phone, effective_query)

            # ── Step 3: Build RAG parameters based on classification ──
            full_history = self.memory.get(sender_phone)
            history = []
            exclude_blob_paths = []

            # If we have case memory, inject it as a system-level context message
            # at the start of history so RAG backend sees prior case knowledge
            case_context_msg = []
            if case_context:
                case_context_msg = [{
                    "role": "user",
                    "content": f"[CASE CONTEXT FROM PREVIOUS SESSIONS]\n{case_context}\n[END CASE CONTEXT]",
                }, {
                    "role": "assistant",
                    "content": "Understood. I have the prior case context and will use it to inform my answers.",
                }]

            if classification == "FRESH":
                # New topic — clear previous shown docs
                # But keep case memory context if available
                self.memory.clear_shown_blob_paths(sender_phone)
                history = case_context_msg  # Fresh query but with case memory
                logger.info(f"FRESH query | topic: {topic} | case_context: {len(case_context)} chars")

            elif classification == "FOLLOW_UP_MORE":
                # User wants more of same → exclude already-shown docs
                exclude_blob_paths = self.memory.get_shown_blob_paths(sender_phone)
                history = case_context_msg + full_history[-4:]  # case context + last 2 turns
                logger.info(f"FOLLOW_UP_MORE | excluding {len(exclude_blob_paths)} shown docs | effective: '{effective_query[:60]}'")

            elif classification == "FOLLOW_UP_DEEP":
                # Deeper question on same topic — full history context + case memory
                history = case_context_msg + full_history[-6:]  # case context + last 3 turns
                logger.info(f"FOLLOW_UP_DEEP | history: {len(history)} msgs | effective: '{effective_query[:60]}'")

            elif classification == "FOLLOW_UP_DOC":
                # Specific document from previous answer
                doc_index = routing.get("referenced_doc_index")
                doc_name = routing.get("referenced_doc_name")
                last_sources = conv_state.get("last_sources", [])

                allowed_file = None
                if doc_index and doc_index <= len(last_sources):
                    allowed_file = last_sources[doc_index - 1].get("blob_path", "")
                elif doc_name:
                    # Find by name match
                    for s in last_sources:
                        if doc_name.lower() in s.get("file_name", "").lower():
                            allowed_file = s.get("blob_path", "")
                            break

                if allowed_file:
                    logger.info(f"FOLLOW_UP_DOC | specific doc: {allowed_file}")
                    response = self.rag_backend.chat(
                        query=effective_query,
                        history=case_context_msg + full_history[-4:],
                        conversation_id=sender_phone,
                        allowed_files=[allowed_file],
                    )
                    self._send_rag_response(
                        chat_id, sender_phone, query, effective_query,
                        response, is_voice, topic
                    )
                    return
                else:
                    # Couldn't find specific doc — fall through to normal search
                    history = case_context_msg + full_history[-4:]
                    logger.info(f"FOLLOW_UP_DOC | doc not found, falling back to normal search")

            # ── Step 4: Query RAG ──
            logger.info(f"Querying RAG: '{effective_query[:60]}' | class={classification} | history={len(history)} | exclude={len(exclude_blob_paths)}")

            response = self.rag_backend.chat(
                query=effective_query,
                history=history,
                conversation_id=sender_phone,
                exclude_blob_paths=exclude_blob_paths,
            )

            # ── Step 5: Handle "no new results" for FOLLOW_UP_MORE ──
            sources = response.get("sources", [])
            if classification == "FOLLOW_UP_MORE" and not sources:
                self.green_api.send_text_message(
                    chat_id,
                    "🤖 I've already shared all the documents I found on this topic. "
                    "Try asking a more specific question or a different topic."
                )
                self.memory.add(sender_phone, "user", query)
                self.memory.add(sender_phone, "assistant", "(no new documents available)")
                return

            # ── Step 6: Send response ──
            self._send_rag_response(
                chat_id, sender_phone, query, effective_query,
                response, is_voice, topic
            )

        except Exception as e:
            logger.error(f"RAG error: {e}", exc_info=True)
            self.green_api.send_text_message(
                chat_id,
                "❌ An error occurred while processing your request. Please try again."
            )

    def _send_rag_response(
        self,
        chat_id: str,
        sender_phone: str,
        original_query: str,
        effective_query: str,
        response: Dict,
        is_voice: bool,
        topic: str,
    ) -> None:
        """Format RAG response, send to WhatsApp, update memory & state."""
        answer = response.get("answer", "I could not find an answer. Please try rephrasing.")
        sources = response.get("sources", [])
        sections = response.get("sections", [])

        # Update conversation history
        self.memory.add(sender_phone, "user", original_query)
        self.memory.add(sender_phone, "assistant", answer)

        # Track shown documents
        new_blob_paths = [s.get("blob_path", "") for s in sources if s.get("blob_path")]
        self.memory.add_shown_blob_paths(sender_phone, new_blob_paths)

        # Format and send
        formatted = self._format_response(
            answer,
            sources,
            sections=sections,
            query_was_voice=is_voice
        )

        # Update conversation state
        self.memory.update_state(
            sender_phone,
            topic=topic,
            last_query=effective_query,
            last_sources=sources,
            last_answer=answer,
            last_formatted_answer=formatted,
        )

        send_result = self.green_api.send_text_message(chat_id, formatted)

        # Store bot message ID(s) for reply tracking (supports multi-message split)
        all_ids = send_result.get("allMessageIds", [])
        sent_parts = send_result.get("sentParts", [])
        if not all_ids:
            sent_message_id = send_result.get("idMessage", "")
            all_ids = [sent_message_id] if sent_message_id else []
            sent_parts = [formatted] if sent_message_id else []
        message_ids = [mid for mid in all_ids if mid]

        self.memory.save_bot_message(
            sender_id=sender_phone,
            whatsapp_message_ids=message_ids,
            answer=answer,
            query=effective_query,
            sources=sources,
            message_parts=sent_parts,
            formatted_answer=formatted,
        )

        logger.info(f"Response sent to {chat_id} | sources: {len(sources)} | topic: {topic}")

        # ── Memory Update Agent: extract and persist case facts ──
        if self.case_memory and self.gpt_client:
            try:
                self._run_memory_update(sender_phone, effective_query, answer, sources)
            except Exception as e:
                logger.error(f"Memory update agent failed (non-blocking): {e}")

    # ------------------------------------------------------------------
    # Memory Update Agent
    # ------------------------------------------------------------------

    def _run_memory_update(self, sender_phone: str, query: str, answer: str, sources: List[Dict]):
        """
        Memory Update Agent: After each RAG response, extract case facts,
        persons, contradictions, timeline events, and open questions.
        Persist to CaseMemoryStore so the lawyer can resume days later.
        """
        if not self.case_memory or not self.gpt_client:
            return

        # Build sources summary for the extraction prompt
        sources_summary = ""
        if sources:
            for i, s in enumerate(sources[:6], 1):
                fname = s.get("file_name", "unknown")
                container = s.get("source_container", "")
                label = "Internal" if "internal" in container else "External"
                sources_summary += f"{i}. {fname} [{label}]\n"

        # Call GPT to extract structured facts
        prompt = prompts.get_extract_case_memory_prompt(query, answer, sources_summary)
        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            extracted = self._parse_memory_extraction(raw)
            logger.info(
                f"Memory extraction: {len(extracted.get('case_facts', []))} facts, "
                f"{len(extracted.get('persons_mentioned', []))} persons, "
                f"{len(extracted.get('open_questions', []))} open questions"
            )
        except Exception as e:
            logger.error(f"Memory extraction GPT call failed: {e}")
            return

        # Resolve which case this belongs to
        case_id = self._resolve_case_id(sender_phone, query)

        # Persist extracted facts
        self.case_memory.update_case_from_extraction(sender_phone, case_id, extracted)
        logger.info(f"Case memory updated | sender={sender_phone[:10]}... | case={case_id}")

    @staticmethod
    def _parse_memory_extraction(raw: str) -> Dict[str, Any]:
        """Parse memory extraction JSON with a small repair fallback for malformed model output."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            candidate = raw[start:end + 1]
            parsed = json.loads(candidate)

        defaults = {
            "case_facts": [],
            "persons_mentioned": [],
            "documents_discussed": [],
            "open_questions": [],
            "topic_summary": "",
            "key_contradictions": [],
            "timeline_events": [],
            "action_items": [],
            "evidence_references": [],
        }
        if not isinstance(parsed, dict):
            raise ValueError("Memory extraction must be a JSON object")

        normalized = dict(defaults)
        normalized.update(parsed)
        for key in ("case_facts", "persons_mentioned", "documents_discussed", "open_questions",
                    "key_contradictions", "timeline_events", "action_items", "evidence_references"):
            if not isinstance(normalized.get(key), list):
                value = normalized.get(key)
                normalized[key] = [] if value in (None, "") else [value]
        if not isinstance(normalized.get("topic_summary"), str):
            normalized["topic_summary"] = str(normalized.get("topic_summary", "")).strip()
        return normalized

    def _resolve_case_id(self, sender_phone: str, query: str) -> str:
        """
        Session/Case Resolver: determine which case this query belongs to.
        For now, uses GPT to match against active cases. Falls back to 'default'.
        """
        if not self.case_memory or not self.gpt_client:
            return "default"

        active_cases = self.case_memory.list_active_cases(sender_phone)
        if not active_cases:
            return "default"

        # If only one case, continue it
        if len(active_cases) == 1:
            return active_cases[0].get("case_id", "default")

        # Multiple cases: ask GPT to resolve
        cases_summary = ""
        for c in active_cases[:5]:
            cases_summary += (
                f"- Case ID: {c.get('case_id')} | Name: {c.get('case_name', 'unnamed')} | "
                f"Last active: {c.get('last_updated', '')[:10]} | "
                f"Persons: {', '.join(c.get('persons', []))} | "
                f"Topic: {c.get('topic_summary', '')}\n"
            )

        prompt = prompts.get_session_case_resolver_prompt(query, cases_summary)
        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            if result.get("case_action") == "START_NEW":
                # Generate a new case ID
                import hashlib
                new_id = hashlib.md5(f"{sender_phone}:{datetime.now().isoformat()}".encode()).hexdigest()[:8]
                case_name = result.get("suggested_case_name", "")
                case = self.case_memory.get_active_case(sender_phone, new_id)
                if case_name:
                    case["case_name"] = case_name
                # Persist immediately so the case survives even if extraction fails later
                self.case_memory.persist_case(sender_phone, new_id)
                logger.info(f"Session resolver: START_NEW case {new_id} ({case_name})")
                return new_id

            resolved_id = result.get("case_id")
            if resolved_id and any(c.get("case_id") == resolved_id for c in active_cases):
                logger.info(f"Session resolver: CONTINUE case {resolved_id}")
                return resolved_id

        except Exception as e:
            logger.error(f"Session resolver failed: {e}")

        # Fallback: most recently active case
        return active_cases[0].get("case_id", "default")

    # ------------------------------------------------------------------
    # Memory Agent (Load at Query Start)
    # ------------------------------------------------------------------

    def _load_case_context(self, sender_phone: str, query: str) -> str:
        """
        Memory Agent: Load relevant case memory to enrich the conversation.
        Returns a context string to prepend to history, or empty string if no memory.
        """
        if not self.case_memory:
            return ""

        case_id = self._resolve_case_id(sender_phone, query)
        context = self.case_memory.get_case_context_summary(sender_phone, case_id)

        if context:
            logger.info(f"Case memory loaded for {sender_phone[:10]}... | case={case_id} | {len(context)} chars")
        return context

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


def _init_cosmos_memory():
    """Initialize Cosmos DB-backed memory if configured. Returns (memory, case_memory) or (None, None)."""
    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "")
    cosmos_key = os.environ.get("COSMOS_KEY", "")
    memory_backend = os.environ.get("MEMORY_BACKEND", "inmemory").lower()

    if memory_backend != "cosmos" or not cosmos_endpoint or not cosmos_key:
        logger.info(f"Memory backend: in-memory (MEMORY_BACKEND={memory_backend})")
        return None, None

    if not COSMOS_AVAILABLE:
        logger.warning("MEMORY_BACKEND=cosmos but azure-cosmos package not installed. Falling back to in-memory.")
        return None, None

    try:
        cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
        db_name = os.environ.get("COSMOS_DATABASE", "legal-assistant")

        # Create database if not exists
        database = cosmos_client.create_database_if_not_exists(id=db_name)

        # Conversations container (history, state, bot messages)
        conv_container = database.create_container_if_not_exists(
            id="conversations",
            partition_key=PartitionKey(path="/sender_id"),
            default_ttl=-1,  # Per-document TTL enabled
        )

        # Case memory container (facts, findings, open questions)
        case_container = database.create_container_if_not_exists(
            id="case_memory",
            partition_key=PartitionKey(path="/sender_id"),
            default_ttl=-1,
        )

        memory = CosmosConversationMemory(conv_container)
        case_memory = CaseMemoryStore(case_container)
        logger.info(f"Cosmos DB memory initialized | db={db_name} | endpoint={cosmos_endpoint[:40]}...")
        return memory, case_memory

    except Exception as e:
        logger.error(f"Cosmos DB initialization failed: {e}. Falling back to in-memory.")
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whatsapp_handler

    green_config = GreenAPIConfig.from_env()
    green_api = GreenAPIClient(green_config)
    rag_backend = RAGBackendClient(
        base_url=os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")
    )
    transcriber = WhisperTranscriber()

    # GPT client for conversation routing (same endpoint as RAG backend uses)
    gpt_client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("OPENAI_KEY", ""),
        api_version="2024-02-01",
    )
    gpt_deployment = os.environ.get("OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    # Initialize persistent memory (Cosmos DB) or fall back to in-memory
    memory, case_memory = _init_cosmos_memory()

    whatsapp_handler = WhatsAppHandler(
        green_api, rag_backend, transcriber,
        gpt_client=gpt_client,
        gpt_deployment=gpt_deployment,
        memory=memory,
        case_memory=case_memory,
    )
    backend_type = "Cosmos DB" if memory else "in-memory"
    logger.info(f"WhatsApp handler initialized (v3.0 + Persistent Memory [{backend_type}])")

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
