"""
Sprint 4: WhatsApp Integration with GreenAPI
Webhook handler for receiving and responding to WhatsApp messages
"""
import os
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GreenAPIConfig:
    api_url: str  # e.g., https://7101.api.greenapi.com
    id_instance: str
    api_token_instance: str
    allowed_group_id: str  # Only respond in this group
    bot_name: str  # Bot name to detect mentions
    
    @classmethod
    def from_env(cls) -> "GreenAPIConfig":
        return cls(
            api_url=os.environ.get("GREENAPI_URL", "https://7101.api.greenapi.com"),
            id_instance=os.environ["GREENAPI_INSTANCE_ID"],
            api_token_instance=os.environ["GREENAPI_TOKEN"],
            allowed_group_id=os.environ.get("ALLOWED_GROUP_ID", ""),
            bot_name=os.environ.get("BOT_NAME", "AI Assistant"),
        )


# ============================================================================
# Pydantic Models for GreenAPI Webhooks
# ============================================================================

class GreenAPIMessageData(BaseModel):
    typeWebhook: str
    senderData: Dict[str, Any]
    messageData: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None
    idMessage: Optional[str] = None


class GreenAPIStatusData(BaseModel):
    typeWebhook: str
    statusData: Optional[Dict[str, Any]] = None


# ============================================================================
# GreenAPI Client
# ============================================================================

class GreenAPIClient:
    def __init__(self, config: GreenAPIConfig):
        self.config = config
        self.base_url = f"{config.api_url}/waInstance{config.id_instance}"
        
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make request to GreenAPI"""
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
        """Send text message to WhatsApp"""
        data = {
            "chatId": chat_id,
            "message": message
        }
        return self._make_request("POST", "sendMessage", data)
    
    def send_file_by_url(self, chat_id: str, url_file: str, file_name: str, caption: str = "") -> Dict:
        """Send file by URL to WhatsApp"""
        data = {
            "chatId": chat_id,
            "urlFile": url_file,
            "fileName": file_name,
            "caption": caption
        }
        return self._make_request("POST", "sendFileByUrl", data)
    
    def download_file(self, url: str) -> bytes:
        """Download voice note or file from WhatsApp"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"File download failed: {e}")
            raise


# ============================================================================
# RAG Backend Client (for chatting with AI)
# ============================================================================

class RAGBackendClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Send query to RAG backend and get answer"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query, "conversation_id": None},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG backend request failed: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Search documents"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG backend search failed: {e}")
            raise


# ============================================================================
# WhatsApp Message Handler
# ============================================================================

class WhatsAppHandler:
    def __init__(self, green_api: GreenAPIClient, rag_backend: RAGBackendClient):
        self.green_api = green_api
        self.rag_backend = rag_backend
        self.conversation_history: Dict[str, list] = {}  # Simple in-memory storage
    
    def _get_conversation_key(self, chat_id: str, sender: str) -> str:
        return f"{chat_id}:{sender}"
    
    def _extract_text_from_message(self, message_data: Dict) -> Optional[str]:
        """Extract text from different message types"""
        if not message_data:
            return None
        
        type_message = message_data.get("typeMessage", "")
        
        if type_message == "textMessage":
            return message_data.get("textMessageData", {}).get("textMessage", "")
        
        elif type_message == "extendedTextMessage":
            return message_data.get("extendedTextMessageData", {}).get("text", "")
        
        elif type_message == "documentMessage":
            # Document received
            caption = message_data.get("fileMessageData", {}).get("caption", "")
            return caption or "[Document received]"
        
        elif type_message == "imageMessage":
            caption = message_data.get("fileMessageData", {}).get("caption", "")
            return caption or "[Image received]"
        
        return None
    
    def _is_voice_message(self, message_data: Dict) -> bool:
        """Check if message is a voice note"""
        if not message_data:
            return False
        type_message = message_data.get("typeMessage", "")
        return type_message == "voiceMessage" or type_message == "audioMessage"
    
    def _transcribe_voice(self, download_url: str) -> str:
        """Transcribe voice note using Whisper (OpenAI)"""
        # TODO: Implement voice transcription
        # For now, return placeholder
        return "[Voice message - transcription pending]"
    
    def _format_answer_with_sources(self, answer: str, sources: list) -> str:
        """Format AI answer with source references for WhatsApp"""
        formatted = f"🤖 *AI Legal Assistant*\n\n{answer}\n\n"
        
        if sources:
            formatted += "*📚 Sources:*\n"
            for i, source in enumerate(sources[:3], 1):  # Top 3 sources
                file_name = source.get('file_name', 'Unknown')
                folder = source.get('folder_path', '')
                source_url = source.get('source_url', '')
                source_container = source.get('source_container', 'unknown')
                
                # Add source icon based on container
                source_icon = "🔒" if source_container == "internal" else "📄"
                source_label = "Internal" if source_container == "internal" else "Legal Docs"
                
                formatted += f"{i}. {source_icon} `{file_name}`\n"
                formatted += f"   🏷️ {source_label}\n"
                if folder:
                    formatted += f"   📁 {folder}\n"
                if source_url:
                    # Truncate URL for WhatsApp display
                    formatted += f"   🔗 [Download PDF]({source_url})\n"
        
        return formatted
    
    def _is_from_allowed_group(self, chat_id: str) -> bool:
        """Check if message is from the allowed group only (not private chats)"""
        # Must be a group (ends with @g.us) AND match allowed group ID
        is_group = chat_id.endswith("@g.us")
        is_allowed = chat_id == self.green_api.config.allowed_group_id
        
        logger.info(f"Received message from: {chat_id}")
        logger.info(f"Allowed group ID: {self.green_api.config.allowed_group_id}")
        logger.info(f"Is group chat: {is_group}, Is allowed: {is_allowed}")
        
        # STRICT: Only allow if it's the exact allowed group
        if not is_group:
            logger.info(f"Ignoring private chat: {chat_id}")
            return False
        
        if not is_allowed:
            logger.info(f"Ignoring message from non-allowed group: {chat_id}")
            return False
        
        return True
    
    def _is_bot_mentioned(self, text: str, chat_id: str) -> bool:
        """Check if bot is mentioned/tag in the message"""
        # In private chats (not groups), always respond
        if not chat_id.endswith("@g.us"):
            return True
        
        # In groups, check for mention/tag
        bot_name = self.green_api.config.bot_name.lower()
        text_lower = text.lower()
        
        # Check for @botname or "bot" or "assistant"
        mention_patterns = [
            f"@{bot_name}",
            "@ai", "@bot", "@assistant",
            "ai legal", "legal ai",
            "assistant", "help me", "question"
        ]
        
        for pattern in mention_patterns:
            if pattern in text_lower:
                return True
        
        logger.info(f"Bot not mentioned in group message: {text[:50]}")
        return False
    
    def _clean_query_text(self, text: str) -> str:
        """Remove bot mentions/tags from query text"""
        bot_name = self.green_api.config.bot_name.lower()
        
        # Remove @botname patterns
        patterns_to_remove = [
            f"@{bot_name}",
            f"@{bot_name.lower()}",
            "@ai", "@bot", "@assistant",
            "ai legal", "legal ai",
            "assistant", "help me"
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = cleaned.replace(pattern, "")
            cleaned = cleaned.replace(pattern.lower(), "")
        
        # Clean up extra spaces
        cleaned = " ".join(cleaned.split())
        
        return cleaned.strip()
    
    def handle_incoming_message(self, webhook_data: Dict[str, Any]) -> None:
        """Process incoming WhatsApp message"""
        try:
            type_webhook = webhook_data.get("typeWebhook", "")
            
            # Only process incoming and outgoing messages (from your account)
            if type_webhook not in ["incomingMessageReceived", "outgoingMessageReceived"]:
                logger.info(f"Ignoring webhook type: {type_webhook}")
                return
            
            # For outgoing messages, only process if from allowed group (your messages)
            if type_webhook == "outgoingMessageReceived":
                logger.info("Processing outgoing message (from your account)")
            
            # Extract sender info
            sender_data = webhook_data.get("senderData", {})
            chat_id = sender_data.get("chatId", "")
            sender_name = sender_data.get("senderName", "User")
            sender_phone = sender_data.get("sender", "")
            
            logger.info(f"Message from {sender_name} ({chat_id})")
            
            # Extract message data
            message_data = webhook_data.get("messageData", {})
            
            # Handle voice messages
            if self._is_voice_message(message_data):
                logger.info("Voice message received")
                # TODO: Download and transcribe
                self.green_api.send_text_message(
                    chat_id, 
                    "🎤 Voice messages are not yet supported. Please send text."
                )
                return
            
            # Extract text
            text = self._extract_text_from_message(message_data)
            
            if not text:
                logger.warning("No text found in message")
                return
            
            # Check if from allowed group
            if not self._is_from_allowed_group(chat_id):
                return
            
            # Check if bot is mentioned (in groups)
            if not self._is_bot_mentioned(text, chat_id):
                return
            
            # Check for special commands
            lower_text = text.lower().strip()
            
            if lower_text in ["/start", "hello", "hi"]:
                welcome_msg = (
                    "👋 *Welcome to AI Legal Assistant!*\n\n"
                    "I can help you search through your legal documents.\n\n"
                    "*Examples:*\n"
                    "• Find cases about property fraud\n"
                    "• What was the judgment in case X?\n"
                    "• Search for documents about Article 19\n\n"
                    "How can I help you today?"
                )
                self.green_api.send_text_message(chat_id, welcome_msg)
                return
            
            if lower_text.startswith("/help"):
                help_msg = (
                    "*🤖 Available Commands:*\n\n"
                    "• *Ask questions* - Just type your query\n"
                    "• `/search <query>` - Search documents\n"
                    "• `/start` - Welcome message\n\n"
                    "*Features:*\n"
                    "• AI-powered search\n"
                    "• Source citations\n"
                    "• Document references\n"
                    "• Fast responses"
                )
                self.green_api.send_text_message(chat_id, help_msg)
                return
            
            # Send "typing..." indicator (optional, GreenAPI may not support this)
            
            # Query the RAG backend with cleaned text (remove mentions)
            try:
                cleaned_query = self._clean_query_text(text)
                logger.info(f"Cleaned query: {cleaned_query}")
                response = self.rag_backend.chat(cleaned_query)
                answer = response.get("answer", "I couldn't find an answer to that question.")
                sources = response.get("sources", [])
                
                # Format and send response
                formatted_answer = self._format_answer_with_sources(answer, sources)
                self.green_api.send_text_message(chat_id, formatted_answer)
                
                logger.info(f"Response sent to {chat_id}")
                
            except Exception as e:
                logger.error(f"RAG backend error: {e}")
                self.green_api.send_text_message(
                    chat_id,
                    "❌ Sorry, I encountered an error processing your request. Please try again."
                )
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")


# ============================================================================
# FastAPI App for Webhook
# ============================================================================

from fastapi import FastAPI

app = FastAPI(
    title="WhatsApp GreenAPI Integration",
    description="Webhook handler for WhatsApp bot",
    version="1.0.0"
)

# Global clients
whatsApp_handler: Optional[WhatsAppHandler] = None

@app.on_event("startup")
async def startup():
    """Initialize clients"""
    global whatsApp_handler
    
    green_config = GreenAPIConfig.from_env()
    green_api = GreenAPIClient(green_config)
    rag_backend = RAGBackendClient(base_url=os.environ.get("RAG_BACKEND_URL", "http://localhost:8000"))
    
    whatsApp_handler = WhatsAppHandler(green_api, rag_backend)
    logger.info("WhatsApp handler initialized!")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "whatsapp-webhook"}


@app.post("/webhook/greenapi")
async def greenapi_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive webhooks from GreenAPI"""
    try:
        payload = await request.json()
        webhook_type = payload.get('typeWebhook', 'unknown')
        sender_data = payload.get('senderData', {})
        chat_id = sender_data.get('chatId', 'unknown')
        
        logger.info(f"=== WEBHOOK RECEIVED ===")
        logger.info(f"Type: {webhook_type}")
        logger.info(f"Chat ID: {chat_id}")
        logger.info(f"Sender: {sender_data.get('senderName', 'unknown')}")
        logger.info(f"========================")
        
        # Process in background
        background_tasks.add_task(whatsApp_handler.handle_incoming_message, payload)
        
        return {"status": "received"}
    
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/webhook/greenapi")
async def greenapi_webhook_verification():
    """For webhook verification (GET request)"""
    return {"status": "webhook active"}


# ============================================================================
# Manual Testing Endpoint
# ============================================================================

@app.post("/send-test-message")
async def send_test_message(chat_id: str, message: str):
    """Send test message (for debugging)"""
    try:
        green_config = GreenAPIConfig.from_env()
        green_api = GreenAPIClient(green_config)
        result = green_api.send_text_message(chat_id, message)
        return {"status": "sent", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
