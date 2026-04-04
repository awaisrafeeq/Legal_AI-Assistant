# WhatsApp Bot to RAG Backend - Complete Flow Chart

## Overview
End-to-end flow from user sending a WhatsApp message to receiving an AI-powered response from the RAG system.

---

## 1. User Query Flow (End-to-End)

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. USER    │────▶│ 2. WHATSAPP     │────▶│ 3. GREENAPI     │────▶│ 4. WEBHOOK      │
│   SENDS      │     │    MESSAGE      │     │    FORWARD      │     │    RECEIVED     │
│   MESSAGE    │     │                 │     │                 │     │                 │
└──────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                           │
                                                                           ▼
```

---

## 2. WhatsApp Bot (Port 8001)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Parse       │───▶│ Check       │───▶│ Check Bot   │───▶│ Clean Query │
│ Webhook     │    │ Allowed     │    │ Mentioned   │    │ (Remove @)  │
│ JSON        │    │ Group Only  │    │ in Group    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Processing Steps:

| Step | Action | Result |
|------|--------|--------|
| Parse | Extract chatId, senderName, text | Raw message data |
| Check Allowed Group | Verify `120363405725998465@g.us` | ❌ Ignore others |
| Check Mention | Look for @Legal-Assistant | ❌ Ignore if not mentioned |
| Clean Query | Remove bot tags | Clean text for RAG |

---

## 3. RAG Backend (Port 8000)

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 5. RECEIVE      │   │ 6. LANGGRAPH    │   │ 7. AZURE AI     │   │ 8. AZURE        │
│    QUERY        │──▶│    WORKFLOW     │──▶│    SEARCH       │──▶│    OPENAI       │
│                 │   │                 │   │                 │   │                 │
│ POST /chat      │   │ • Translate     │   │ • Hybrid        │   │ • Generate      │
│ {query: "..."}  │   │   (EN/FR)       │   │ • BM25 +        │   │ • Format        │
│                 │   │ • Detect Lang   │   │   Vectors       │   │   Answer        │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Return Format:
```json
{
  "answer": "The bankruptcy file number is...",
  "sources": [...],
  "source_container": "legal-documents"
}
```

---

## 4. Response Back to User

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 9. FORMAT       │   │ 10. ADD SOURCE  │   │ 11. SEND        │
│    ANSWER       │──▶│    INDICATORS   │──▶│    RESPONSE     │──▶┌─────────────────┐
│                 │   │                 │   │                 │   │ 12. USER        │
│ • Answer text   │   │ • 📄 External   │   │ GreenAPI        │   │    RECEIVES     │
│ • Source docs   │   │ • 🏛️ Internal   │   │ send_text_msg   │   │    RESPONSE     │
│ • Citations     │   │ • 🔗 Links      │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## Key Components

| Component | Technology | Port | Purpose |
|-----------|------------|------|---------|
| 🟢 GREENAPI | WhatsApp API | - | Webhook Gateway |
| 🔵 WhatsApp Bot | FastAPI | 8001 | Filters, Cleans, Routes |
| 🟠 RAG Backend | FastAPI + LangGraph | 8000 | AI Processing |
| 🔴 Azure Search | AI Search | - | BM25 + Vector Search |
| 🟣 Azure OpenAI | GPT-4 + Embeddings | - | LLM + Embeddings |

