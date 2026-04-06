# Legal AI Assistant - Azure Deployment Guide

## Overview
2 Azure App Service Web Apps with GitHub Actions CI/CD:
- **Backend (Port 8000)**: RAG API - `legal-ai-backend.azurewebsites.net`
- **WhatsApp Bot (Port 8001)**: Webhook handler - `legal-ai-whatsapp.azurewebsites.net`

---

## Prerequisites

1. **Azure Account** with subscription
2. **GitHub Repository** with code pushed

---

## Step 1: Azure Portal - Manual Web App Creation

### 1. Create Resource Group
- Go to [Azure Portal](https://portal.azure.com)
- Search "Resource Groups" → Click "Create"
- **Resource Group Name**: `legal-ai-rg`
- **Region**: `East US` (or your preferred region)
- Click "Review + Create" → "Create"

### 2. Create App Service Plan
- Search "App Service plans" → Click "Create"
- **Resource Group**: `legal-ai-rg`
- **Name**: `legal-ai-plan`
- **Operating System**: Linux
- **Region**: Same as resource group
- **Pricing Plan**: Basic B1 (for testing)
- Click "Review + Create" → "Create"

### 3. Create Backend Web App (Port 8000)
- Search "App Services" → Click "Create" → "Web App"
- **Resource Group**: `legal-ai-rg`
- **Name**: `legal-ai-backend` (must be unique, becomes `legal-ai-backend.azurewebsites.net`)
- **Publish**: Code
- **Runtime stack**: Python 3.11
- **Operating System**: Linux
- **Region**: Same as above
- **Pricing Plan**: `legal-ai-plan` (Basic B1)
- Click "Review + Create" → "Create"

### 4. Create WhatsApp Web App (Port 8001)
- Repeat Step 3
- **Name**: `legal-ai-whatsapp`
- Same settings as backend
- Click "Review + Create" → "Create"

---

## Step 2: Configure Environment Variables

### Backend App (legal-ai-backend)

Go to: Azure Portal → legal-ai-backend → Configuration → Application settings

Add these from your `.env` file:

| Setting Name | Value |
|-------------|-------|
| `SEARCH_ENDPOINT` | Your Azure AI Search endpoint |
| `SEARCH_KEY` | Your search key |
| `SEARCH_INDEX_EXTERNAL` | `legal-docs-external` |
| `SEARCH_INDEX_INTERNAL` | `legal-docs-internal` |
| `OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `OPENAI_KEY` | OpenAI API key |
| `OPENAI_CHAT_DEPLOYMENT` | `chat` |
| `OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-ada-002` |
| `CONTAINER_SAS_URL` | External container SAS URL |
| `INTERNAL_CONTAINER_SAS_URL` | Internal container SAS URL |
| `MIN_SEARCH_SCORE` | `0.15` |
| `CROSS_ENCODER_MIN_SCORE` | `0.5` |
| `PORT` | `8000` |

### WhatsApp App (legal-ai-whatsapp)

Go to: Azure Portal → legal-ai-whatsapp → Configuration → Application settings

Add these from your `.env`:

| Setting Name | Value |
|-------------|-------|
| `RAG_BACKEND_URL` | `https://legal-ai-backend.azurewebsites.net` |
| `GREENAPI_URL` | `https://7103.api.greenapi.com` |
| `GREENAPI_INSTANCE_ID` | Your GreenAPI instance ID (e.g., `7103553770`) |
| `GREENAPI_TOKEN` | Your GreenAPI token |
| `ALLOWED_GROUP_ID` | WhatsApp group ID (e.g., `120363405725998465@g.us`) |
| `BOT_NAME` | `Legal-Assistant` |
| `BOT_PHONE` | Bot phone number (e.g., `19542032639`) |
| `ALLOWED_DM_PHONES` | Comma-separated phone numbers for DMs |
| `WHISPER_ENDPOINT` | Azure Whisper endpoint |
| `WHISPER_KEY` | Whisper API key |
| `OPENAI_WHISPER_DEPLOYMENT` | `whisper` |
| `OPENAI_ENDPOINT` | Azure OpenAI endpoint (fallback for Whisper) |
| `OPENAI_KEY` | Azure OpenAI key (fallback for Whisper) |
| `PORT` | `8001` |

**IMPORTANT**: Click "Save" after adding all variables!

---

## Step 3: Get Publish Profiles

### For Backend:
1. Azure Portal → legal-ai-backend → Get publish profile
2. Download `legal-ai-backend.PublishSettings`
3. Open file, copy entire XML content

### For WhatsApp:
1. Azure Portal → legal-ai-whatsapp → Get publish profile
2. Download `legal-ai-whatsapp.PublishSettings`
3. Copy XML content

---

## Step 4: Add GitHub Secrets

Go to: GitHub Repo → Settings → Secrets and variables → Actions

Add **Repository secrets**:

| Secret Name | Value |
|------------|-------|
| `AZURE_BACKEND_PUBLISH_PROFILE` | Backend publish profile XML |
| `AZURE_WHATSAPP_PUBLISH_PROFILE` | WhatsApp publish profile XML |

---

## Step 5: Deploy

Push code to main branch:
```bash
git add .
git commit -m "Add Azure deployment workflows"
git push origin main
```

GitHub Actions automatically deploy both apps!

---

## Step 6: Verify Deployment

### Check Backend:
```bash
curl https://legal-ai-backend.azurewebsites.net/health
```

### Check WhatsApp:
```bash
curl https://legal-ai-whatsapp.azurewebsites.net/webhook
```

### View Logs:
Azure Portal → Web App → Log stream

---

## Step 7: Update GreenAPI Webhook

1. Go to GreenAPI settings
2. Update webhook URL to:
   ```
   https://legal-ai-whatsapp.azurewebsites.net/webhook
   ```

---

## Troubleshooting

### App won't start:
- Check Application Logs in Azure Portal
- Verify all environment variables are set
- Check startup.txt file exists in deployment

### 500 errors:
- Check Log stream for Python tracebacks
- Verify `requirements.txt` has all dependencies

### WhatsApp not responding:
- Verify `RAG_BACKEND_URL` points to correct Azure URL
- Check GreenAPI webhook is updated
- Test backend is reachable from WhatsApp app

---

## Scaling Up

When ready for production:
1. Upgrade App Service Plan to S1 or higher
2. Enable Application Insights for monitoring
3. Configure Azure CDN for faster responses
4. Enable auto-scaling based on CPU/memory

---

## Cost Estimation (Monthly)

- **Basic B1 Plan**: ~$13/month per app ($26 total)
- **Azure AI Search**: ~$25/month (Basic tier)
- **Azure OpenAI**: Pay per use (~$5-20/month for typical usage)
- **Storage**: ~$1-5/month
- **Total**: ~$60-80/month for both apps running
