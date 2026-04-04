# Reminders

## MVP Scope (Phase 1)

- WhatsApp-based AI legal assistant MVP will initially support **PDF files only**.
- PDFs include:
  - text-based PDFs
  - scanned/image PDFs (require OCR)
- Primary workflow:
  - ingest PDFs from Azure Blob Storage
  - extract text (OCR where needed)
  - index into Azure AI Search
  - answer queries via RAG (retrieval + summarized response)
  - return secure download link (SAS)

## Phase 2 Backlog (Add Later)

- Add support for additional file types stored in Blob:
  - DOCX
  - XLSX/Excel
  - standalone images (JPG/PNG/HEIC/etc.)
  - videos
- For standalone images and PDF pages that are mostly images/photographs:
  - add captioning/tagging pipeline to generate searchable descriptions

## Search Quality Notes

- If filenames/paths do not match the user query, retrieval must rely on:
  - extracted document text (OCR/text extraction)
  - generated metadata (captions/tags) for images
  - vector embeddings for semantic similarity search

## Localization

- Project is for a Canadian client.
- Keep responses, templates, and terminology aligned to the client’s jurisdiction and language needs.
