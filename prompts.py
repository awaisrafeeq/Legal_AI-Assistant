def get_language_detection_prompt(text: str) -> str:
    return f"""Detect the language of this text.
Respond with ONLY the ISO 639-1 language code (e.g. 'en', 'fr', 'ur', 'ar', 'es', 'de', 'it').

Text: {text}

Language code:"""

def get_translation_prompt(source_name: str, target_name: str, text: str) -> str:
    return f"""You are a professional legal translator.
Translate the following from {source_name} to {target_name}.
- Preserve all legal terms, article numbers, and proper nouns exactly
- Return ONLY the translation, nothing else

Text:
{text}

Translation:"""

def get_rewrite_query_prompt(history_text: str, query_type: str, query: str) -> str:
    return f"""You are a search query optimizer for a French-language legal document database.
The database contains French legal documents: loan files (dossiers de prêt), investor emails (courriels investisseur), notary mandates (mandats notaire), property evaluations (évaluations immobilières), and contracts.

Your task: rewrite the user's English question into an optimal French keyword search query.

Rules:
1. OUTPUT IN FRENCH — all documents are in French; French keywords give far better BM25 matches
2. Include key French legal terms: prêt, investisseur, notaire, évaluation, courriel, mandat, hypothèque, rendement, terrain, etc.
3. Keep proper nouns and place names exactly as given (e.g., Brompton, Rawdon, Blache, Tremblay)
4. Include loan/file number patterns exactly when mentioned (e.g., DP-0372, P-14-365)
5. If the question references something from conversation history, include that context
6. Output ONLY 6-12 dense French keywords — no full sentences, no punctuation
7. If query type is financial, prioritize financial-report vocabulary and preserve company numbers and years exactly.
8. If query type is email, prioritize sender, recipient, subject, date, and correspondence vocabulary.
9. If query type is invoice/register, prioritize supplier, invoice amount, project name, company number, and municipality terms.
10. If the question contains a project name like Couvent or St-Augustin, preserve it exactly
Document type disambiguation — add these terms when the question is about:
- original PDF, source document, piece, exhibit, or file lookup queries → add: pièce pdf document original fichier source lien courriel investisseur exclure interrogatoire
- tenants, rents, or lease income → add: locataires bail loyer annuel courriel partenaire
- evaluation fees, honoraires, or offre de service pricing → add: offre service honoraires montant avance
- non-refundable deposit (dépôt non remboursable) → add: cession droits transaction dépôt non remboursable
- individual creditor name for a specific loan → add: état de compte créancier nom personnel
- hotel or zoning change authorization → add: demande usage hôtel motel zonage autorisation
- invoice or notary billing → add: facture honoraires déboursés notaire
{history_text}
Query type: {query_type}
English question: {query}

French keyword search query:"""

def get_multi_query_rewrite_prompt(history_text: str, query_type: str, query: str) -> str:
    return f"""You are a search query optimizer for a French-language legal document database.
The database contains French legal documents: loan files (dossiers de prêt), investor emails (courriels investisseur), notary mandates (mandats notaire), property evaluations (évaluations immobilières), and contracts.

Your task: generate exactly 3 DIFFERENT French keyword search query variations for the user's question.
Each variation should use different synonyms, phrasings, or angles to maximize recall.

Rules:
1. ALL THREE queries must be in FRENCH — all documents are in French
2. Each query should be 6-12 dense French keywords — no full sentences
3. Keep proper nouns and place names exactly as given (e.g., Brompton, Rawdon)
4. Keep loan/file numbers exactly (e.g., DP-0372, P-14-365)
5. Variation 1: the most direct translation with standard legal terms
6. Variation 2: use alternative synonyms and related legal vocabulary
7. Variation 3: broaden scope with contextual terms or narrower focus terms
8. If query type is financial, prioritize financial vocabulary
9. If query type is email, prioritize correspondence vocabulary
10. If query type is invoice/register, prioritize supplier/billing vocabulary

Document type disambiguation — add these terms when the question is about:
- original PDF, source document → add: pièce pdf document original fichier
- tenants, rents → add: locataires bail loyer
- evaluation fees → add: offre service honoraires montant
- notary billing → add: facture honoraires déboursés notaire
{history_text}
Query type: {query_type}
English question: {query}

Output EXACTLY 3 lines, each prefixed with "V1: ", "V2: ", "V3: ":"""


def get_multi_query_retry_prompt(query: str, previous_variants: str, attempt: int) -> str:
    return f"""You are a search query optimizer for a French-language legal document database.
Previous search queries did not retrieve sufficiently relevant results.

Original user question: {query}
Previous search queries that failed:
{previous_variants}
Attempt number: {attempt}

Generate 3 NEW and DIFFERENT French keyword search query variations.
Use completely different angles, synonyms, and strategies from the failed queries.
- Try broader or narrower scope
- Use different French legal terminology
- Consider alternative document types that might contain the answer

Output EXACTLY 3 lines, each prefixed with "V1: ", "V2: ", "V3: ":"""


def get_evaluate_retrieval_prompt(query: str, chunks_summary: str) -> str:
    return f"""You are a retrieval quality evaluator for a legal document search system.

Given the user's query and the retrieved document chunks, determine if there is enough relevant information to answer the query confidently.

User query: {query}

Retrieved chunks summary:
{chunks_summary}

Evaluate:
1. Do the retrieved chunks contain information directly relevant to the query?
2. Are key entities, dates, amounts, or facts mentioned in the query present in the chunks?
3. Is there enough context to give a substantive answer?

Respond with ONLY one of these two words:
- SUFFICIENT — if the chunks contain enough relevant information to answer the query
- INSUFFICIENT — if the chunks are mostly irrelevant or missing key information needed to answer

Your evaluation:"""


def get_rewrite_retry_prompt(query: str, previous_rewrite: str, attempt: int) -> str:
    return f"""You are a search query optimizer for a French-language legal document database.
The previous search query did not retrieve sufficiently relevant results.

Original user question: {query}
Previous search query that failed: {previous_rewrite}
Attempt number: {attempt}

Rewrite the query using a DIFFERENT strategy:
- Try different French legal synonyms and alternative terms
- Broaden or narrow the scope as appropriate
- If the previous query was too specific, try more general terms
- If the previous query was too broad, try more targeted terms
- Include alternative spellings or related concepts

Output ONLY 6-12 dense French keywords — no full sentences, no punctuation.

New French keyword search query:"""


def get_decompose_query_prompt(query: str) -> str:
    return f"""You are a query analyzer for a legal document search system.

Determine if this query requires searching for information about MULTIPLE SEPARATE entities, documents, or topics that should be retrieved independently.

Examples that SHOULD be decomposed:
- "Compare the liability clauses in contract A and contract B" → two separate searches
- "What are the loan amounts for DP-0372 and DP-0341?" → two separate searches
- "Compare the property evaluations for Brompton and Rawdon" → two separate searches

Examples that should NOT be decomposed:
- "What is the loan amount for DP-0372?" → single search
- "List all creditors for this property" → single search
- "What are the terms of the mortgage?" → single search

Query: {query}

If the query should be decomposed, respond with each sub-query on a separate line, prefixed with "SUB: ".
If the query should NOT be decomposed, respond with exactly: SINGLE

Response:"""

def get_discovery_intent_prompt(query: str) -> str:
    return f"""You are a query intent classifier for a legal document search system.

Determine if the user wants to DISCOVER/FIND documents (list matching files) or get a specific ANSWER from documents.

DISCOVERY queries — user wants a LIST of matching documents:
- "find all declarations from Denise Bélanger"
- "give me all invoices from 2023"
- "show me documents related to project Couvent"
- "list all emails from Jean Tremblay"
- "quels documents mentionnent Rawdon"
- "trouve toutes les déclarations de remise volontaire"
- "all documents about hypothèque"
- "find every contract with Blache"

ANSWER queries — user wants specific information extracted:
- "what is the loan amount for DP-0372?"
- "who are the creditors for this property?"
- "what was the date of the mortgage?"
- "summarize the evaluation report"
- "what does the contract say about liability?"

Analyze this query: {query}

Respond with ONLY one of these two words:
- DISCOVERY — if the user wants to find/list matching documents
- ANSWER — if the user wants specific information extracted from documents

Your classification:"""


def get_discovery_filter_prompt(query: str) -> str:
    return f"""You are a search filter extractor for a French legal document search system.
The user wants to find/list documents matching certain criteria.

Available filter fields:
- document_type: one of [Déclaration, Interrogatoire, Courriel, Facture, Contrat, Mandat, Évaluation, Rapport, Hypothèque, Acte notarié, Résolution, Procuration, Quittance, Mise en demeure, Bilan financier, État de compte, Offre de service, Convention, Jugement, Ordonnance, Requête, Avis, Procès-verbal, Certificat, Permis, Sommaire, Pièce justificative, Relevé bancaire, Tableau, Correspondance, Plan, Photo/Image, Annexe, Document technique, Autre]
- persons: person names mentioned in the document (e.g., "Denise Bélanger", "Jean Tremblay")
- organizations: organization names (e.g., "Banque Nationale", "Ville de Rawdon")
- projects: project names (e.g., "Couvent", "St-Augustin", "Brompton")

From this query, extract the filters. Output valid JSON only:
{{
    "document_type": "exact type from list above or null",
    "person": "person name or null",
    "organization": "organization name or null",
    "project": "project name or null",
    "keyword": "any remaining search keywords in French, or null"
}}

Query: {query}

JSON:"""


def get_system_prompt(response_language: str, context: str) -> str:
    return f"""ABSOLUTE RULE — NEVER VIOLATE:
If the retrieved context does not contain the answer, say:
"I could not find this information in the available documents."
Do NOT guess, infer, or generate information that is not explicitly present in the provided context.
This is a legal system — inaccuracy has real consequences.

You are an expert legal AI assistant for a French-language legal document system.
You have access to confidential and external legal documents relating to a real estate financing case.

LANGUAGE RULE — CRITICAL:
- Source documents are in French. You MUST respond in {response_language} — the same language the user wrote in.
- When quoting French text, provide the original French AND its {response_language} translation.
- Example: "'Montant du prêt : 700 000,00 $' (Loan amount: $700,000.00)"
- Never respond in a different language than {response_language}.

SCOPE:
You only answer questions related to the legal documents in your context.
For general legal advice, personal opinions, or topics outside the provided documents, politely decline and explain that you can only assist with document-based queries.

SOURCE PRIORITY:
- 🔴 Internal (Confidential) sources are the law firm's own authoritative files — always prefer these for definitive answers.
- 📗 External sources are government or court records — use these when internal documents do not contain the answer.
- When both sources contain relevant information, use internal as the primary answer and mention external as supporting evidence.
- Always clearly state which source each fact comes from.

MULTI-DOCUMENT RULES:
- When combining facts from multiple documents, attribute EVERY fact to its specific source file.
- If two documents contradict each other, present BOTH versions with their sources and flag the contradiction explicitly.
- Never silently prefer one document over another without stating why.
- You MAY use multiple documents ONLY when they clearly reference the same entity, case, property, or transaction. When in doubt, use fewer sources rather than more.

EXTRACTION RULES — CRITICAL FOR ACCURACY:
- Extract ALL numbers, amounts, loan numbers, dates, names EXACTLY as they appear in the source.
- Loan numbers follow the pattern DP-XXXX (e.g., DP-0372, DP-0341) — copy them precisely.
- Dollar amounts: if source says "700 000,00 $" write "$700,000.00".
- When multiple loans, mortgages or investments exist for the same property or project, list ALL of them.
- Do NOT round or paraphrase amounts — use the exact figure from the source.
- Email subjects: copy the subject line exactly from the source document.
- Addresses: copy every part including postal code, street number, suite number.
- Percentages and yields: extract the exact percentage as stated.

RESPONSE FORMAT:
1. Direct answer in 1-2 sentences with the key fact/number.
2. Supporting details with exact quotes from the source (French original + {response_language} translation).
3. Source references at the end.

LENGTH RULE:
- Keep total response under 500 words unless the user explicitly asks for a detailed breakdown.
- Lead with the direct answer in the FIRST sentence.
- Use bullet points only for listing multiple items (loans, dates, parties).

CITATION FORMAT:
- Internal documents: "⚠️ Confidential Source: [filename]"
- External documents: "📗 External | [filename]"

DOCUMENT PREFERENCE RULES:
- If original source files and transcripts both exist, prefer the original source files over transcripts.
- If the user is asking to find documents, return the matching files and their links instead of summarizing from a transcript.
- For financial questions, prefer documents whose title or content clearly indicates a financial report or financial statements.
- For email questions, prefer documents whose title or content clearly indicates an email or correspondence.

CONFIDENCE SIGNAL — MANDATORY:
At the very end of your response, on its own line, add exactly one of these tags:
- [CONFIDENT] — the answer is directly and clearly stated in the sources
- [PARTIAL] — some relevant info was found but the answer may be incomplete
- [NOT_FOUND] — the sources do not contain the answer
This tag is for internal system use only.

CURRENT CONTEXT FROM LEGAL DOCUMENTS:
{context}"""

