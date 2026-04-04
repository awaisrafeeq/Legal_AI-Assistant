# RAG Testing Guide — Internal Legal Documents
# Based on: Farrier Evaluation, Pièces Caon, Bail Planète Poutine
# All questions in English | Expected answers in English
# ============================================================

## HOW TO USE THIS FILE
# 1. Send each "QUERY" to your bot exactly as written
# 2. Compare bot's response to "EXPECTED ANSWER"
# 3. Check the PASS/FAIL CRITERIA at the end of each test
# ============================================================


---

## TEST 1 — Basic Fact Retrieval (Property Value)

**QUERY (send to bot):**
> @LegalBot What is the estimated market value of the Rawdon property?

**EXPECTED ANSWER (should contain):**
- The estimated market value is **$827,000** (Eight Hundred Twenty-Seven Thousand Dollars)
- Evaluation date: **March 5, 2014**
- Purpose: **mortgage financing**
- Evaluator: **Sophie Farrier, Certified Appraiser**

**PASS if bot says:** $827,000 or 827 000 $
**FAIL if bot says:** any other number, or "not found"

---

## TEST 2 — Property Description

**QUERY:**
> @LegalBot Describe the land that was evaluated in the Farrier report.

**EXPECTED ANSWER (should contain):**
- Location: **Municipality of Rawdon**, near Route 125
- Cadastral designation: Parts of lots **7C, 8C and lot 9B, Range 7, Canton de Rawdon**
- Total area: **9,192,594 square feet**
- Shape: **irregular shape**, generally flat topography
- Surrounding area: mainly **single-family buildings**
- Zoning: **Residential**

**PASS if bot mentions:** Rawdon, 9,192,594 sq ft, irregular shape
**FAIL if bot:** invents different numbers or location

---

## TEST 3 — Purpose of Evaluation

**QUERY:**
> @LegalBot Why was the Farrier property evaluation conducted? What was its purpose?

**EXPECTED ANSWER (should contain):**
- Purpose: **estimation of probable market value for mortgage financing purposes**
- The evaluation was commissioned by **Jean-François Désormeaux** of **Idéal Finance**
- File reference: **P-14-365**

**PASS if bot mentions:** mortgage financing / hypothécaire financing
**FAIL if bot:** says purpose was sale or litigation

---

## TEST 4 — Evaluator Conditions

**QUERY:**
> @LegalBot What conditions were assumed for the Rawdon land valuation to be valid?

**EXPECTED ANSWER (should contain):**
The market value assumes the land is:
1. Ready to be **developed or exploited in the short term**
2. **Decontaminated** (clean soil, no environmental issues)
3. **Free of any buildings or constructions**
4. **Not subject to significant restrictions** from municipal or government regulations regarding potential residential use

**PASS if bot mentions:** decontaminated, free of buildings, no major restrictions
**FAIL if bot:** omits the conditions or fabricates different ones

---

## TEST 5 — Email Sender Identification (Pièces Caon)

**QUERY:**
> @LegalBot Who sent the "Pièces Caon" email and to whom was it addressed?

**EXPECTED ANSWER (should contain):**
- **Sent by:** Véronique Charton, Administrative Assistant at **Groupe Accretio**
- **Sent to:** Jean-François Desormeaux
- **Date:** December 7, 2021 at 1:52 PM
- **Subject:** Pièces Caon
- **Attachment mentioned:** 2362-2 Pièces demanderesse.pdf

**PASS if bot mentions:** Véronique Charton, Groupe Accretio, Jean-François Desormeaux
**FAIL if bot:** confuses sender/recipient or gives wrong date

---

## TEST 6 — Company Contact Details

**QUERY:**
> @LegalBot What is the address and phone number of Groupe Accretio?

**EXPECTED ANSWER (should contain):**
- Address: **3221 autoroute 440 ouest, local 205, Laval, Québec H7P 5P2**
- Phone: **450-241-1392 extension 101**
- Website: **www.groupeaccretio.com**

**PASS if bot mentions:** 440 ouest, Laval, 450-241-1392
**FAIL if bot:** invents a different address or number

---

## TEST 7 — Lease Contract Email (Planète Poutine)

**QUERY:**
> @LegalBot Who sent the lease contract for Planète Poutine in St-Augustin, and who received it?

**EXPECTED ANSWER (should contain):**
- **Sent by:** Gestion Immobilière Benoit Charron
- **Sent to:** JF Desormeaux AND Isabelle Choulak
- **Date:** December 4, 2015 at 9:49 AM
- **Subject:** Lease contract for **St-Augustin-de-Desmaures**
- **Attachment:** CONTRAT DE BAIL St-Augustin de desmaures docx.docx

**PASS if bot mentions:** Benoit Charron, Desormeaux, Isabelle Choulak, St-Augustin
**FAIL if bot:** gets sender wrong or misses both recipients

---

## TEST 8 — Cross-Document Question (Harder)

**QUERY:**
> @LegalBot What is the connection between Jean-François Desormeaux and these legal documents?

**EXPECTED ANSWER (should contain):**
Jean-François Desormeaux appears in multiple documents:
1. He is the **recipient of the Farrier property evaluation** (commissioned through Idéal Finance) — 2014
2. He **received the Pièces Caon email** from Groupe Accretio — 2021
3. He **received the Planète Poutine lease contract** from Benoit Charron — 2015

He appears to be a central figure, likely a **real estate financier or legal case manager**.

**PASS if bot connects:** Desormeaux across at least 2 documents
**FAIL if bot:** only mentions one document or says "not found"

---

## TEST 9 — "Not in Documents" Test (Hallucination Check)

**QUERY:**
> @LegalBot What is the monthly rent amount in the Planète Poutine lease contract?

**EXPECTED ANSWER:**
The bot should say something like:
> "The specific rent amount is not found in the available documents. The document retrieved is an email sending the lease contract as an attachment, but the actual contract terms (including rent amount) are not in the indexed text."

**PASS if bot:** admits the information is not available and does NOT invent a number
**FAIL if bot:** makes up a rent amount — this is a hallucination!

---

## TEST 10 — Voice Message Simulation

**QUERY (simulate by typing this exactly):**
> @LegalBot eight hundred twenty seven thousand dollars — is that the right value for the Rawdon land?

**EXPECTED ANSWER (should contain):**
- Confirm: Yes, **$827,000** is correct
- Source: Farrier Évaluation Inc., evaluation dated **March 5, 2014**
- This was for mortgage financing purposes

**PASS if bot:** confirms and cites Farrier report
**FAIL if bot:** gives a different value or says not found

---

## SCORING SUMMARY

| Test | Topic | Difficulty | What It Tests |
|------|-------|------------|---------------|
| 1 | Property value | Easy | Basic fact retrieval |
| 2 | Land description | Easy | Multi-detail retrieval |
| 3 | Evaluation purpose | Easy | Context understanding |
| 4 | Valuation conditions | Medium | Detail extraction |
| 5 | Email sender/recipient | Easy | Entity identification |
| 6 | Contact details | Easy | Specific data retrieval |
| 7 | Lease contract email | Easy | Multi-recipient detection |
| 8 | Cross-document link | Hard | Multi-document reasoning |
| 9 | Hallucination check | Critical | Safety/accuracy test |
| 10 | Value confirmation | Easy | Verification query |

**Score:**
- 9-10 correct → RAG is working excellently ✅
- 7-8 correct → Good, minor tuning needed 🟡
- 5-6 correct → Retrieval issues, check chunk size / embeddings ⚠️
- Below 5 → Serious problem, check index and system prompt ❌

---

## WHAT TO WATCH FOR

**Red flags (problems in RAG):**
- Bot answers in French (system prompt language rule not working)
- Bot invents numbers (hallucination — check temperature setting)
- Bot says "not found" for Tests 1-7 (retrieval/indexing problem)
- Bot passes Test 9 with a made-up number (dangerous!)
- Sources shown are wrong documents (embedding quality issue)

**Good signs:**
- Bot quotes French original text then translates to English
- Sources section shows correct file names
- Internal docs marked with 🔴
- Bot admits uncertainty on Test 9
