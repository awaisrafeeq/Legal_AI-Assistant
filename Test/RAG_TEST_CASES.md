# RAG Test Cases for Internal Documents

Complete test suite based on 9 internal documents provided in `e:\Jeff\Test\`

---

## Document 1: Farrier Meadows Evaluation
**File**: `140305-ID-Évaluation FARRIER Meadows.pdf.txt`  
**Type**: Property Evaluation Report

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 1.1 | What is the market value of the Rawdon property in the Farrier report? | **$827,000.00** (HUIT CENT VINGT SEPT MILLE DOLLARS), dated March 5, 2014 | HIGH |
| 1.2 | What is the total land area of the Farrier property? | **9,192,594 square feet** (9 192 594 pieds carrés) | HIGH |
| 1.3 | Who prepared the Farrier evaluation and when? | **Sophie Farrier**, Évaluateur Agréé, March 5, 2014, File # P-14-365 | HIGH |
| 1.4 | Who commissioned the Farrier evaluation and for what purpose? | **Jean-François Désormeaux, IDÉAL FINANCE** for mortgage financing (financement hypothécaire) | MEDIUM |
| 1.5 | What is the location/cadastre of the Farrier property? | Parts of **lots 7C, 8C and lot 9B, Rang 7, Canton de Rawdon**, near Route 125 | HIGH |
| 1.6 | What is the zoning of the Farrier property? | **Residential (Résidentiel)** with potential for ~152 lot subdivision | MEDIUM |
| 1.7 | What was the valuation rate used? | **$0.09 per square foot** | MEDIUM |
| 1.8 | What is the municipal evaluation and matricule? | **$154,500** (2014), Matricule # **8398-41-5065**, taxes: $1,235.07 | LOW |

---

## Document 2: Bélanger à Blache Email
**File**: `141205-ID-D Bélanger à Blache St-Charles 13-15.pdf.txt`  
**Type**: Investment Confirmation

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 2.1 | What was Blache's investment amount in the Joliette project? | **$100,000** invested by company **7 327 587 Canada inc.** in lot **2 901 481** | HIGH |
| 2.2 | Who sent the investment confirmation to Blache and when? | **Denise Bélanger** on **December 5, 2014** at 11:34:58 | HIGH |
| 2.3 | What project was Blache investing in? | **Joliette / Saint-Charles-Borromée** project | MEDIUM |
| 2.4 | What is Denise Bélanger's role and contact? | Directrice, exploitation et administration, Tél: **450 438-6492**, Saint-Jérôme | MEDIUM |
| 2.5 | Who was copied on the Blache email? | **Jean-François Desormeaux** (Desoremaux Jean-Francois) | LOW |

---

## Document 3: Bélanger à Tremblay Email
**File**: `141205-ID-D Bélanger à Tremblay 9119 modif St-Charles 13-15.pdf.txt`  
**Type**: Investment Confirmation

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 3.1 | What was Tremblay's investment amount? | **$250,000** in Joliette project | HIGH |
| 3.2 | Who was the Tremblay email sent to? | **guyt1@videotron.ca** (Monsieur Tremblay) | MEDIUM |
| 3.3 | What was the date and time of Tremblay email? | **December 5, 2014** at 11:34:56 | MEDIUM |
| 3.4 | Did Blache and Tremblay invest in the same lot? | YES - both in lot **2 901 481**, same project Joliette/Saint-Charles-Borromée | MEDIUM |

---

## Document 4: VCharton Trois-Rivières Mandat
**File**: `150217-ID-VCharton Mandat et Courriel Inv Trois-Riv Thibeau Resto.pdf.txt`  
**Type**: Administrative Email

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 4.1 | What documents were sent for Trois-Rivières Thibeau? | **Mandat notaire** and **Courriel investisseur**, File # **DP-0341** | HIGH |
| 4.2 | What is the notary contact for Trois-Rivières? | **Lucie Lafontaine** at **lucie@lamoureuxnotaires.ca** | HIGH |
| 4.3 | When was the Trois-Rivières email sent? | **February 17, 2015** at 15:41:00 | MEDIUM |
| 4.4 | Who sent the Trois-Rivières documents? | **Véronique Charton** from Idéal Finance | MEDIUM |
| 4.5 | Who was copied on the Trois-Rivières email? | Jean-François Desormeaux, dbelanger@idealfinance.ca | LOW |

---

## Document 5: Capital Transit Convention
**File**: `150225-ID-VCharton Convention Capital Transit 3RVS Thibeau Resto.pdf.txt`  
**Type**: Contract Modification

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 5.1 | What changes were made to the Capital Transit convention? | 1) Loss responsibility: **100% Idéal Finance** (changed from 50/50), 2) Capital Transit receives **$100,000** at project end | HIGH |
| 5.2 | When was the Capital Transit convention modified? | **February 25, 2015** at 13:41:00 | HIGH |
| 5.3 | Who are the Capital Transit contacts? | **ppapillon@capitaltransit.ca** and **vtremblay@capitaltransit.ca** | MEDIUM |
| 5.4 | Who sent the Capital Transit modification? | **Véronique Charton** from Idéal Finance | MEDIUM |

---

## Document 6: Brompton Del Negro Loan
**File**: `150715-ID-Courriel partenaire Brompton Del Negro.pdf.txt`  
**Type**: Partner Investment / Loan Terms

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 6.1 | What is the loan number and amount for Brompton? | Loan # **DP-0372**, Amount: **$700,000.00** | HIGH |
| 6.2 | What is the partner split for Brompton loan? | **M Del Negro: $650,000.00** / **Idéal Finance: $50,000.00** | HIGH |
| 6.3 | What is the yield/interest rate for Brompton? | **12%** yield | HIGH |
| 6.4 | What is the loan term for Brompton? | **12 months** | MEDIUM |
| 6.5 | What is the property collateral for Brompton? | **33,975 square feet** at intersection **Laval and de la Croix Nord, Brompton (Sherbrooke)** | HIGH |
| 6.6 | What is the property evaluation for Brompton? | **$3,102,123** | MEDIUM |
| 6.7 | Who is the borrower for Brompton loan? | **9320-0715 Québec inc** | MEDIUM |
| 6.8 | What tenants are at the Brompton property? | **Tim Hortons** ($50,000/year rent) and **Shell/depanneur/poste d'essence** ($150,000/year rent) | MEDIUM |
| 6.9 | What existing mortgage is on Brompton property? | **$1,100,000** second mortgage (2ième hypothèque) currently exists | HIGH |
| 6.10 | What is the mortgage rank for new Brompton loan? | **1st rank** (1er rang) | MEDIUM |
| 6.11 | What is the property type and zoning for Brompton? | **Commercial**, zoning: petroleum, fast food, commercial | LOW |
| 6.12 | What are the interest payment terms for Brompton? | 6 months prepaid, then monthly (or at repayment) | LOW |

---

## Document 7: Lehoux Interest Transfer
**File**: `150817-ID- Lehoux Intérêts prêt transféré Brompton.pdf.txt`  
**Type**: Loan Transfer Clarification

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 7.1 | What was transferred from Rawdon to Brompton? | Interest payments from **rue Queen à Rawdon** loan transferred to **Brompton** project | HIGH |
| 7.2 | When was the Rawdon rue Queen loan notarized? | **November 2014** | HIGH |
| 7.3 | Was the Rawdon loan repaid or transferred? | **Transferred** (not repaid), duration continued on Brompton | HIGH |
| 7.4 | Who was the Lehoux email sent to? | **cjg@live.ca** (M. Lehoux) | MEDIUM |
| 7.5 | What are the loan terms for Rawdon/Brompton? | Interest payable in **12 months** or at repayment | MEDIUM |
| 7.6 | When was the Lehoux email sent? | **August 17, 2015** at 16:26:00 | LOW |
| 7.7 | Who sent the Lehoux clarification? | **Véronique Charton** | LOW |

---

## Document 8: Charron Planète Poutine
**File**: `151204-ID-Charron à l'équipe contrat bail Planète Poutine ST-AUG.pdf.txt`  
**Type**: Lease Contract Email

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 8.1 | What is the subject of Charron's email? | **Lease contract (contrat de bail)** for **Planète Poutine** in **St-Augustin-de-Desmaures** | HIGH |
| 8.2 | Who sent the Planète Poutine email? | **Gestion immobilière Benoit Charron** | MEDIUM |
| 8.3 | Who received the Planète Poutine email? | **JF Desormeaux** and **Isabelle Choulak** | MEDIUM |
| 8.4 | When was the Planète Poutine email sent? | **December 4, 2015** at 09:49:05 | MEDIUM |
| 8.5 | What was attached to the email? | **CONTRAT DE BAIL St-Augustin de desmaures docx.docx** | LOW |

---

## Document 9: Pièces Caon
**File**: `Pièces Caon.pdf.txt`  
**Type**: Administrative Email

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| 9.1 | What was sent in the Pièces Caon email? | Attachment **2362-2 Pièces demanderesse.pdf** | HIGH |
| 9.2 | When was the Pièces Caon email sent? | **December 7, 2021** at 13:52:57 | HIGH |
| 9.3 | What is Véronique Charton's Laval address? | **3221 autoroute 440 ouest, local 205, Laval, Québec H7P 5P2** | HIGH |
| 9.4 | What is Véronique Charton's phone at Laval? | **450-241-1392 poste 101** | MEDIUM |
| 9.5 | Who sent the Pièces Caon email? | **Véronique Charton** from Groupe Accretio | MEDIUM |
| 9.6 | Who received the Pièces Caon email? | **Jean-Francois Desormeaux** | MEDIUM |

---

## Cross-Document Test Cases (Advanced)

| # | Query | Expected Answer | Priority |
|---|-------|-----------------|----------|
| C1 | What investments were made in the Joliette project (lot 2 901 481)? | Blache: **$100,000** + Tremblay: **$250,000** = **$350,000 total** (both Dec 5, 2014) | HIGH |
| C2 | What projects involve Véronique Charton? | Trois-Rivières Thibeau (Feb 2015), Capital Transit convention (Feb 2015), Brompton interest transfer (Aug 2015), Pièces Caon (Dec 2021) | MEDIUM |
| C3 | What loans involve Idéal Finance as a partner investor? | Brompton Del Negro: $50,000 (12% yield), Capital Transit: 100% loss responsibility | MEDIUM |
| C4 | What properties are in the Rawdon area? | Farrier evaluation (9,192,594 sq ft, $827K), rue Queen loan (Nov 2014, transferred to Brompton) | MEDIUM |
| C5 | What are the different Idéal Finance/Groupe Accretio addresses? | Saint-Jérôme: 7 rue John F. Kennedy (2014-2015), Laval: 3221 autoroute 440 (2021+) | MEDIUM |
| C6 | What is the total of all investments in this test set? | Blache $100K + Tremblay $250K + Del Negro $650K + Idéal $50K = **$1,050,000** | LOW |

---

## Test Results Template

Use this format to record your RAG test results:

```
Query: [paste query here]
Expected: [expected answer]
RAG Response: [what RAG actually returned]
Status: [PASS / FAIL / PARTIAL]
Notes: [any issues observed]
```

---

## Critical Success Criteria

**HIGH Priority Tests Must Return:**
- Exact numeric values: $827,000; $700,000; $650,000; $100,000; $250,000; 9,192,594 sq ft; 12%; $3,102,123
- Specific names: Sophie Farrier, Denise Bélanger, Véronique Charton
- Specific dates: March 5, 2014; December 5, 2014; February 25, 2015; August 17, 2015
- File numbers: DP-0372, DP-0341, P-14-365

**Failure Indicators:**
- "I don't have that information"
- Wrong numeric values (hallucinated numbers)
- Mixing up similar projects (Joliette vs Trois-Rivières vs Brompton)
- Missing cross-document connections
