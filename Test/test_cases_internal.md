# Internal Documents Test Cases

Based on: `140305-ID-Évaluation FARRIER Meadows.pdf.txt` (Farrier Evaluation Report)

---

## Document Summary

**Property Evaluation Report by Sophie Farrier (Farrier Évaluation Inc.)**
- **Date**: March 5, 2014 (5 mars 2014)
- **Client**: Jean-François Désormeaux, IDÉAL FINANCE
- **Property**: Parts of lots 7C, 8C and lot 9B, Rang 7, Canton de Rawdon
- **Total Area**: 9,192,594 square feet (9 192 594 pieds carrés)
- **Market Value**: **$827,000.00** (HUIT CENT VINGT SEPT MILLE DOLLARS)
- **Rate**: $0.09 per square foot
- **Purpose**: Mortgage financing (pour des fins de financement hypothécaire)
- **Evaluator**: Sophie Farrier, Évaluateur Agréé
- **Zoning**: Residential (Résidentiel)
- **Potential**: Subdivision into ~152 lots

---

## Test Cases (Expected Answers)

### 1. Market Value Test (Exact Figure)
**Query**: What is the market value of the Rawdon property in the Farrier report?

**Expected Answer**:
- The market value is **$827,000.00** (eight hundred twenty-seven thousand dollars)
- Date of evaluation: March 5, 2014
- Rate: $0.09 per square foot for 9,192,594 sq ft

---

### 2. Property Details Test
**Query**: What is the total land area and location of the property evaluated by Farrier?

**Expected Answer**:
- **Total area**: 9,192,594 square feet (9 192 594 pieds carrés)
- **Location**: Canton de Rawdon, near Route 125
- **Cadastre**: Parts of lots 7C, 8C and lot 9B, Rang 7
- **Municipality**: Rawdon (Québec)

---

### 3. Evaluator Information Test
**Query**: Who prepared the Farrier evaluation report and when?

**Expected Answer**:
- **Evaluator**: Sophie Farrier, Évaluateur Agréé (Certified Evaluator)
- **Company**: Farrier Évaluation Inc.
- **Date**: March 5, 2014 (5 mars 2014)
- **Our file number**: P-14-365

---

### 4. Client/Purpose Test
**Query**: Who commissioned the Farrier evaluation and for what purpose?

**Expected Answer**:
- **Client**: Jean-François Désormeaux, IDÉAL FINANCE
- **Purpose**: Mortgage financing (pour des fins de financement hypothécaire)
- **Property**: Land in Rawdon for development

---

### 5. Zoning and Usage Test
**Query**: What is the zoning and optimal use of the Farrier property?

**Expected Answer**:
- **Zoning**: Residential (Résidentiel)
- **Optimal use**: Residential development
- **Potential**: Subdivision into approximately 152 lots (based on plan by Pierre Robitaille, surveyor)
- **Current use**: Vacant land (terrain vacant), not currently exploited

---

### 6. Calculation Method Test
**Query**: How was the market value calculated in the Farrier report?

**Expected Answer**:
- **Method**: Comparison method (méthode de comparaison)
- **Calculation**: 9,192,594 sq ft × $0.09/sq ft = $827,333 → rounded to **$827,000**
- **Rate range**: Comparable sales ranged from $0.02 to $0.23 per sq ft

---

### 7. Municipal Assessment Test
**Query**: What are the municipal evaluation details for the Farrier property?

**Expected Answer**:
- **Matricule # 8398-41-5065** (largest portion)
- **2014 Municipal evaluation**: $154,500 (terrain only)
- **Taxes 2014**: $1,235.07
- **Role**: Triennial 2012-2013-2014
- **Note**: Multiple matricules (8297-98-5278, 8397-35-6049, etc.)

---

### 8. Environmental Assessment Test
**Query**: Were there any environmental concerns noted in the Farrier evaluation?

**Expected Answer**:
- **Soil condition**: No known environmental analysis (Aucune analyse environnementale du sol consultée)
- **Risk assessment**: Believed to have no contamination risk based on past site use
- **Assumption**: Site is considered decontaminated (site décontaminé) for valuation purposes

---

### 9. Limiting Clauses Test
**Query**: What are the limitations mentioned in the Farrier evaluation report?

**Expected Answer**:
- Assumes property is free of legal issues (titles valid, free of charges)
- No responsibility for hidden construction defects, mold, contaminated soil
- Based on visual inspection only (inspection sommaire)
- Cannot be used for purposes other than mortgage financing without written consent
- Report cannot be published without client consent

---

### 10. Comparable Sales Test
**Query**: What comparable sales were used in the Farrier evaluation?

**Expected Answer**:
- **Time period**: 2003-2013 sales analyzed
- **Sample comparables**: Various Rawdon residential land sales
- **Rate range**: $0.02 to $0.23 per square foot
- **Total comparables**: 300+ transactions in table (annexed)

---

## Success Criteria

| Test | Priority | Expected Retrieval |
|------|----------|-------------------|
| 1. Market value ($827,000) | **HIGH** | Exact numeric value |
| 2. Property area (9,192,594 sq ft) | **HIGH** | Exact figure with location |
| 3. Evaluator (Sophie Farrier) | HIGH | Name and credentials |
| 4. Client/Purpose | MEDIUM | IDÉAL FINANCE + mortgage |
| 5. Zoning/Usage | MEDIUM | Residential + 152 lots |
| 6. Calculation method | MEDIUM | Comparison + $0.09/sq ft |
| 7. Municipal data | LOW | Matricule numbers |
| 8. Environmental | LOW | No contamination risk |
| 9. Limitations | LOW | Clauses listed |
| 10. Comparables | LOW | 2003-2013 sales |

---

## Notes

- Document contains EXACT numeric values that RAG should retrieve
- Critical test: $827,000 market value (not "not found")
- Property area is specific: 9,192,594 square feet
- Date is specific: March 5, 2014
- Evaluator name is specific: Sophie Farrier

---

## Document 2: Bélanger à Blache (141205-ID-D Bélanger à Blache St-Charles)
**Type**: Investment Confirmation Email | **From**: Denise Bélanger

### Key Facts:
- **Date**: December 5, 2014
- **To**: yvesfblache@hotmail.com (Monsieur Blache)
- **CC**: Jean-François Desormeaux
- **Project**: Joliette / Saint-Charles-Borromée
- **Lot**: 2 901 481
- **Investment Amount**: **$100,000**
- **Investor Company**: 7 327 587 Canada inc.

### Test Cases:

**Query 2.1**: What was Blache's investment amount in the Joliette project?
- **Expected**: $100,000 invested by 7 327 587 Canada inc., lot 2 901 481

**Query 2.2**: Who sent the investment confirmation to Blache and when?
- **Expected**: Denise Bélanger, December 5, 2014, for Joliette/Saint-Charles-Borromée project

**Query 2.3**: What is Denise Bélanger's role?
- **Expected**: Directrice, exploitation et administration, Tél: 450 438-6492

---

## Document 3: Bélanger à Tremblay (141205-ID-D Bélanger à Tremblay)
**Type**: Investment Confirmation Email | **From**: Denise Bélanger

### Key Facts:
- **Date**: December 5, 2014
- **To**: guyt1@videotron.ca (Monsieur Tremblay)
- **Investment Amount**: **$250,000**
- **Project**: Joliette, lot 2 901 481

### Test Cases:

**Query 3.1**: What was Tremblay's investment amount?
- **Expected**: $250,000 in Joliette project (lot 2 901 481)

**Query 3.2**: How much did Blache and Tremblay invest together in Joliette?
- **Expected**: Combined $350,000 ($100,000 + $250,000)

---

## Document 4: VCharton Mandat Trois-Riv (150217-ID-VCharton Mandat et Courriel)
**Type**: Administrative Email | **From**: Véronique Charton (Idéal Finance)

### Key Facts:
- **Date**: February 17, 2015
- **To**: Lucie Lafontaine (lucie@lamoureuxnotaires.ca)
- **Project**: Trois Rivières-Thibeau (3ème dossier)
- **File**: DP-0341

### Test Cases:

**Query 4.1**: What documents were sent for the Trois-Rivières Thibeau project?
- **Expected**: Mandat notaire and Courriel investisseur, file DP-0341, Feb 17, 2015

**Query 4.2**: Who is the notary contact for Trois-Rivières project?
- **Expected**: Lucie Lafontaine at lamoureuxnotaires.ca

---

## Document 5: Convention Capital Transit (150225-ID-VCharton Convention)
**Type**: Contract Modification Email | **From**: Véronique Charton

### Key Facts:
- **Date**: February 25, 2015
- **Modification 1**: Loss responsibility changed to **100% Idéal Finance** (was 50/50)
- **Modification 2**: Capital Transit will receive **$100,000** at project end

### Test Cases:

**Query 5.1**: What changes were made to the Capital Transit convention?
- **Expected**: 100% Idéal Finance loss responsibility, $100,000 payment to Capital Transit

**Query 5.2**: Who are the Capital Transit contacts?
- **Expected**: ppapillon@capitaltransit.ca, vtremblay@capitaltransit.ca

---

## Document 6: Brompton Del Negro Loan (150715-ID-Courriel partenaire Brompton)
**Type**: Partner Investment Email | **From**: Idéal Finance

### Key Facts:
- **Date**: July 15, 2015
- **Loan #**: DP-0372
- **Loan Amount**: **$700,000.00**
- **Term**: 12 months
- **Interest**: 12% yield
- **Partner Split**: Del Negro: $650,000 / Idéal Finance: $50,000
- **Property**: 33,975 sq ft at Laval/de la Croix Nord, Brompton
- **Evaluation**: $3,102,123
- **Tenants**: Tim Hortons ($50K/year), Shell ($150K/year)
- **Existing Debt**: $1,100,000 2nd mortgage

### Test Cases:

**Query 6.1**: What are the details of loan DP-0372?
- **Expected**: $700,000, 12 months, Del Negro: $650K, Idéal: $50K, 12% yield

**Query 6.2**: What is the property collateral for the Brompton loan?
- **Expected**: 33,975 sq ft, Brompton, eval: $3,102,123, 1st rank mortgage

**Query 6.3**: What tenants are at the Brompton property?
- **Expected**: Tim Hortons ($50K/year), Shell/depanneur/gas station ($150K/year)

**Query 6.4**: What existing mortgage is on the Brompton property?
- **Expected**: $1,100,000 2nd mortgage exists, new mortgage is 1st rank

---

## Document 7: Lehoux Interest Transfer (150817-ID-Lehoux Intérêts prêt transféré)
**Type**: Interest Payment Clarification | **From**: Véronique Charton

### Key Facts:
- **Date**: August 17, 2015
- **To**: cjg@live.ca (M. Lehoux)
- **Subject**: Rawdon rue Queen → Brompton transfer
- **Loan Date**: November 2014
- **Terms**: Interest payable in 12 months or at repayment

### Test Cases:

**Query 7.1**: What was transferred from Rawdon to Brompton?
- **Expected**: Interest payments from rue Queen Rawdon loan transferred to Brompton project

**Query 7.2**: When was the Rawdon rue Queen loan notarized?
- **Expected**: November 2014

**Query 7.3**: Was the Rawdon loan repaid or transferred?
- **Expected**: Transferred (not repaid), duration continued on Brompton

---

## Document 8: Charron Planète Poutine (151204-ID-Charron à l'équipe)
**Type**: Team Email | **From**: Benoit Charron

### Key Facts:
- **From**: Gestion immobilière Benoit Charron
- **To**: JF Desormeaux, Isabelle Choulak
- **Subject**: Lease contract for Planète Poutine ST-Augustin-de-Desmaures

### Test Cases:

**Query 8.1**: What is the subject of Charron's email?
- **Expected**: Lease contract (contrat bail) for Planète Poutine in ST-Augustin-de-Desmaures

---

## Document 9: Pièces Caon (Pièces Caon.pdf)
**Type**: Administrative Email | **From**: Groupe Accretio

### Key Facts:
- **Date**: December 7, 2021
- **To**: Jean-Francois Desormeaux
- **From**: Véronique Charton (new address)
- **New Address**: 3221 autoroute 440 ouest, Laval H7P 5P2
- **Attachment**: 2362-2 Pièces demanderesse.pdf

### Test Cases:

**Query 9.1**: What was sent in the Pièces Caon email?
- **Expected**: Attachment 2362-2 Pièces demanderesse.pdf, Dec 7, 2021

**Query 9.2**: What is Véronique Charton's Laval address?
- **Expected**: 3221 autoroute 440 ouest, local 205, Laval H7P 5P2

---

## Cross-Document Test Cases

**Cross 1**: What investments were made in the Joliette project?
- **Expected**: Blache: $100,000 + Tremblay: $250,000 = $350,000 total

**Cross 2**: What projects involve Véronique Charton?
- **Expected**: Trois-Rivières (Feb 2015), Capital Transit (Feb 2015), Brompton interest (Aug 2015), Pièces Caon (Dec 2021)

**Cross 3**: What properties are in the Rawdon area?
- **Expected**: Farrier evaluation (9M+ sq ft), rue Queen loan (Nov 2014)

**Cross 4**: What are the different Idéal Finance addresses?
- **Expected**: Saint-Jérôme (7 rue John F. Kennedy, 2014-2015), Laval (3221 autoroute 440, 2021+)

---

## Updated Success Criteria

| Priority | Test Type | Examples |
|----------|-----------|----------|
| **HIGH** | Exact numeric values | $827,000; $700,000; $100,000; $250,000; $650,000 |
| **HIGH** | Specific people | Sophie Farrier, Denise Bélanger, Véronique Charton |
| **HIGH** | Specific dates | March 5, 2014; December 5, 2014; February 25, 2015 |
| **MEDIUM** | Project names | Joliette, Trois-Rivières Thibeau, Brompton Del Negro |
| **MEDIUM** | Loan terms | DP-0372, 12% yield, 12 months, 1st rank |
| **LOW** | Contact details | Phone numbers, addresses, emails |
| **LOW** | Cross-document facts | Combined investments, transfer relationships |
