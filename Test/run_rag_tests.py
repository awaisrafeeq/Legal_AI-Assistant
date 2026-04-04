"""
RAG Comprehensive Test Suite
Covers ALL internal documents in Test\ folder.
Run with: python run_rag_tests.py
"""
import requests
import json
from datetime import datetime

RAG_URL = "http://localhost:8000/chat"

TEST_CASES = [

    # =========================================================
    # DOC 1: Farrier Meadows Evaluation (140305)
    # =========================================================
    {
        "query": "What is the market value of the Rawdon property in the Farrier report?",
        "expected": "$827,000.00 (HUIT CENT VINGT SEPT MILLE DOLLARS), dated March 5, 2014",
        "priority": "HIGH", "source": "Farrier Meadows"
    },
    {
        "query": "What is the total land area of the Farrier property?",
        "expected": "9,192,594 square feet",
        "priority": "HIGH", "source": "Farrier Meadows"
    },
    {
        "query": "Who prepared the Farrier evaluation and what is the file number?",
        "expected": "Sophie Farrier, Évaluateur Agréé, file P-14-365, dated March 5, 2014",
        "priority": "HIGH", "source": "Farrier Meadows"
    },
    {
        "query": "Who commissioned the Farrier evaluation and for what purpose?",
        "expected": "Jean-François Désormeaux, Idéal Finance, for mortgage financing (financement hypothécaire)",
        "priority": "HIGH", "source": "Farrier Meadows"
    },
    {
        "query": "What is the cadastral location of the Farrier property?",
        "expected": "Parts of lots 7C, 8C and lot 9B, Rang 7, Canton de Rawdon, near Route 125",
        "priority": "HIGH", "source": "Farrier Meadows"
    },
    {
        "query": "What is the municipal evaluation and matricule number for the Farrier property?",
        "expected": "Municipal evaluation $154,500 (2014), Matricule 8398-41-5065, taxes $1,235.07",
        "priority": "MEDIUM", "source": "Farrier Meadows"
    },
    {
        "query": "What is the zoning of the Farrier Rawdon property?",
        "expected": "Residential zoning, potential for approximately 152 lots subdivision",
        "priority": "MEDIUM", "source": "Farrier Meadows"
    },

    # =========================================================
    # DOC 2: Bélanger à Blache (141205)
    # =========================================================
    {
        "query": "What was Blache's investment amount in the Joliette project?",
        "expected": "$100,000 invested by company 7 327 587 Canada inc. in lot 2 901 481",
        "priority": "HIGH", "source": "Bélanger à Blache"
    },
    {
        "query": "Who sent the investment confirmation to Blache and on what date?",
        "expected": "Denise Bélanger, Directrice exploitation et administration, December 5, 2014",
        "priority": "HIGH", "source": "Bélanger à Blache"
    },
    {
        "query": "What is Denise Bélanger's phone number and office address?",
        "expected": "450 438-6492, 7 rue John-F.-Kennedy local 11, Saint-Jérôme",
        "priority": "MEDIUM", "source": "Bélanger à Blache"
    },

    # =========================================================
    # DOC 3: Bélanger à Tremblay (141205)
    # =========================================================
    {
        "query": "What was Tremblay's investment amount in the Joliette project?",
        "expected": "$250,000 in Joliette project lot 2 901 481, December 5, 2014",
        "priority": "HIGH", "source": "Bélanger à Tremblay"
    },
    {
        "query": "What email address was the Tremblay investment confirmation sent to?",
        "expected": "guyt1@videotron.ca",
        "priority": "MEDIUM", "source": "Bélanger à Tremblay"
    },

    # =========================================================
    # DOC 4: VCharton Trois-Rivières Mandat (150217)
    # =========================================================
    {
        "query": "What documents were sent for Trois-Rivières Thibeau on February 17, 2015?",
        "expected": "Mandat notaire DP-0341 and Courriel investisseur, sent to Lucie Lafontaine",
        "priority": "HIGH", "source": "VCharton Mandat"
    },
    {
        "query": "What is the notary email contact for Trois-Rivières Thibeau?",
        "expected": "Lucie Lafontaine at lucie@lamoureuxnotaires.ca",
        "priority": "HIGH", "source": "VCharton Mandat"
    },
    {
        "query": "Who sent the Trois-Rivières Thibeau mandat and courriel investisseur?",
        "expected": "Véronique Charton, administrative assistant at Idéal Finance, February 17, 2015",
        "priority": "MEDIUM", "source": "VCharton Mandat"
    },

    # =========================================================
    # DOC 5: Capital Transit Convention (150225)
    # =========================================================
    {
        "query": "What changes were made to the Capital Transit and Idéal Finance convention?",
        "expected": "1) Loss responsibility changed to 100% Idéal Finance (no longer 50/50), 2) Capital Transit receives $100,000 at end of project",
        "priority": "HIGH", "source": "Capital Transit Convention"
    },
    {
        "query": "When was the Capital Transit convention modification email sent?",
        "expected": "February 25, 2015 at 13:41",
        "priority": "MEDIUM", "source": "Capital Transit Convention"
    },
    {
        "query": "What are the Capital Transit email contacts?",
        "expected": "ppapillon@capitaltransit.ca and vtremblay@capitaltransit.ca",
        "priority": "MEDIUM", "source": "Capital Transit Convention"
    },

    # =========================================================
    # DOC 6: Brompton Del Negro Loan (150715)
    # =========================================================
    {
        "query": "What is the DP-0372 loan number, amount and borrower for Brompton?",
        "expected": "Loan DP-0372, amount $700,000.00, borrower 9320-0715 Québec inc, dated July 15, 2015",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "What is the partner split for the Brompton DP-0372 loan?",
        "expected": "M Del Negro: $650,000.00 and Idéal Finance: $50,000.00",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "What is the yield percentage and term for the Brompton Del Negro loan?",
        "expected": "12% yield, term 12 months",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "What is the collateral property size and location for the Brompton loan?",
        "expected": "33,975 square feet at intersection Laval and de la Croix Nord, Brompton, Sherbrooke",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "What is the property evaluation amount for the Brompton Del Negro collateral?",
        "expected": "Evaluation $3,102,123",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "What existing mortgage is on the Brompton property?",
        "expected": "$1,100,000 second mortgage (2ième hypothèque) currently exists on Brompton",
        "priority": "HIGH", "source": "Brompton Del Negro"
    },
    {
        "query": "In the Brompton partner email from July 2015, which businesses are listed as actual tenants and what are their annual rent amounts in the lease?",
        "expected": "Tim Hortons pays $50,000 per year and Shell dépanneur poste d'essence pays $150,000 per year",
        "priority": "MEDIUM", "source": "Brompton Del Negro"
    },
    {
        "query": "What is the mortgage rank and property type for the Brompton DP-0372 loan?",
        "expected": "First rank (1er rang), commercial property with petroleum fast food zoning",
        "priority": "MEDIUM", "source": "Brompton Del Negro"
    },

    # =========================================================
    # DOC 7: Lehoux Transfer (150817)
    # =========================================================
    {
        "query": "What loan was transferred from Rawdon rue Queen to Brompton?",
        "expected": "Loan from rue Queen Rawdon notarized November 2014 was transferred to Brompton, not repaid, conditions continue",
        "priority": "HIGH", "source": "Lehoux Transfer"
    },
    {
        "query": "When was the Rawdon rue Queen loan notarized?",
        "expected": "November 2014",
        "priority": "HIGH", "source": "Lehoux Transfer"
    },
    {
        "query": "Was the Rawdon loan repaid or transferred to Brompton?",
        "expected": "Transferred (not repaid), duration and conditions continued on Brompton",
        "priority": "HIGH", "source": "Lehoux Transfer"
    },
    {
        "query": "The August 17, 2015 email confirming the Rawdon rue Queen loan was transferred to Brompton — what was the recipient email address?",
        "expected": "M. Lehoux, email sent to cjg@live.ca on August 17, 2015",
        "priority": "MEDIUM", "source": "Lehoux Transfer"
    },

    # =========================================================
    # DOC 8: Charron Planète Poutine (151204)
    # =========================================================
    {
        "query": "What is the subject of the email about the Planète Poutine lease in St-Augustin?",
        "expected": "CONTRAT DE BAIL St-Augustin de desmaures, Benoit Charron",
        "priority": "HIGH", "source": "Charron Planète Poutine"
    },
    {
        "query": "Who received the Planète Poutine bail contract email?",
        "expected": "JF Desormeaux and Isabelle Choulak received the email from Benoit Charron",
        "priority": "MEDIUM", "source": "Charron Planète Poutine"
    },
    {
        "query": "What is the filename of the attachment in Benoit Charron's December 2015 email about the bail de St-Augustin-de-Desmaures?",
        "expected": "CONTRAT DE BAIL St-Augustin de desmaures docx.docx",
        "priority": "MEDIUM", "source": "Charron Planète Poutine"
    },

    # =========================================================
    # DOC 9: Pièces Caon (2021)
    # =========================================================
    {
        "query": "What attachment was sent in the Pièces Caon email?",
        "expected": "2362-2 Pièces demanderesse.pdf",
        "priority": "HIGH", "source": "Pièces Caon"
    },
    {
        "query": "When was the Pièces Caon email sent and by whom?",
        "expected": "December 7, 2021 at 13:52 by Véronique Charton from Groupe Accretio",
        "priority": "HIGH", "source": "Pièces Caon"
    },
    {
        "query": "What is Véronique Charton's Laval office address and phone number?",
        "expected": "3221 autoroute 440 ouest local 205, Laval Québec H7P 5P2, phone 450-241-1392 poste 101",
        "priority": "HIGH", "source": "Pièces Caon"
    },

    # =========================================================
    # DOC 10: Transaction Gestion Tremblay / Charron (20150925)
    # =========================================================
    {
        "query": "What is the total price in the Gestion Immobilière Tremblay transaction with Charron?",
        "expected": "Total $2,150,000 for lots in Canton de Cathcart, approximately 33,000,000 square feet",
        "priority": "HIGH", "source": "Transaction Tremblay-Charron"
    },
    {
        "query": "In the September 2015 Tremblay-Charron cession document, what is the exact amount stated for the dépôt non remboursable paid by Claude Charron at signing — not the balance de prix de vente, specifically the initial deposit amount?",
        "expected": "$550,000 non-refundable deposit paid by Claude Charron at signing",
        "priority": "HIGH", "source": "Transaction Tremblay-Charron"
    },
    {
        "query": "What is the balance of purchase price and its interest rate in the Tremblay transaction?",
        "expected": "Balance $275,000 at 4% interest per year, payable November 15, 2014",
        "priority": "MEDIUM", "source": "Transaction Tremblay-Charron"
    },
    {
        "query": "Who are the parties to the Gestion Immobilière Tremblay transaction?",
        "expected": "9219-7771 Québec inc Gestion Immobilière Tremblay (cédant), 9303-197 Québec Inc and Claude Charron (cessionnaires)",
        "priority": "MEDIUM", "source": "Transaction Tremblay-Charron"
    },

    # =========================================================
    # DOC 11: Lamoureux Invoices / State of Disbursements (20160208)
    # =========================================================
    {
        "query": "What is the address of the Lamoureux notaires office?",
        "expected": "333 Boul. St-Martin Ouest suite 214, Laval Québec H7M 1Y7, phone 669-8998",
        "priority": "HIGH", "source": "Lamoureux Invoices"
    },
    {
        "query": "What is the total invoice amount on the Lamoureux notaires bill for preparing the prêt hypothécaire acte on lot Partie 5 681 444 Saint-Augustin in February 2016?",
        "expected": "$1,147.29 for prêt hypothécaire lot 5 681 444 Saint-Augustin",
        "priority": "MEDIUM", "source": "Lamoureux Invoices"
    },
    {
        "query": "What was the total mortgage loan amount received in the Lamoureux disbursements?",
        "expected": "$395,000 received from lender, with balance $284,301.79 remitted to borrower",
        "priority": "MEDIUM", "source": "Lamoureux Invoices"
    },

    # =========================================================
    # DOC 12: État de compte Janson 3RV (2017-05-29)
    # =========================================================
    {
        "query": "What is the loan balance for the Trois-Rivières Prairies Janson account as of May 2017?",
        "expected": "Balance $300,000 for lot 5 632 108, 1165-1167-1175 boulevard des Prairies Trois-Rivières, valid June 6, 2017",
        "priority": "HIGH", "source": "État de compte Janson"
    },
    {
        "query": "In the May 2017 état de compte for 1165-1167-1175 boulevard des Prairies Trois-Rivières, who is the individual person named as creditor owed $300,000?",
        "expected": "Jean-Pierre Janson, payable through notaire Me Pierre Aubin",
        "priority": "MEDIUM", "source": "État de compte Janson"
    },

    # =========================================================
    # DOC 13: Zonage St-Augustin (150422)
    # =========================================================
    {
        "query": "What is the zoning at the St-Augustin industrial park site?",
        "expected": "Zone IA-6 allows industry, banking, restoration services, petroleum sales per Article 4.20.2 règlement 480-85",
        "priority": "HIGH", "source": "Zonage ST-AUG"
    },
    {
        "query": "According to the April 2015 municipal règlement 480-85 document for St-Augustin zone IA-6, what hotel or accommodation use was being requested in addition to existing petroleum and restaurant permissions?",
        "expected": "Hotel motel hôtel authorization requested for zone IA-6 Saint-Augustin zoning",
        "priority": "MEDIUM", "source": "Zonage ST-AUG"
    },

    # =========================================================
    # DOC 14: BBD Offre de service (2016)
    # =========================================================
    {
        "query": "In BBD Évaluateurs Agréés' February 2016 offre de service, what is the exact 'Nos honoraires' total amount and the advance payment required before starting the evaluations for Idéal Développement?",
        "expected": "13 evaluations for $26,000 plus taxes, advance payment $6,500 required",
        "priority": "HIGH", "source": "BBD Offre de service"
    },
    {
        "query": "Who prepared the BBD evaluation offer and what is the delivery date?",
        "expected": "Philippe Lamarre, Évaluateur agréé at BBD Évaluateurs Agréés, delivery by March 16 (February 15, 2016)",
        "priority": "MEDIUM", "source": "BBD Offre de service"
    },
    {
        "query": "Which lots are included in the BBD evaluation mandate for Brompton and Saint-Augustin?",
        "expected": "Brompton lot 3102123 and Saint-Augustin-De-Desmaures lot 5681444",
        "priority": "MEDIUM", "source": "BBD Offre de service"
    },

    # =========================================================
    # DOC 15: Échéancier 3 semaines (Réunion 7)
    # =========================================================
    {
        "query": "What was JF Desormeaux's task regarding St-Lin shares by April 1?",
        "expected": "Sell 5 million dollars in shares on St-Lin and remove 3 shareholders who are not satisfied",
        "priority": "HIGH", "source": "Échéancier Réunion 7"
    },
    {
        "query": "What was the Trois-Rivières Thibeau zoning CCU date in the task schedule?",
        "expected": "Trois-Rivières Thibeau zoning change goes to CCU on April 19",
        "priority": "MEDIUM", "source": "Échéancier Réunion 7"
    },

    # =========================================================
    # CROSS-DOCUMENT TESTS
    # =========================================================
    {
        "query": "What is the total of all investments made in the Joliette lot 2 901 481 project?",
        "expected": "Blache $100,000 plus Tremblay $250,000 total $350,000, both December 5, 2014",
        "priority": "HIGH", "source": "Cross: Joliette investments"
    },
    {
        "query": "What are all the documents and projects involving Véronique Charton?",
        "expected": "Trois-Rivières Thibeau February 2015, Capital Transit February 2015, Brompton transfer August 2015, Pièces Caon December 2021",
        "priority": "MEDIUM", "source": "Cross: Véronique Charton"
    },
    {
        "query": "What are the two Idéal Finance office addresses across different years?",
        "expected": "Saint-Jérôme 7 rue John-F. Kennedy suite 11 (2014-2015), Laval 3221 autoroute 440 ouest local 205 H7P 5P2 (2017+)",
        "priority": "MEDIUM", "source": "Cross: IF addresses"
    },
    {
        "query": "What Rawdon area properties and loans are in the documents?",
        "expected": "Farrier evaluation 9,192,594 sq ft $827,000 and rue Queen loan November 2014 transferred to Brompton",
        "priority": "MEDIUM", "source": "Cross: Rawdon"
    },
    {
        "query": "What loans involve Idéal Finance as a co-investor or partner?",
        "expected": "Brompton DP-0372 Del Negro $650,000 Idéal Finance $50,000 co-investor partner",
        "priority": "MEDIUM", "source": "Cross: IF as investor"
    },
]


def extract_keywords(expected_text):
    """Extract key terms from expected answer for matching"""
    import re
    keywords = []
    keywords.extend(re.findall(r'\$[\d,\.]+', expected_text))
    keywords.extend(re.findall(r'[DP]-\d+[\-\d]*', expected_text))
    keywords.extend(re.findall(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'[a-z]* \d{1,2},? \d{4}\b', expected_text, re.IGNORECASE))
    keywords.extend(re.findall(r'\d{1,3}(?:,\d{3})+', expected_text))
    keywords.extend(re.findall(r'\b[A-Z][a-z]+\b', expected_text))
    keywords = list(set([kw for kw in keywords if len(kw) > 2]))
    return keywords[:10]


def run_test(test_case, index):
    """Run a single test case against RAG backend"""
    print(f"\n{'='*60}")
    print(f"Test #{index + 1} | {test_case['priority']} | {test_case['source']}")
    print(f"Query: {test_case['query']}")
    print(f"Expected: {test_case['expected']}")
    print("-" * 60)

    try:
        response = requests.post(RAG_URL, json={"query": test_case['query']}, timeout=60)
        if response.status_code == 200:
            data = response.json()
            rag_response = data.get('answer', 'No answer field')
            print(f"RAG Response:\n{rag_response[:300]}...")
            print("-" * 60)
            expected_keywords = extract_keywords(test_case['expected'])
            found_keywords = [kw for kw in expected_keywords if kw.lower() in rag_response.lower()]
            status = "PASS" if len(found_keywords) >= len(expected_keywords) * 0.5 else "FAIL"
            print(f"Keywords: {len(found_keywords)}/{len(expected_keywords)} | Status: {status}")
            return {
                "test_num": index + 1,
                "query": test_case['query'],
                "expected": test_case['expected'],
                "rag_response": rag_response,
                "keywords_found": found_keywords,
                "keywords_total": len(expected_keywords),
                "status": status,
                "priority": test_case['priority'],
                "source": test_case['source']
            }
        else:
            return {
                "test_num": index + 1, "query": test_case['query'],
                "expected": test_case['expected'],
                "rag_response": f"ERROR: HTTP {response.status_code}",
                "status": "ERROR", "priority": test_case['priority'], "source": test_case['source']
            }
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to RAG backend at localhost:8000")
        return {
            "test_num": index + 1, "query": test_case['query'],
            "expected": test_case['expected'],
            "rag_response": "ERROR: Connection refused",
            "status": "ERROR", "priority": test_case['priority'], "source": test_case['source']
        }
    except Exception as e:
        return {
            "test_num": index + 1, "query": test_case['query'],
            "expected": test_case['expected'],
            "rag_response": f"ERROR: {str(e)}",
            "status": "ERROR", "priority": test_case['priority'], "source": test_case['source']
        }


def save_results(results):
    output = {
        "timestamp": datetime.now().isoformat(),
        "rag_url": RAG_URL,
        "total_tests": len(results),
        "results": results
    }
    with open("e:\\Jeff\\Test\\rag_test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("RAG COMPREHENSIVE TEST SUITE — ALL INTERNAL DOCUMENTS")
    print("=" * 60)
    print(f"Backend: {RAG_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []
    for i, test_case in enumerate(TEST_CASES):
        result = run_test(test_case, i)
        results.append(result)
        save_results(results)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    high_tests = [r for r in results if r['priority'] == 'HIGH']
    high_passed = sum(1 for r in high_tests if r['status'] == 'PASS')

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)} | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")
    print(f"Score: {passed}/{len(results)} = {round(passed/len(results)*100)}%")
    print(f"HIGH Priority: {high_passed}/{len(high_tests)} = {round(high_passed/len(high_tests)*100)}%")

    # Group by source document
    sources = {}
    for r in results:
        src = r.get('source', 'Unknown')
        if src not in sources:
            sources[src] = {'pass': 0, 'total': 0}
        sources[src]['total'] += 1
        if r['status'] == 'PASS':
            sources[src]['pass'] += 1

    print("\nBy Document:")
    for src, counts in sources.items():
        pct = round(counts['pass'] / counts['total'] * 100)
        status = "✓" if pct >= 50 else "✗"
        print(f"  {status} {src}: {counts['pass']}/{counts['total']} ({pct}%)")

    print(f"\nResults saved to: rag_test_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
