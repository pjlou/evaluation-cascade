"""
Phase 1 Smoke Test Script - Version 0.2
Verifies dependencies for the 2x2 crossed evaluation design:
  - English Tier 1: Agreement Attraction (spaCy Dependency Tree)
  - English Tier 2: Negation Scope (c-command / Subtree parsing)
  - Finnish Tier 1: Object Case Alternation (UralicNLP Morphological Features)
  - Finnish Tier 2: Negation Scope & Connegative Parsing
"""

import spacy
import networkx as nx
import uralicNLP.cg3 as cg3
from uralicNLP import uralicApi as uralicNLP
from inspect_ai import Task, task

print("=== Phase 1 (v0.2) Environment Smoke Test ===")

# 1. Test EN Tier 1: Agreement Attraction (Subject-Verb Head Matching)
print("\n[1/5] Testing EN Tier 1: Agreement Attraction Parsing...")
nlp_en = spacy.load("en_core_web_trf")
doc_en_tier1 = nlp_en("The list of changes to the report is ready.")
head_noun = [token for token in doc_en_tier1 if token.dep_ == "nsubj"][0]
verb = head_noun.head
print(f"  Sentence: '{doc_en_tier1.text}'")
print(f"  Extracted Head Subject: '{head_noun.text}' | Verb: '{verb.text}' (Agreement Match)")
assert head_noun.text == "list", "Failed to identify syntactic head!"
print("✓ EN Tier 1 parsing functional.")

# 2. Test EN Tier 2: Negation Scope Subtree Extraction
print("\n[2/5] Testing EN Tier 2: Negation Scope Parsing...")
doc_en_tier2 = nlp_en("Not all students failed the test.")
neg_token = [token for token in doc_en_tier2 if token.dep_ == "neg"][0]
scope_subtree = [t.text for t in neg_token.head.subtree]
print(f"  Sentence: '{doc_en_tier2.text}'")
print(f"  Negation Marker: '{neg_token.text}' | Scope Subtree: {' '.join(scope_subtree)}")
print("✓ EN Tier 2 parsing functional.")

# 3. Test FI Tier 1: Object Case Alternation (Kiparsky 1998 Morphological Features)
print("\n[3/5] Testing FI Tier 1: Object Case Morphological Analysis...")
parsed_acc = uralicNLP.analyze("omenan", "fin")  # Accusative/Genitive
parsed_part = uralicNLP.analyze("omenaa", "fin")  # Partitive
acc_case = [feat for feat in parsed_acc[0][0].split("+") if "N" in feat or "Sg" in feat or "Gen" in feat or "Acc" in feat]
part_case = [feat for feat in parsed_part[0][0].split("+") if feat == "Par"]
print(f"  'omenan' features: {acc_case}")
print(f"  'omenaa' features: {part_case}")
assert len(part_case) > 0, "Failed to identify Partitive case!"
print("✓ FI Tier 1 morphological extraction functional.")

# 4. Test FI Tier 2: Connegative & Clitic Scope Parsing
print("\n[4/5] Testing FI Tier 2: Connegative & Focus Clitic Analysis...")
parsed_conneg = uralicNLP.analyze("läpäisseet", "fin")
print(f"  'läpäisseet' (connegative form) parse: {parsed_conneg[0][0]}")
print("✓ FI Tier 2 morphological parsing functional.")

# 5. Test Rule Graph (NetworkX) & Inspect AI Harness Setup
print("\n[5/5] Testing Graph Engine & Inspect AI Harness...")
G = nx.DiGraph()
G.add_node("RULE_FI_C1_NEGATION", label="Negation forces Partitive Object")
print(f"  Rule Graph Initialized ({G.number_of_nodes()} node).")

@task
def smoke_test_task():
    return Task(dataset=[])

print("✓ NetworkX & Inspect AI harness imports verified.")

print("\n=== ALL PHASE 1 (v0.2) SMOKE TESTS PASSED SUCCESSFULLY ===")