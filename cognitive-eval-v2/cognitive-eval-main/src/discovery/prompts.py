# src/discovery/prompts.py
"""
Free-form generation prompts for Cascade Stage 2 (statistical failure discovery).

These are deliberately NOT multiple-choice and NOT scored against a gold
label. They exist to produce open-ended text that can be embedded and
clustered to surface failure patterns we haven't already written a Tier 1
rule for -- including patterns tied to lexical items and phrasings that
don't appear anywhere in the graded dataset.

Two families, mirroring the two Finnish phenomena already in the graded
suite, but deliberately using nouns/quantifiers NOT already covered there
(the graded set uses omena/kirja/vesi and kaikki/kukaan -- this set uses
different lexical items so discovered clusters aren't just re-finding
memorized graded examples).
"""

from typing import NamedTuple


class DiscoveryPrompt(NamedTuple):
    id: str
    family: str  # "case_completion" | "negation_paraphrase"
    prompt: str


FREE_GENERATION_PROMPTS: list[DiscoveryPrompt] = [
    # --- Free object-case completion (no options given) ---------------
    # Atelic / ongoing contexts -> partitive expected
    DiscoveryPrompt("disc-case-001", "case_completion",
        "Täytä lause luonnollisesti suomeksi: 'Söin ___ (kala).' "
        "Vastaa vain täydennetyllä sanalla."),
    DiscoveryPrompt("disc-case-002", "case_completion",
        "Täytä lause luonnollisesti suomeksi: 'Join ___ (mehu).' "
        "Vastaa vain täydennetyllä sanalla."),
    # Telic / completed contexts -> accusative expected
    DiscoveryPrompt("disc-case-003", "case_completion",
        "Täytä lause luonnollisesti suomeksi, kun teko on valmis: "
        "'Söin koko ___ (kala) illalliseksi.' Vastaa vain täydennetyllä sanalla."),
    DiscoveryPrompt("disc-case-004", "case_completion",
        "Täytä lause luonnollisesti suomeksi, kun teko on valmis: "
        "'Luin ___ (lehti) kokonaan.' Vastaa vain täydennetyllä sanalla."),
    # Negated contexts -> partitive expected regardless of telicity
    DiscoveryPrompt("disc-case-005", "case_completion",
        "Täytä kielteinen lause luonnollisesti suomeksi: 'En syönyt ___ (kala).' "
        "Vastaa vain täydennetyllä sanalla."),
    DiscoveryPrompt("disc-case-006", "case_completion",
        "Täytä kielteinen lause luonnollisesti suomeksi: 'En lukenut ___ (lehti).' "
        "Vastaa vain täydennetyllä sanalla."),
    # Mass-noun / unquantized objects -> partitive expected even if telic-sounding
    DiscoveryPrompt("disc-case-007", "case_completion",
        "Täytä lause luonnollisesti suomeksi: 'Join koko illan ___ (vesi).' "
        "Vastaa vain täydennetyllä sanalla."),

    # --- Free negation-scope paraphrase (no options given) -------------
    # kaikki (universal) negated -> "not all" reading expected
    DiscoveryPrompt("disc-neg-001", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Kaikki eivät tulleet juhliin.'"),
    DiscoveryPrompt("disc-neg-002", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Kaikki oppilaat eivät osanneet vastausta.'"),
    # kukaan (negative-polarity existential) negated -> "none" reading expected
    DiscoveryPrompt("disc-neg-003", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Kukaan ei tullut juhliin.'"),
    DiscoveryPrompt("disc-neg-004", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Kukaan oppilaista ei osannut vastausta.'"),
    # moni / harva -- quantifiers NOT covered anywhere in the graded set;
    # included specifically to see whether new, ungraded failure patterns
    # show up on quantifiers the rule graph has no node for yet.
    DiscoveryPrompt("disc-neg-005", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Monet eivät tulleet juhliin.'"),
    DiscoveryPrompt("disc-neg-006", "negation_paraphrase",
        "Selitä omin sanoin englanniksi, mitä tämä lause tarkoittaa: "
        "'Harvat oppilaat osasivat vastauksen.'"),
]
