# Cognitive-Eval: Verification Cascade Roadmap

## Where this starts

The project currently has one verification tier: rule-based, deterministic,
forced-choice. Every item — English and Finnish, all phenomena — now runs
through the same `extract_final_choice` pattern against a hand-authored rule
graph. That's a real and correct simplification: it puts every phenomenon on
equal footing and removes the production-vs-selection confound from the
dataset entirely.

It also has a side effect worth naming honestly: the project can no longer
measure whether a model can *produce* a correctly inflected Finnish word form
from scratch — only whether it can recognize the right one when shown both
options. That's a deliberate trade-off, not a loss to quietly paper over. The
plan below reintroduces free-form generation, but as raw material for
unsupervised discovery rather than as a second graded task — which avoids
reopening the original confound while still getting value from open-ended
output.

## The organizing idea

"Never LLM-as-judge" was the wrong frame from the start, by your own
diagnosis. The right frame is a **verification cascade**: try the cheapest
method that can actually resolve the question, and escalate only when it
can't.

```
Cascade Stage 1 (existing)   Rule-based / deterministic verification
                     — dependency parse, morphological tags, rule graph
                     — cheapest, fastest, fully auditable
                     — only works where a phenomenon has a known rule

Cascade Stage 2 (new)        Statistical / embedding-based failure discovery
                     — unsupervised, no rule graph required
                     — finds failure PATTERNS, not pass/fail on a known item
                     — this is the tier that closes the mock-interview gap

Cascade Stage 3 (new)        Model-based judgment, deliberately scoped
                     — used only where Cascade Stage 1 genuinely can't apply
                     — rubric fixed and disclosed, cross-validated against
                       a small hand-labeled set to report judge reliability

Cascade Stage 4 (documented, not built)  Human review
                     — logged as the fallback when Cascade Stage 3 disagrees
                       with itself across runs or confidence is low
```

Building Cascade Stages 2 and 3, plus the routing logic between all of them, is
the whole plan. Each one maps directly to something specific from the mock
interviews and the reframing conversation.

*(Naming note: "Cascade Stage" is used throughout, rather than "Tier," to
avoid colliding with this project's existing Tier 1/Tier 2 phenomenon
terminology — morphological vs. clausal. Same fix was made in the actual
code and README when Phase A shipped.)*

---

## Phase A — Statistical failure discovery (highest priority)

This is the direct fix for the Reducto/ByteDance gap: finding failure
patterns in large, uncurated output, not just scoring known test items.

**What it does:** generate a much larger set of free-form outputs — have
models translate, paraphrase, or freely complete Finnish sentences involving
your existing phenomena (case alternation, negation) without multiple-choice
scaffolding — embed the outputs (sentence-transformers), cluster them
(k-means and DBSCAN, compare both), and inspect which clusters correspond to
genuine, coherent failure types versus noise.

**Why free generation belongs here and not in the graded dataset:**
clustering is unsupervised — it doesn't need a gold label to be useful. This
is exactly the right place for the production-based task you moved away from
in the main dataset: not as a second accuracy number to report, but as raw
material for discovering failure types you haven't already written a rule
for.

**Concrete build:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(raw_model_outputs)

# Compare both — DBSCAN surfaces outliers KMeans would force into a cluster
kmeans_labels = KMeans(n_clusters=8, random_state=0).fit_predict(embeddings)
dbscan_labels = DBSCAN(eps=0.4, min_samples=3).fit_predict(embeddings)

# 2D projection for the dashboard, not for the analysis itself
coords = PCA(n_components=2).fit_transform(embeddings)
```

**The closing move — this is the part that makes it more than a tooling
exercise:** once a cluster is manually inspected and understood well enough
to name (e.g., "models frequently substitute the illative for the partitive
in atelic contexts with motion verbs" — a real pattern, not hypothetical,
given what you're already seeing in the case-alternation logs), *codify it
as a new deterministic rule graph node (Cascade Stage 1).* Discovery happens
once, at Cascade Stage 2 cost; detection happens forever after, at Cascade
Stage 1 cost. That loop — expensive discovery, cheap ongoing detection — is
the compute-proportionate philosophy made concrete rather than asserted, and
it's a genuinely strong thing to walk an interviewer through.

**Deliverable:** a notebook/script plus a short write-up: here's a cluster
that turned out to be a real, previously-unknown failure mode; here's the
rule graph node it became.

**Effort estimate:** the smallest, highest-leverage piece of this whole
roadmap — a few days, not weeks, and it's the one to do first regardless of
how much time you have for the rest.

---

## Phase B — Cascade routing + a deliberately scoped Cascade Stage 3

> **Status update:** Rescoped and substantially complete. Given the shift
> toward evaluation-engineering roles (see Evaluation Cascade spec),
> compositional logic as a *new linguistic phenomenon* was dropped from
> this phase — it would have required new dataset authoring, which is now
> explicitly deprioritized. What shipped instead: a generic,
> application-agnostic dispatcher (`src/cascade/`) implementing the same
> Stage 1→2→3→4 routing, tested against synthetic fixtures rather than new
> real content, and deliberately schema-compatible with Evaluation
> Cascade's FR-3/FR-7 so it's directly portable rather than
> Cognitive-Eval-only work. The dispatcher is now wired into the Inspect
> scorer, and Cascade Stage 3 has a real Ollama-backed judge implementation.
> Remaining: a dashboard panel showing resolution rate per stage. That panel
> is deferred until cascade run logs exist from the new scorer path.

This is where the compositional-logic phenomenon (already flagged as future
work) gets built — and where you demonstrate you understand *when* an
LLM-judge is the right tool, answering the ByteDance question directly.

**Routing logic** — one dispatcher, with logging of which stage resolved each
item:
```python
def verify(item, model_output):
    if item.rule_node_id and rule_graph.has_node(item.rule_node_id):
        result = verify_via_rule(item, model_output)          # Cascade Stage 1
        if result.resolved:
            return result

    if item.phenomenon in STATISTICALLY_SCOPED_PHENOMENA:
        result = verify_via_similarity(item, model_output)     # Cascade Stage 2
        if result.confident:
            return result

    return verify_via_llm_judge(item, model_output)            # Cascade Stage 3
```

**Cascade Stage 3, scoped narrowly:** compositional logic only, with a fixed,
disclosed rubric — not a general-purpose "ask an LLM if this is right."
Cross-validate the judge against a small hand-labeled set (even 15–20 items)
and report agreement (precision/recall of the judge against your own
labels). That meta-evaluation step — measuring the evaluator, not just using
it — is itself a distinct, resume-worthy skill and directly answers "how do
you know your judge is trustworthy" before anyone has to ask.

**Dashboard addition:** a panel showing what fraction of items were resolved
at each stage. "83% resolved at Cascade Stage 1, no LLM call needed" is a
concrete, quantifiable compute-proportionality claim — far stronger than
describing the philosophy in prose.

**Effort estimate:** larger than Phase A — this is a second full sub-feature,
not a quick addition. Reasonable to treat as "in progress" on your resume
rather than something to finish before your current application cycle
resolves.

---

## Phase C — Documentation catch-up (do this regardless of how far A/B get)

- Update the README/report to state plainly that all items are now
  forced-choice, and why (removes the format confound; the earlier blog
  draft's "next steps" section should be revised to point at Phase A's
  free-generation clustering as where production ability now lives, rather
  than implying it'll return to the graded dataset).
- Update the About-section language and the blog draft with the
  compute-proportionate framing once Phase A produces a real example —
  "here's a failure mode I found by clustering, and the rule it became" is a
  much stronger blog post than the philosophy alone.

---

## Phase D — Linguistic formalization (three options, not equal priority)

> **Status update: postponed indefinitely.** This entire phase depended on
> deepening the linguistics-specific differentiation (formal semantics,
> the aspect/case calculus, morphology breadth) that the evaluation-
> engineering pivot has deliberately moved away from. Not rejected as an
> idea — genuinely worth returning to once Finnish proficiency and
> priorities make it relevant again — but it should not receive further
> planning time until then.

This phase is about replacing hand-labeled gold answers with something that
*derives* the answer from linguistic structure — a qualitatively different
kind of rigor than anything built so far, since even the rule graph currently
depends on a human deciding each item's label by hand.

### D1 (do this first) — Extend formal semantics cross-linguistically

The original project design (back when it was English-only) planned to cover
quantifier scope ambiguity, double negation, and compositional logic. Only
one piece of that — a single negation construction per language — actually
got built. This is the most direct test of the project's actual thesis
("is structural competence genuine or a surface artifact of English-heavy
training") that the project can offer, because quantifier scope and negation
are the *same logical operation* realized through unrelated grammatical
machinery in English (word order, auxiliary negation) versus Finnish (case
marking, connegative constructions, focus clitics). Extending this properly —
real multi-quantifier scope ambiguity, double negation, simple conditionals,
tested in both languages — would make this the flagship demonstration of the
whole project, not a single data point in it.

This also reuses Phase A's discovery infrastructure directly: extend the free
generation prompt set in `src/discovery/prompts.py` to cover richer
quantifier/logic constructions, and the same embed-and-cluster pipeline can
surface formal-semantic failure patterns, not just morphological ones.

### D2 (do this second, and only as a generator, not a live verifier) — Compositional aspect/case calculus

Build a small pipeline that parses a sentence, derives lexical aspect + NP
boundedness + polarity + adverbial class (e.g. *kokonaan* forces telic,
*tunnissa* vs. *tunnin* flips telic/atelic) as features, then applies
Kiparsky's C1–C4 rules to produce the predicted case. This is a genuinely
different order of rigor than the current dataset: today a human decides
"this item is Aspect=Atelic" when writing the JSON; a calculus *derives* that
label from the sentence itself.

The real payoff isn't a more sophisticated single test — it's that this
solves dataset scaling in a principled way. Feed the calculus new verb/noun/
adverbial combinations and it generates correctly-labeled minimal pairs
automatically, instead of hand-authoring each one.

Scope this as an offline generator you run to produce test items with
defensible labels — keep it separate from live verification, since getting
the aspectual classification subtly wrong would quietly mislabel everything
downstream, and that's a harder failure mode to catch than a wrong test item
written by hand. This is a multi-week build, not a few days, because getting
verb lexical class and NP quantization right requires real linguistic care.

Worth noting: D1 and D2 are the same underlying move — replace a hand-labeled
item with a small compositional engine that derives the answer — applied to
two different linguistic domains (quantifier/negation logic vs. the
morphosyntax-aspect interface). Building the generator pattern once for D2
makes extending it to D1 later meaningfully cheaper.

### D3 (cheap footnote, not a priority) — Broader morphology as a confound control, not a reasoning test

Consonant gradation, vowel harmony, and the full local case system (illative/
inessive/elative, etc.) are mechanical, rule-governed phonological
alternations — a model either has memorized the correct forms or it hasn't.
That's a test of raw linguistic exposure, not reasoning, and should be framed
that way rather than as a cross-linguistic reasoning result.

Where it does earn a place: as a baseline that separates two currently
conflated explanations for a failure. If a model fails object case
alternation, there's no way today to tell whether that's a reasoning failure
(doesn't track aspect) or a competence floor (doesn't reliably produce
Finnish morphology at all, so of course everything downstream fails). A small,
cheap battery of basic morphology sanity checks — cheaper to build than D2,
since the rules are more mechanical and less ambiguous — lets that
distinction get made before interpreting the harder results.

The model-selection angle here is nearly free and worth doing regardless of
whether the rest of D3 happens: several of these models' technical reports or
model cards disclose approximate training-data language composition.
Checking what's documented about Finnish/Uralic-language representation in
Qwen2.5, Llama 3.1, and Ministral's training mix would let the project state
*why* one family is expected to outperform another on Finnish specifically,
rather than only observing that it does.

---

## Priority, given your timeline

Do Phase A now — it's small, it's the direct fix for the specific gap that
showed up in two separate mock interviews, and it's genuinely usable as an
in-progress talking point even half-finished ("I'm currently building an
unsupervised layer on top of the rule-based tier to catch failure modes I
haven't hand-written rules for yet" is a strong, honest, live answer in an
interview happening this week).

Treat Phase B as a stretch goal you work on between applications, not a
blocker to anything — if it's still "in progress" when an offer comes
through, that's a fine place to leave it. Phase C is cheap and should happen
alongside whichever of A or B you actually finish, not held for both.

Phase D sits after B, not before it — it's a bigger, more open-ended
investment than either A or B, and D1 in particular depends on Phase A's
discovery infrastructure already existing. If you only get to one piece of
Phase D before your timeline forces a decision, make it D1 (cross-linguistic
formal semantics): it's the one that most directly proves the claim the
project is actually built around, and it's the natural next chapter for the
blog thread once Phase A produces its first real cluster-to-rule story. D2 is
worth real investment only once D1's scope is clear, since building the
generator pattern for D2 first makes D1 cheaper, not the other way around.
D3 is the one thing in this entire roadmap you should feel free to do in a
spare afternoon whenever one appears, with no need to sequence it against
anything else.
