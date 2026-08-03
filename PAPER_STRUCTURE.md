# Paper Structure and Strategy

## Title

**When Does Human Feedback Help Retrieval-Augmented Ticket Resolution?**

## Central Narrative Arc

Two signals drive retrieval: semantic similarity (FAISS) and historical usefulness (feedback
popularity). They are orthogonal. The lift formula trades one for the other for 98% of
tickets. Whether the trade-off helps or hurts depends on baseline answer quality — but
no pre-generation signal can predict baseline answer quality. The gap between what we
can measure (retrieval similarity) and what we need to know (answer quality) is
structural, withstood cross-validation on every deployable signal, and persists across
three different LLMs. A learned gate combining multiple weak signals is the only path
forward.

---

## Section 1: Introduction (~0.5 pages)

**Hook**: IT support tickets are repetitive but not perfectly repetitive. RAG systems
retrieve similar historical tickets and generate replies. But text similarity is not
procedural correctness. A ticket about "opening .ica files in Citrix" shares words with
troubleshooting tickets, but the correct response is a form redirect to software
installation.

**The idea**: augment retrieval with historical human feedback. Candidates that were
useful in past support contexts get promoted, even if they're less textually similar.

**The tension**: this is a trade-off. When semantic retrieval already finds the right
procedure, promoting popular but different candidates disrupts a good answer. When
semantic retrieval finds the wrong procedure, the same promotion rescues the generation.

**Research questions**: (1) When does feedback help, and when does it hurt? (2) Can
we predict this before generation? (3) Does the answer depend on the LLM?

**Main claim**: feedback is a conditional retrieval prior, not an always-on correction.
The condition — whether the baseline retrieval is procedurally correct — cannot be
predicted from retrieval-level signals alone.

---

## Section 2: Methodology (~0.5 pages)

- **System**: RAG pipeline (DistilBERT classifiers → FAISS retrieval with all-MiniLM-L6-v2 → GPT generation via OpenRouter). 250 anonymous IT tickets, 27 teams, 8 types.
- **Feedback**: 250 queries × 1,575 candidates. Human raters scored each candidate (≥0.8 = useful, ≤0.4 = not useful).
- **Lift formula**: Laplace-corrected Bayesian estimator. p = (pos+1)/(pos+neg+2), lift = (p−0.5)·min(1,n/2)·0.80, clamped ±0.20. enhanced_score = FAISS + lift.
- **Methods compared**: M1 (global), M2 (team-only), M3 (type-only), M4 (team∩type).
- **Evaluation**: strict leave-one-out (250 fold), identical candidate pool, same generator. Primary metric: cosine similarity of generated reply to reference answer.
- **Models**: gpt4o-mini, gpt54-nano, gpt56-luna. Luna reported as primary (newest); mini and nano as cross-model validation.

**Figure 1**: System diagram showing baseline retrieval path vs feedback retrieval path.

---

## Section 3: The Orthogonality Mechanism (~1 page)

### 3.1 The Lift Fundamentally Changes Retrieval

The Laplace formula promotes candidates from FAISS ranks 7-100+ into the top-5. 98% of
tickets get at least one completely new candidate. 0% of those new candidates have
higher FAISS scores than the ones they replace (mean FAISS: added 0.60 vs dropped 0.69).

**Figure 2**: 3-panel: overlap distribution, FAISS comparison histogram, promoted-rank histogram.

### 3.2 Popularity and Semantic Relevance Are Orthogonal

The core mechanism: feedback popularity `(pos−neg)/(pos+neg+1)` and semantic similarity
(FAISS score) are uncorrelated at r = 0.06 across 1,250 candidate-ticket pairs. Even
within a single query's top-5, higher FAISS does not imply higher human rating
(mean within-query r = 0.04, 51% positive, 49% negative).

**Why**: IT replies are procedural. A high-FAISS candidate shares words with the query
but may contain the wrong procedure. A lower-FAISS candidate may use different words
but contain the correct form redirect. Human raters read the reply content; FAISS only
sees the embedding.

**Figure 3**: Scatter plot of 1,250 points (popularity vs FAISS), with within-query
examples (R-1, R-10, R-100).

> Claim 1: Feedback popularity and semantic relevance are orthogonal signals (r=0.06). The lift formula trades one for the other for 98% of tickets.

### 3.3 Formula Comparison

Three lift formulas compared on 21K candidate lifts from the feedback DB. Laplace is
the only viable choice: σ=0.089 (widest dynamic range), r=0.85 with human scores,
r=0.17 with observation count (not sample-size biased). Tanh is too conservative
(σ=0.024). Bayesian LCB collapses to zero for 82% of candidates.

> Claim 2: The Laplace formula is statistical justified: it maximises discrimination while faithfully representing human scores and controlling sample-size bias.

---

## Section 4: The Oracle Signal and the Gating Problem (~1 page, luna primary)

### 4.1 When Does Feedback Help?

Stratifying by baseline answer quality (oracle diagnostic) reveals a clean monotonic
pattern: +0.21 mean delta in the lowest baseline-quality decile (76% of tickets
improved), −0.09 in the highest decile (84% worsened). The pattern holds across
all 3 models: mini (+0.065 to −0.041), nano (+0.048 to −0.028), luna (+0.043 to −0.035).

The magnitude shrinks with better generators because baseline quality improves,
leaving less room for feedback to help. But the direction never reverses.

> Claim 3: Feedback helps when the baseline answer is weak and hurts when it is strong. The effect is consistent across 3 LLMs and follows a monotonic oracle pattern.

### 4.2 Can We Predict This Before Generation?

**No.** We tested 6 deployable pre-generation signals with 5-fold cross-validated
threshold sweeps. Every signal collapses on held-out folds: CV gate δ = +0.0005
(mini), +0.007 (nano), −0.006 (luna) vs always-on δ = +0.012, +0.010, +0.004.

**Why**: retrieval similarity predicts answer quality at r=0.24 (luna), r=0.38 (mini).
Answer quality predicts feedback success at r=−0.41 (luna), r=−0.44 (mini). The
indirect path is r ≈ 0.24 × −0.41 ≈ −0.10 — a retrieval gate sees only ~10-15% of
the oracle's signal.

**Rescue vs disruption profiles** show why: rescues (>+0.02 delta) and disruptions
(<-0.02 delta) have nearly identical top-1 FAISS scores (0.76 vs 0.75). Retrieval
physically cannot distinguish which ticket will benefit from which will be harmed.

**Figure 4**: Side-by-side oracle decile plot (strong negative slope) vs retrieval
decile plot (flat line). Rescue/disruption profile table.

> Claim 4: No deployable pre-generation signal can gate feedback. The gap between
> what we can predict (retrieval similarity) and what we need to predict (answer quality)
> is structural. A simple threshold gate on any retrieval signal fails under cross-validation.

### 4.3 Model-Independent Cross-Check

Claims 1-2 (retrieval mechanism) are model-independent — they use only the FAISS
index and feedback database, identical across all LLM runs. Claim 4 (gating failure)
holds across all 3 models. Claim 3 (oracle pattern) has consistent direction with
varying magnitude.

> Claim 5: The core retrieval findings are model-independent. The answer-level findings
> hold in direction across gpt4o-mini, gpt54-nano, and gpt56-luna, with varying magnitude.

---

## Section 5: What the Scoped Models Teach Us (~0.5 pages)

M1 > M2 > M3 > M4 with gpt4o-mini. M4 > M1 > M2 > M3 with gpt56-luna. The
ordering flips because stronger generators extract more value from precise feedback
scoping.

But the key lesson is about evidence sparsity: M4 collapses for 33% of tickets because
the intersection of team and type produces groups too small for meaningful lift
computation (38% of team×type combos have exactly 1 ticket). This is a data
availability problem, not a scope quality problem.

**Figure 5**: Horizontal grouped bar chart showing per-team delta across M1-M4.

> Claim 6: Narrower feedback scopes are limited by evidence availability, not scope
> quality. The optimal scope depends on both data density and generator capability.

---

## Section 6: Discussion (~0.5 pages)

**Contributions**: (1) We quantify the orthogonality between popularity and semantics
in procedural IT data — a fundamental property, not a system artefact. (2) We
characterise the oracle signal and rigorously test deployable approximations, finding
none viable. (3) We validate findings across 3 LLMs, establishing model-independence
for the core mechanism.

**Limitations**: single dataset (one enterprise), single feedback collection protocol
(FAISS top-5 rating), limited to 250 query tickets. Team/type gating signals are
dataset-specific and may not transfer.

**Future Work**: a learned gating model that combines retrieval features (top-k
similarity, retrieval margin, candidate agreement), team/type priors, and
feedback-evidence strength into a pre-generation reliability estimator. The
orthogonality finding suggests this model should estimate procedural correctness directly,
not approximate it through text similarity. Cross-model validation shows that
a well-calibrated gate needs to account for generator capability.

---

## Section 7: Conclusion (~0.25 pages)

Human feedback improves retrieval-augmented ticket resolution, but only when the
baseline retrieval is procedurally wrong. The central challenge is that no
pre-generation retrieval signal can distinguish this case from the case where
the baseline is already correct. This gap is structural — it arises from the
orthogonality of text similarity and procedural usefulness, which we show is a
fundamental property of IT support data. A learned reliability model combining
multiple weak signals is the necessary next step.

---

## Claims Table (for reference)

| # | Claim | Evidence | Model Status |
|---|---|---|---|
| C1 | Popularity ⊥ semantics (r=0.06) | 1,250-pair scatter | Independent |
| C2 | Laplace formula justified | 21K lift comparison | Independent |
| C3 | Feedback helps weak baseline, hurts strong (oracle) | Decile analysis | 3-model confirm |
| C4 | No pre-generation signal gates feedback | 5-fold CV, all fail | 3-model confirm |
| C5 | Core retrieval findings model-independent | 3-model comparison | 3-model confirm |
| C6 | Scoped models limited by evidence, not scope quality | Combo count + M4 collapse | Independent |