# Paper Strategy — gpt56-luna (Newest Model)

## Why Luna

The core claims (orthogonality, retrieval mechanism, gating failure, team stability)
are model-independent. Luna is the newest and most capable LLM tested. Using it
avoids the reviewer question "why such an old model?" and the smaller deltas make
the paper more honest: feedback's benefit is conditional, not universal. gpt4o-mini
and gpt54-nano results are noted in a cross-model validation paragraph.

## Central Arc

Popularity and semantics are orthogonal (r=0.06). The lift formula trades one for
the other. This helps when the baseline retrieval is procedurally wrong and hurts
when it is right. The problem: no pre-generation signal can tell which case you're
in. This gap is structural.

---

## Section 1: Introduction

RAG for IT tickets retrieves similar historical tickets to generate replies. But
text similarity is not procedural correctness. A query about "opening .ica files"
retrieves Citrix troubleshooting tickets — the correct response is a form redirect.

**The idea**: augment retrieval with human feedback. Candidates historically rated
as useful get promoted, even if less textually similar.

**The question**: when does this trade-off help, and when does it hurt?

---

## Section 2: Methodology

**Pipeline**: DistilBERT classifiers (team: 23, type: 10 labels) → FAISS retrieval
(all-MiniLM-L6-v2, 384-dim, top-5) → gpt56-luna generation via OpenRouter.

**Data**: 250 anonymised IT tickets, 27 teams, 8 types. Feedback DB: 250 queries ×
1,575 candidates = ~375K human-scored pairs (≥0.80 = useful, ≤0.40 = not useful).

**Lift formula** (Laplace-corrected Bayesian):
  p = (pos + 1) / (pos + neg + 2)
  lift = (p − 0.5) · min(1, n/2) · 0.80    (clamped ±0.20)
  enhanced_score = FAISS + lift

Top-5 by enhanced_score (desc) become the feedback retrieval.

**Methods**: M1 (global feedback), M2 (team-only), M3 (type-only), M4 (team ∩ type).

**Evaluation**: strict leave-one-out (250-fold). For each query, all feedback rows
involving that ticket are temporarily removed. Baseline and feedback share the same
candidate pool and generator; only retrieval differs. Primary metric: cosine
similarity (all-MiniLM-L6-v2) between generated reply and reference answer.

**Table 1**: Dataset — 8 types (43% "other"), 27 teams, 38% of team×type combos
have exactly 1 ticket.

---

## Section 3: The Lift Formula Is a Trade-Off

### 3.1 Retrieval Changes Fundamentally

The Laplace lift replaces candidates for 98% of tickets (mean overlap 1.7/5, only 2%
pure reorderings). 0% of promoted candidates have higher FAISS than those they
replace (added: 0.60 vs dropped: 0.69). Median FAISS rank of promoted: 17. This is
not subtle — it substantially changes the generator's evidence.

**Figure 1**: 3-panel — overlap distribution, FAISS dropped vs added, promoted-rank
histogram.

### 3.2 Popularity and Semantics Are Orthogonal

For 1,250 candidate-ticket pairs in the baseline top-5, feedback popularity
(pos−neg)/(pos+neg+1) correlates with FAISS at **r = 0.058**. Within a single
query's top-5, higher FAISS does not imply higher human rating (mean within-query
r = 0.040, σ = 0.503). Only 5% of candidates are in the top 25% of both popularity
AND FAISS — what you'd expect from independent rankings.

**Why**: IT replies are procedural. A high-FAISS candidate shares words but may
contain the wrong procedure. A lower-FAISS candidate uses different words but the
right form. Humans read the reply; FAISS reads the embedding.

**Figure 2**: 1,250-point scatter (popularity vs FAISS), flat cloud, r = 0.06.

> **Finding 1**: The lift formula trades semantic similarity for historical
> usefulness. These two signals are orthogonal.

### 3.3 Formula Choice

Laplace compared against Tanh and Bayesian LCB on 21K lifts from the feedback DB.
Laplace wins on all dimensions: σ = 0.089 (widest range), r = 0.85 with human
scores (faithful), r = 0.17 with sample size (not biased). Tanh is too narrow
(σ = 0.024). Bayesian collapses (82% zero). Full comparison in supplementary
notebook.

---

## Section 4: When Does the Trade-Off Pay Off?

### 4.1 Oracle Pattern

Stratifying by baseline answer cosine (oracle — reference required):

| Quintile | Baseline quality | Mean delta | Pct improved |
|----------|-----------------|------------|-------------|
| Q1 | 0.03–0.37 | **+0.21** | 76% |
| Q2 | 0.37–0.57 | +0.01 | 49% |
| Q3 | 0.57–0.72 | −0.02 | 50% |
| Q4 | 0.72–0.89 | −0.03 | 48% |
| Q5 | 0.89–1.00 | **−0.09** | 16% |

r = −0.41 between baseline quality and delta. Near-monotonic.

> **Finding 2**: Feedback helps when the baseline answer is weak and hurts when
> it is strong. The direction is consistent and the pattern is monotonic.

### 4.2 Can We Gate on Retrieval Signals?

**No.** Retrieval similarity predicts answer quality at r = 0.24 (explains 6% of
variance). The indirect path r ≈ 0.24 × −0.41 ≈ −0.10 is too weak.

**5-fold CV gate sweep** — find best threshold on train, evaluate on test:

| Signal | CV mean delta | vs always-on (+0.004) |
|--------|-------------|----------------------|
| Top-1 FAISS | −0.006 | −0.010 |
| Top-5 FAISS | +0.004 | 0.000 |
| Retrieval margin | −0.006 | −0.010 |

No signal beats always-on.

**Figure 3**: Side-by-side — oracle deciles (steep negative slope, r=−0.41) vs
retrieval deciles (flat, r=−0.03). Both with linear regression fits and unified y-axis.

> **Finding 3**: No deployable pre-generation signal gates feedback. The gap
> between what we predict (r=0.24) and what we need (r=−0.41) is structural.

### 4.3 Rescue vs Disruption: Indistinguishable at Retrieval Level

| Group | n | Baseline cos | Top-1 FAISS | Mean delta |
|-------|---|-------------|-------------|-----------|
| Rescues (>+0.02 δ) | 69 | 0.54 | 0.76 | +0.16 |
| Disruptions (<−0.02 δ) | 77 | 0.72 | **0.75** | −0.13 |

Both groups have near-identical top-1 FAISS (~0.75). Retrieval cannot tell rescue
from disruption. Baseline answer quality differs substantially (0.54 vs 0.72) —
but FAISS cannot see this difference.

---

## Section 5: Supporting Evidence

### 5.1 Method Ordering: M4 > M1 > M2 > M3

| Method | Mean delta | Notes |
|--------|-----------|-------|
| M4 team∩type | **+0.010** | Strictest scope, benefits from luna's generation quality |
| M1 global | +0.004 | Largest evidence pool |
| M2 team-only | +0.003 | — |
| M3 type-only | −0.002 | — |

M4 outperforms M1 with luna (unlike mini where M1 > M4). This suggests stronger
generators extract more value from precise feedback. However, M4 collapses for
33% of tickets because 38% of team×type combos have only 1 ticket — evidence
sparsity, not scope quality.

### 5.2 Team Identity: Weak but Cross-Scope Stable

Team delta rankings are correlated across M1-M4 (r = 0.68–0.87). (GI-UX) Group
benefits under all four scopes. (GI-SaaS) Salesforce is harmed under all four.
Cross-model validation with mini and nano confirms r > 0.71. This stability
supports team identity as a structural signal — but it is dataset-specific and
serves as a supporting prior, not a standalone gate.

**Figure 4**: Horizontal grouped bar chart — per-team delta across M1-M4.

---

## Section 6: Discussion

**Contributions**: (1) We quantify the orthogonality between feedback popularity
and semantic similarity — a fundamental property of procedural data. (2) We show
the oracle pattern and rigorously test deployable gating approximations. (3) We
establish the structural gap: no pre-generation signal bridges the distance between
retrieval similarity (r=0.24) and answer quality (r=−0.41).

**Cross-model agreement** (gpt4o-mini, gpt54-nano): all findings hold in direction.
Magnitude varies with generator capability (Appendix A).

**Limitations**: single dataset, single feedback protocol, 250 query tickets.
Team/type gating signals are dataset-specific.

**Future work**: a learned reliability estimator combining retrieval features,
team/type priors, and feedback evidence into a pre-generation gate.

---

## Section 7: Conclusion

Human feedback improves retrieval-augmented ticket resolution, but only when the
baseline retrieval is procedurally wrong — a case that cannot be identified from
any pre-generation retrieval signal. This gap is structural: retrieval similarity
and feedback usefulness are orthogonal dimensions of IT support data. A learned
model combining multiple weak signals is the necessary next step.

---

## Figures (4)

1. **3-panel retrieval change** — overlap, FAISS comparison, promoted ranks
2. **Popularity vs FAISS scatter** — 1,250 points, r=0.06
3. **Oracle vs retrieval deciles** — side-by-side with linear fits
4. **Cross-scope team bar chart** — M1-M4 per-team deltas

## Tables (3)

1. Dataset statistics
2. Method comparison (M1-M4) with oracle pattern
3. CV gate sweep — all signals fail
