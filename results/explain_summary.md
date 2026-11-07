# Explainability: Integrated Gradients on the replay-model probes (analysis only, no retraining)

Script: `scripts/explain_ig.py` (`localize`, `examples`). Log: `explain_ig_log.txt`. Tables: `explain_ig_localization.csv`
(aggregate), `explain_ig_samples.csv` (per alarm), `explain_ig_examples.csv`. Figures: `figures/fig_explain_*`.

## Method

- **Model:** the frozen replay-model B probes (`data/stage3_probes/replay_{logreg,mlp}_{y_voltage,y_overload}.joblib`),
  re-implemented in torch (standardizer + linear / 1-hidden-layer ReLU MLP, with pooled or flat features exactly as frozen).
  The torch probability matches sklearn's `predict_proba` within 1e-6 on every sample.
- **Input:** the 14 × 384 frozen bus embeddings of one sample. **Output:** the probe's logit.
- **Baseline:** per-bus mean embedding of the probe-TRAIN samples (pf_probe + dk_probe) that are *negative* for that
  label, i.e. a typical grid without that violation. 64-step trapezoid path. Completeness (attributions sum to
  logit(x) − logit(baseline)) holds to ≤ 1.6 % (median 0.2 %). The gap comes from the max/min pooling and ReLU kinks.
- **Element attribution:** bus = sum of IG over its 384 dims (signed). Branch = sum of its two end buses (the probes
  only see bus embeddings).
- **Correct alarms only:** samples where the probe fires *and* the true label is positive.
- **Localization = top-3 hit:**
  - voltage: the 3 buses with the largest attribution include a bus with a true Vm violation;
  - overload: the 3 in-service branches with the largest attribution include a truly overloaded branch. Also reported at
    bus level: the top-3 buses include an endpoint of an overloaded branch.
- **References (same top-3 rule):** chance (random top-3, exact hypergeometric), per-bus residual |PBE|, and the
  predicted-state ranking (predicted voltage violation / thermal excess, i.e. what predict-then-check points at).

## Results (`explain_ig_localization.csv`)

| Data | Label | Probe | Correct alarms | **IG top-3** | Residual | PtC state | Chance | Mean # true elements |
|---|---|---|---|---|---|---|---|---|
| PFΔ N-0/N-1 | voltage (bus) | logreg | 471 | **0.994** | 0.616 | 1.000 | 0.509 | 3.1 buses |
| PFΔ N-0/N-1 | voltage (bus) | MLP | 481 | 0.990 | 0.620 | 1.000 | 0.505 | 3.0 |
| **N-2** | voltage (bus) | logreg | 555 | **0.984** | 0.636 | 1.000 | 0.586 | 4.0 |
| **N-2** | voltage (bus) | MLP | 573 | 0.984 | 0.625 | 1.000 | 0.577 | 3.9 |
| PFΔ N-0/N-1 | overload, endpoint bus | logreg | 296 | **0.848** | 0.314 | – | 0.465 | 1.45 lines |
| PFΔ N-0/N-1 | overload, endpoint bus | MLP | 298 | 0.651 | 0.315 | – | 0.465 | 1.45 |
| **N-2** | overload, endpoint bus | logreg | 85 | **0.882** | 0.659 | – | 0.423 | 1.11 |
| **N-2** | overload, endpoint bus | MLP | 80 | 0.863 | 0.650 | – | 0.425 | 1.11 |
| PFΔ N-0/N-1 | overload, branch | logreg | 296 | 0.108 | 0.054 | 1.000 | 0.211 | 1.45 |
| PFΔ N-0/N-1 | overload, branch | MLP | 298 | 0.128 | 0.060 | 1.000 | 0.211 | 1.45 |
| **N-2** | overload, branch | logreg | 85 | 0.271 | 0.318 | 0.918 | 0.178 | 1.11 |
| **N-2** | overload, branch | MLP | 80 | 0.600 | 0.312 | 0.912 | 0.180 | 1.11 |
| datakit | voltage (bus) | both | 24 | 1.000 | 0.750 | 1.000 | 0.270 | 1.3 |
| datakit | overload, endpoint bus / branch | logreg | 37 | 1.000 / 0.000 | 0.595 / 0.000 | – / 1.000 | 0.396 / 0.158 | 1.0 |

## Findings

1. **Voltage explanations localize well and do not degrade on N-2.** The top-3 attributed buses contain a violating bus
   in 98–99 % of correct alarms (chance 51–59 %, residual 62–64 %). This holds on both PFΔ N-0/N-1 and the unseen N-2.
   Caveat: most voltage alarms have 3–4 violating buses (low voltages cluster around buses 9–14), so the chance level
   is high, and a probe that always points at the weak far end would score well too.
2. **Overload explanations find the right area, not the right line.** At bus level, the top-3 contain an endpoint of
   the overloaded line in 85 % (PFΔ) and 88 % (N-2) of alarms for logreg (chance 42–47 %). On PFΔ the overloads are
   almost all lines 1–5 and 5–6. IG points at bus 6 (and bus 12), i.e. one end of 5–6. Turned into a branch ranking
   (sum of the two end buses), the high-attribution bus 6 promotes the lines 6–11/6–12/6–13 instead, so the branch-level
   hit rate is *below* chance on PFΔ (11 %). This attribution rule, not the probe, is the weak point: bus embeddings
   cannot say which line is meant.
3. **Under N-2, overload explanations change rather than degrade.** On N-2 the overloaded lines are mostly 4–9 and 1–5.
   Bus-level localization stays at 86–88 %, and branch-level rises to 27 % (logreg) / 60 % (MLP), above chance (18 %).
   The PFΔ result was dominated by one line pattern, so N-2 is the fairer test. The explanations do not break under
   unseen outages, but the probe's *detection* does (overload recall 0.79, `final_n2_summary.md`), so explanations exist
   only for the alarms it still raises.
4. **The predicted state localizes better than any attribution.** PtC's own predicted violations hit 100 % (91–92 % for
   N-2 overloads, the alarms where the replay model's flows are off). With the replay model, predict-then-check is
   already an explanation ("bus 14 is predicted at 0.93 p.u."). IG on the probe adds a view of what the embedding
   encodes, not a better locator.
5. **The per-bus residual is a poor locator** (31–66 %, near or below chance for overloads on PFΔ). It marks where the
   model's state is physically inconsistent (often buses 9–10 in the examples), not where the violation is.

## Example scenarios (`figures/fig_explain_grid_examples.png`, `explain_ig_examples.csv`)

Picked by rule before looking at the attributions: a normal N-2 sample with median residual, the middle correct voltage
alarm without overload, the middle correct overload alarm (both N-2, logreg), and datakit sample 438 (the worst-error
sample of the original model; it has both labels, and the overload probe alarms).

| Example | Sample | Outage | p(alarm) | IG top-3 buses | True violation |
|---|---|---|---|---|---|
| normal | N-2 #404 | line 9–14 | 0.00 (voltage) | 14, 12, 6 (alarm not raised) | none |
| voltage violation | N-2 #512 | generators at buses 6 and 8 | 0.97 | 14, 6, 4 | buses 5, 10–14 low |
| overload | N-2 #429 | lines 2–5, 7–9 | 1.00 | 6, 5, 4 | line 1–5 at 104 % (+ low V at 4, 5, 9, 10, 14) |
| datakit 438 | datakit #438 | line 1–2 | 1.00 (overload) | 2, 8, 1 | line 1–5 at 316 % (bus 5 low V) |

In #429 and #438 the attribution sits at the ends of the overloaded line (bus 5 / buses 1–2), and the residual sits
elsewhere (buses 9–10 and bus 5). In #438 the probe "explains" the overload by the generator at bus 2 next to the
outaged slack line, which is physically sensible: losing line 1–2 forces the slack's power through 1–5.
