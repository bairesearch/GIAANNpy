# Beam search review and improvement suggestions

## Observations
- Beam expansion clones the full activation, connection, and time tensors per candidate, which can be expensive when beams branch at depth >1 and the sparse tensors are large. The current loop clones state for every candidate before computing activation gains and scores, potentially stressing memory and GPU bandwidth.
- Candidate scoring currently depends on `inferenceBeamScoreStrategy` and returns `None` for any unrecognized strategy value, which would propagate a `None` score into later sums. The scoring function also lacks explicit normalization for beam depth or sequence length, favoring beams with more steps when activation gains stay positive.
- Concept-column candidate selection aggregates activations per column and now records the total activation per column for scoring when `inferenceBeamScoreStrategy` is `nodeActivation`, while instance-node candidate selection still relies on raw activation/connection values without considering temporal decay or recency stored in `state["time"]`.
- The prediction intentionally emits the first node set (depth 0) from the highest-scoring beam path; ensuring completed beams participate in best-path selection avoids missing higher-scoring paths that terminate early.

## Recommendations
- Reuse activation deltas when possible by deferring `cloneBeamActivationState` until a beam actually survives pruning, or by storing lightweight deltas per node and applying them lazily when a beam is promoted to the next depth. This reduces cloning pressure during large search widths.
- Make `computeBeamNodeScore` return a safe default (e.g., activation-based) when `inferenceBeamScoreStrategy` is unrecognized, and consider adding a length-normalization or coverage term (such as `newScore / (len(sequence)+1)`) so deeper beams are not automatically favored when activation gains remain positive.
- Incorporate activation magnitude into the concept-column candidate score (e.g., blend mean activation and connection strength for the connection-based strategy) and optionally weight by recency from `state["time"]` to bias toward fresher context. A similar weighting could be applied in `selectBeamCandidatesInstanceNodes` to disfavor stale activations even when raw values remain high.
- If future tasks need deeper-branch guidance, expose a toggle to select which position in `bestBeam["sequence"]` should be emitted (first, last, or argmax), leaving the current depth-0 emission as the default.
