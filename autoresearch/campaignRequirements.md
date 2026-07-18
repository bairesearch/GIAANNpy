# Campaign Requirements

The Campaign Requirements provide campaign specific settings for configuring the autoresearch baseline, and monitoring, evaluating and recording experiments.

These are the user requirements specific to this autoresearch campaign:

1. When monitoring autoresearch train/eval runs, just wait until they are finished before you think about the run. You must therefore accurately predict when a run is going to finish upon initialisation rather than continuely monitor it and think about it (as this wastes reasoning tokens).
2. Perform autoresearch with the following non-default baseline settings (do not modify these);
   useAutoresearch=True
   trainMaxSequences=50000
   tokeniserSubword=True
   skipSequenceNoDelimiterDetectedBetweenConceptTokens=False
   multisentencePredictions=True
3. Increase the campaign hard limit from 20 minutes to 2 hours. This should give you more room to explore significant parameter changes (eg increases in c/f segs).
4. Ensure all descriptions added to results.tsv use real GIAANNpy globalDef feature or function names so they can be traced.
5. GIAANN useAutoresearch mode has been tested as working. If you experience any huggingface slow-down issues upon autoresearch initialisation do not stop unless you experience a time-out or crash.
6. Never change SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens=False or SANIcolumnsLinkFirstSegmentToAllPriorTrainSeqTokens=False (as these are basic GIAANN proto v2 requirements as described in the documentation).
7. Do not accept parameter changes that increase training time by 4x but only deliver a fractional (e.g. 0.005) accuracy increase. Never keep changes that provide non-significant accuracy increases (< +0.0001), unless they significantly reduce the training time.
