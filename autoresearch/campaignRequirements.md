# Campaign Requirements

The Campaign Requirements provide campaign specific settings for configuring the autoresearch baseline, and monitoring, evaluating and recording experiments.

These are the user requirements specific to this autoresearch campaign:

1. When monitoring autoresearch train/eval runs, just wait until they are finished before you think about the run. You must therefore accurately predict when a run is going to finish upon initialisation rather than continuely monitor it and think about it (as this wastes reasoning tokens).
2. Perform autoresearch with the following non-default baseline settings (do not modify these);
   useAutoresearch=True
   trainMaxSequences=50000
   multisentencePredictions=True
3. Increase the campaign hard limit from 20 minutes to 2 hours. This should give you more room to explore significant algorithm or implementation changes.
4. Ensure all descriptions added to results.tsv use real GIAANNpy globalDef feature or function names so they can be traced.
5. GIAANN useAutoresearch mode has been tested as working. If you experience any huggingface slow-down issues upon autoresearch initialisation do not stop unless you experience a time-out or crash.
6. This autoresearch is focused on fundamental GIAANN algorithm or implementation (i.e. code) issues in the train or eval phase that limit test-set accuracy eval. It is not focused on minor incremental improvements via parameter tweaking.
7. Never keep changes that provide non-significant accuracy increases (< +0.005), unless they significantly reduce the training time.
