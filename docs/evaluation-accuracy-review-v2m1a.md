# Evaluation accuracy review (proto v2m1a)

This review follows the v2m1a corrections for prompt boundaries, repeated-feature
transitions, and external prime-concept transitions. It concentrates on whether
the reported train- or test-set result measures the intended prediction task.

## High-impact findings

### 1. Benchmark accuracy is teacher-forced after every token

For benchmark and autoresearch runs,
`inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures` is set to
`False`. `selectNextColumnFeaturePredictionPhase` consequently replaces the
predicted next state with the target column and target feature before the next
iteration. Each top-1 decision is therefore conditioned on the correct history,
not on the model's generated history.

This is a valid next-token (teacher-forced) metric, but it is not autoregressive
sequence-generation accuracy. Reporting it simply as `averageTop1Accuracy`
makes comparisons with free-running inference ambiguous and can substantially
overstate multi-token generation performance after the first error.

**Required correction:** report the conditioning mode alongside the metric and
publish teacher-forced and prediction-fed results as separate metrics. Do not
silently change the existing benchmark series, because that would make historical
results incomparable.

### 2. Test inference expands the model vocabulary from test targets

`inferenceAddNewFeatures` defaults to `True`. Before scoring a sequence,
`expandSequenceForInference` passes the entire sequence (including every target)
through concept and feature discovery with new-feature creation enabled. This
mutates the network schema using test-set information before prediction.

Although no connections are trained by that pass, the evaluation is no longer
against a frozen model/vocabulary. It can alter candidate sets, column indices,
memory use, and probability normalisation. It also makes the result dependent on
test sequence order. This especially compromises bits-per-byte and any comparison
with a fixed-vocabulary model.

**Required correction:** add a frozen-vocabulary evaluation mode and make it the
benchmark/autoresearch default. Out-of-vocabulary targets must receive an explicit
miss (and zero probability for probabilistic scoring), rather than being inserted
from the answer text. Keep dynamic expansion as a separately named open-vocabulary
experiment.

### 3. Unprocessable evaluation sequences can disappear from the denominator

Several paths finish without recording misses:

* a post-tokenisation sequence no longer than the seed returns before inference;
* a sequence with no detected concepts never calls prediction;
* a sequence rejected by the no-delimiter rule skips prediction; and
* length/token-limit rejection only updates the progress bar.

The accuracy counters are updated inside prediction, so these paths exclude hard
examples rather than scoring them as failures. This creates selection bias and
allows preprocessing/configuration changes to improve accuracy solely by rejecting
more sequences.

**Required correction:** track `presented`, `accepted`, `rejected`, and `scored`
sequence/token counts. A benchmark must either fail hard when an item cannot be
scored or include all of its prediction tokens as misses. The report must expose
the rejection reason counts. Bits-per-byte evaluation should fail hard if any
target probability cannot be produced.

### 4. The secondary `inferenceTokens` accuracy includes copied seed targets

Seed-phase selection takes the observed target feature, and the same general
accuracy counter records those seed decisions. The output then prints both
`predictionTokens` and `inferenceTokens`; the latter includes seed tokens that
were supplied rather than predicted. It is therefore predictably higher and is
not an evaluation accuracy.

The autoresearch summary correctly publishes prediction-token accuracy, but the
interactive output leaves the inflated aggregate available for accidental use.

**Required correction:** rename the aggregate as a diagnostic seed-plus-prediction
match rate, or remove it from evaluation output. Only prediction-token accuracy
should be labelled top-1 accuracy.

### 5. Default training can reuse an existing database

Outside autoresearch, `trainLoadExistingDatabase` defaults to `True`. A nominal
training run can therefore accumulate prior runs, including a database created
with different train limits or configuration. Subsequent train-set and test-set
accuracy cannot be attributed to the stated run alone, and a previously exposed
evaluation item can persist in the database.

**Required correction:** benchmark runs must fail hard if the output database is
not empty, unless an explicit resume mode is selected and its checkpoint identity,
prior sequence count, dataset identity, and configuration fingerprint are reported.

## Medium-impact findings

### 6. Probabilistic scoring excluded the supplied prefix (resolved)

The BPB path now records target probabilities only for prediction tokens and uses
the UTF-8 byte length of the post-seed prediction span. Its numerator and
denominator therefore measure the same continuation region as top-1 accuracy.

### 7. The test corpus is a checked-in derived prompt, not a reproducible split

Normal inference reads a prompt file rather than loading the dataset split used
to produce it. The repository contains several manually selected `longTrain` and
`longTest` variants, while configuration chooses among them by flags. There is no
machine-checked manifest tying a prompt to the source dataset revision, split
algorithm, preprocessing configuration, or exclusion boundary.

**Required correction:** generate a manifest containing dataset name/config/revision,
source indices, content hash, generation options, and the maximum training boundary.
Inference must verify the manifest before scoring. Prefer loading a native test
split directly where one exists.

## Validation priorities

1. Add counter-level tests proving that rejected/short/no-concept sequences cannot
   improve accuracy by disappearing.
2. Add a two-token adversarial fixture where the first prediction is wrong and
   teacher-forced versus prediction-fed histories yield different second tokens.
3. Assert that frozen evaluation leaves column/feature counts and serialized data
   byte-for-byte unchanged.
4. Assert that OOV targets are misses with zero probability in frozen mode.
5. Record and verify a dataset/prompt manifest before every benchmark run.

Until findings 1-5 are resolved, results should be described as exploratory
teacher-forced accuracy on accepted sequences with dynamic test-vocabulary
expansion, rather than held-out autoregressive test accuracy.
