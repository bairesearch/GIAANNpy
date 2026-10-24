# generalisedBenchmarkTool

Compare repository configurations using the inference implementation in `proto`. The tool prepares frozen source copies, applies configuration edits, selects the inference prompt and records results. **All chunk creation, worker scheduling, thread limits and accuracy aggregation are performed by `proto/GIAANNcmn_inferenceChunks.py` and `proto/GIAANNcmn_inferenceChunkWorker.py`.**

Configurations execute **one at a time** in filename order. Each configuration can use the full native worker pool; `-numWorkers 12` means at most 12 native inference workers for the active configuration. There is no separate benchmark worker pool or allocation of workers between configurations.

## Run

From this directory, using the Python environment containing the GIAANN dependencies:

```bash
python generalisedBenchmarkTool.py -numWorkers 12 -sequencesPerJob 25
```

The repository defaults to this directory's parent. The Python interpreter defaults to the interpreter running the command. Neither `-repository` nor `-python` is needed when those defaults are correct. `-python`, when supplied, is an executable file path.

Outputs default to `runs/` in the benchmark tool's root directory (`generalisedBenchmarkTool/runs/`). The directory is created automatically and ignored by Git. Both this default directory and the selected output directory are excluded from repository snapshots, so previous results are never copied into new runs. To choose a different location:

```bash
python generalisedBenchmarkTool.py -numWorkers 12 -sequencesPerJob 25 -runsDirectory /absolute/path/to/benchmark-runs
```

Both worker and chunk arguments are optional. Without them, each configuration uses its native `inferenceWorkersCount` and `inferenceChunksSequenceCount` settings. `-numWorkers` overrides `inferenceWorkersCount`; `-sequencesPerJob` overrides `inferenceChunksSequenceCount`. These values are written into the private source copy before importing the model, so every native child process inherits them. Zero is invalid for either override. The old `-sequencesPerJob 0` execution mode is no longer supported.

The output directory may be inside the repository, but must not contain the repository or configuration directory, overlap the configuration directory, or be inside `proto/` or excluded metadata directories. Existing runs retain their saved paths; changing the default does not relocate a running benchmark.

Native `optimisationInferenceChunks=True` is required. For parallel inference, also enable `optimisationInferenceWorkers=True`; a configuration with workers disabled can run serial native chunks when `-numWorkers` is omitted. `useGeneralisedBenchmarkTool` no longer disables the repository's native optimisation flags.

## Configuration files

Provide at least one `.txt` file under `codeConfigurations/`, or select another directory with `-configurations`. An empty file is an unmodified baseline. All files are selected in filename order unless `-numConfigurations N` limits the selection.

Each non-comment line replaces one existing source line:

```text
proto/GIAANNcmn_globalDefs.py:line106:\tuseBenchmarkDefaultsEvalTestSet = True
```

The number is one-based and refers to the original source snapshot. The text after the second colon replaces the entire line. Literal `\t` prefixes become tabs; actual tabs also work. Blank lines and lines starting with `#` are ignored. Multiple files and lines can be edited in one configuration. Review the line numbers in `codeConfigurationsExamples` against the current repository before using them.

Edits apply to private copies; repository source files and the trained database remain unchanged. The snapshot includes tracked and nonignored untracked files, excluding benchmark output directories, Git metadata, Python caches and local credential/tool directories. Invalid paths, duplicate edits, symlinks, edits to `.nlc` specifications and Python syntax errors fail before inference.

Operational overrides are recorded separately in `execution.diff` and `plan.json`. They set the frozen prompt path, disable template copying and database writes, select the native output directory, and apply any requested worker/chunk counts. Model and dataset choices still come from the edited repository configuration. Unsupported native inference settings fail through the native validator.

## Results and progress

The printed run directory contains:

1. `source/` and `engine/`: the starting repository and benchmark driver snapshots.
2. `codeConfigurations/<name>/`: the configuration text, edited source, `changes.diff`, `execution.diff`, selected prompt, plan, original settings and effective settings.
3. `output/<name>/worker.log`: output from that configuration's native inference controller, including native chunk progress and final accuracy.
4. `output/<name>/native/giaann-inference-*/`: native manifests, source snapshots, per-chunk logs, per-sequence counts and the native `summary.json`.
5. Top-level `summary.json` and `summary.csv`: configuration status, prediction-only and seed-inclusive accuracy, token counts, timings and links to native results.

The benchmark reads the native aggregate; it does not average chunk percentages or duplicate the inference accounting hooks. Native worker elapsed time is summed process time, not wall time. Peak worker RSS is the largest individual worker, not total concurrent RAM. Worker count limits concurrency, not total memory.

The top-level console prints configuration starts and finishes. Detailed chunk progress is in the active configuration's `worker.log` and native summary. Incomplete configurations have no completed accuracy in the top-level summary.

## Preparation, sampling and resume

Validate and freeze inputs without inference:

```bash
python generalisedBenchmarkTool.py -numWorkers 12 -prepareOnly
```

Use `-maxSequences N` for an explicit sample of the first N nonempty prompt lines; results are marked sampled when this selects fewer than all prompt lines. Native inference applies its usual short-sequence and length filtering. Omit this argument for a full evaluation.

Resume with only the saved run path:

```bash
python generalisedBenchmarkTool.py -resume /absolute/path/to/run
```

Completed configurations are validated and retained. **An interrupted or failed configuration restarts from its beginning**, because the native chunk scheduler currently has no chunk-resume facility. Its previous output moves into `attempts/`. Worker and chunk settings are frozen; create a new run to change them.

Ctrl-C or SIGTERM is forwarded to the active configuration controller; the native scheduler stops its workers. A lock prevents duplicate configuration runners. Resume refuses a previous attempt with recorded native workers still alive. Keep the trained database unchanged during a run and any resume. This tool does not set a memory limit.

The new manifest schema does not accept old benchmark runs. Retain their frozen `engine/` files and use that original engine to resume an old run. Existing result folders are not rewritten by this upgrade.

## Validation tests

```bash
python test_generalisedBenchmarkTool.py
```

These tests use the actual repository chunk scheduler and worker with a small deterministic model fixture. They check configuration edits, native delegation, uneven chunks and short-line filtering, overrides, sampling, isolation, read-only database protection, interruption cleanup, locking, resume and corrupted-result rejection. They do not require a trained database.
