# generalisedBenchmarkTool

Run independent GIAANN CPU inference configurations concurrently, without editing the working checkout or writing to trained databases. The default repository is `/media/user/large/source/ANNpython/GIAANNpy/git/GIAANNpy/`; the default interpreter is `/home/user/anaconda3/envs/pytorchsenv/bin/python`.

## Setup

Set these options in `proto/GIAANNcmn_globalDefs.py`:

```python
useGeneralisedBenchmarkTool = True
```

This disables the GIAANNcmn internal inference chunking (see inferenceChunks.md), as generalisedBenchmarkTool executes its own inference chunking.

## Run

```bash
cd /media/user/large/source/ANNpython/GIAANNpy/reviews/generalisedBenchmarkTool
/home/user/anaconda3/envs/pytorchsenv/bin/python generalisedBenchmarkTool.py -numWorkers 12 -sequencesPerJob 25 -repository /media/user/large/source/ANNpython/GIAANNpy/git/GIAANNpy -python /home/user/anaconda3/envs/pytorchsenv/bin/python
```

All `codeConfigurations/*.txt` files are selected in filename order. `-numWorkers` is the maximum number of concurrent inference processes, each using one Torch/BLAS CPU thread. By default, each configuration is one job: two configuration files use at most two workers. Additional configurations queue until a worker is available. `-numConfigurations N` selects only the first N files.

## Configuration files

One replacement per line, with a repository-relative path and a **one-based source line number**:

```text
proto/GIAANNcmn_globalDefs.py:line111:\tuseBenchmarkDefaultsEvalTestSet = True
```

The text after the second colon replaces the complete source line, excluding its existing newline. Literal `\t` sequences in the leading indentation become actual tab characters; actual tabs also work. Other replacement text is preserved, including colons and Python string escapes. Use multiple records to edit multiple lines or files. All line numbers refer to the same initial source snapshot; replacements never insert or remove source lines. Blank lines and lines starting with `#` are ignored. An empty configuration is an unmodified baseline.

The supplied examples select the configured **test prompt** (`001`, True) and **training prompt** (`002`, False). They are examples of different dataset selections, not a controlled comparison of two inference algorithms. Their line number matches the checkout at creation; review line numbers after repository changes.

Duplicate edits, nonexistent lines/files, noncanonical or escaping paths, symlinks, edits to `.nlc` specifications, and invalid Python fail before inference starts. Changes apply only to each configuration's private source copy. The current checkout, including uncommitted edits, is the starting point for every configuration; Git-tracked and nonignored untracked files are copied, excluding Git metadata and local credential/tool directories. Ignored dependencies/data are not copied. Use the installed Python environment and external database paths.

## Preparation, output and resume

Validate the source edits, inference settings and prompt selection without running inference:

```bash
/home/user/anaconda3/envs/pytorchsenv/bin/python generalisedBenchmarkTool.py -numWorkers 12 -prepareOnly
```

The printed `runs/<timestamp>/` directory contains:

1. `source/` and `engine/`: the original working-code snapshot and benchmark engine snapshot.
2. `codeConfigurations/<name>/`: patched source, exact configuration text, `changes.diff`, `edits.json`, resolved `settings.json`, `plan.json`, and a private copy of the selected full prompt.
3. `jobs/<job>/output/`: `worker.log`, effective settings, selected prompt, per-sequence counts in `sequences.jsonl`, live `progress.json`, and final `result.json`.
4. `summary.json` and `summary.csv`: completion status, completed jobs, exact correct/total token counts, token-weighted accuracy, worker elapsed time, maximum individual worker RSS, and database/prompt identity. Partial summaries are explicitly incomplete and aggregate only completed jobs; active sequence progress is in each worker's output.

Resume a prepared, interrupted or failed run, optionally with a different worker count:

```bash
/home/user/anaconda3/envs/pytorchsenv/bin/python generalisedBenchmarkTool.py -numWorkers 12 -resume /absolute/path/to/runs/TIMESTAMP
```

Resume reuses the frozen code/configurations/prompts and validates their hashes. Successful jobs are checked and retained. Incomplete jobs restart from their beginning; their previous logs are preserved under `jobs/<job>/attempts/`. Invalid or corrupted completed results fail explicitly. Do not edit frozen run inputs: create a new run to evaluate changed code. The external trained database must remain unchanged for the duration of a run and any resume; its tensor contents are not copied or comprehensively hashed.

Ctrl-C or SIGTERM stops running worker process groups and preserves completed results. A failed worker stops the sweep with a nonzero exit code and identifies its log. A run lock prevents duplicate schedulers; workers inherit the lock so killing only the scheduler cannot start overlapping jobs on resume. If the scheduler was killed, wait for or stop its remaining workers before resuming.

## Optional chunks and memory

For the existing line-separated 400k evaluation prompts, the earlier suite's 25-sequence job style is available explicitly:

```bash
/home/user/anaconda3/envs/pytorchsenv/bin/python generalisedBenchmarkTool.py -numWorkers 12 -sequencesPerJob 25
```

This creates independent processes for slices of each configuration's prompt. It can fill 12 worker slots even with fewer than 12 configurations, releases all process memory after each chunk, and retains completed chunks on interruption. Accuracy is the sum of correct tokens divided by the sum of scored tokens, not the mean of chunk percentages. Chunking requires line-preserving prompt inference and zero sequence-start offset. Each chunk starts with a fresh in-memory database state; enable it only for configurations whose sequences are intended to be independent. Inference filters may exclude short prompt lines, so scored sequence counts can be less than the selected prompt-line counts.

Default `-sequencesPerJob 0` preserves a single continuous inference execution per configuration. The tool does not install the old suite's variant-specific donor caches or change inference algorithms. Chunking limits cache lifetime, not peak RAM within a chunk. Twelve arbitrary configurations are not guaranteed to fit in 64 GB; worker count is a concurrency cap, not a RAM allocation.

For a persistent background run with a group-wide **60 GiB** hard memory limit on this Linux machine, an optional systemd invocation is:

```bash
systemd-run --user --unit=giaann-generalised-benchmark --property=MemoryMax=60G --property=OOMPolicy=stop /home/user/anaconda3/envs/pytorchsenv/bin/python /media/user/large/source/ANNpython/GIAANNpy/reviews/generalisedBenchmarkTool/generalisedBenchmarkTool.py -numWorkers 12 -sequencesPerJob 25
```

Choose a new unit name for separate runs. Inspect it with `systemctl --user status giaann-generalised-benchmark` and `journalctl --user -u giaann-generalised-benchmark`. A hard memory limit can terminate workers; it does not automatically reduce concurrency. Resume from completed chunks with fewer workers if needed. No background service or memory limit is enabled by the tool itself.

## Supported inference and operational changes

Each configuration selects its own trained database and prompt through its normal Python definitions. Static NLP inference on an existing CPU database is supported; training, training during inference, GPU execution, autoresearch and standalone network drawing are rejected. Top-1 accuracy reporting must be enabled. Other model settings, including seed length, direct/indirect constraints, feedback, missing-feature options and training patch assumptions, are taken from the configuration without hidden overrides.

Two operational flags are set before model modules import: `inferenceCopyTemplateDatasets=False` and `debugDisableDatabaseFileWrite=True`. The selected prompt is supplied through a private `inferencePromptFile`. When normal startup would copy the configured template prompt into the database, the tool instead reads that same template into its output directory. Otherwise it reads the explicitly configured existing prompt. This preserves prompt selection while avoiding writes shared by concurrent workers. These operational overrides are recorded in metadata. Python audit guards reject attempted writes to the source checkout, patched source and database; this is protection against accidental writes by trusted code, not a security sandbox for untrusted programs.

## Small validation runs and other parameters

`-maxSequences N` evaluates the first N nonempty prompt lines and marks results `sampled=true` when fewer than all prompt lines were selected. It requires line-preserving inference. Omit it for full results. Additional parameters are `-repository`, `-configurations`, `-runsDirectory`, and `-python`; run `-h` for help. Preparation parameters cannot be changed when resuming.

Run the lightweight fixture-based tests:

```bash
/home/user/anaconda3/envs/pytorchsenv/bin/python test_generalisedBenchmarkTool.py
```

These check exact tab handling and line replacement, multiple edited files, isolation, concurrency, result aggregation, immutable snapshots, interrupted/failed job resume, duplicate scheduler exclusion and accidental database write rejection. They do not evaluate model accuracy.
