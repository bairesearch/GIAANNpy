# autoresearch

This is an experiment to have GIAANN do its own research.

All paths in this file are relative to the repository root unless stated otherwise. Follow `AGENTS.md` throughout the campaign. Treat `GIAANNproto*.nlc` as read-only requirements.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a date-based tag such as `jul13`. The branch `GIAANNpy/<tag>` must not already exist.
2. **Protect existing work**: inspect the current branch, commit, and `git status --short`. Never discard pre-existing changes. Ask the user to resolve unrelated changes; include intended `proto` configuration changes in the campaign baseline.
3. **Create the branch**: create `GIAANNpy/<tag>` from the user-approved base revision.
4. **Read the current context**: read `AGENTS.md`, `README.md`, every `GIAANNproto*.nlc` requirement file, and all relevant `proto/*.py` files. The NLP autoresearch path includes common and `GIAANNnlp_*` code; do not modify unrelated `GIAANNor_*` code while `useModalityNLP=True`.
5. **Validate the campaign configuration**: require `useModalityNLP=True`, `useAutoresearch=True`, `executionMode="trainAndInference"`, `datasetType="oscar"`, `trainMaxSequences=5000`, `trainLoadExistingDatabase=False`, held-out evaluation, and top-1 accuracy reporting. Confirm that the configured database directory is writable or can be created under a writable parent, and that its template contains the derived `inferencePromptFileName`. Hard-fail setup if any invariant is not satisfied.
6. **Freeze comparison semantics**: record the source revision, hardware/software environment, dataset slice, derived evaluation prompt, tokenizer settings, sentence/sequence settings, and accuracy settings. Do not compare runs after changing any of these; start a separate campaign and baseline instead.
7. **Commit the baseline state**: ensure the exact intended `proto` baseline is committed and the working tree has no unresolved changes.
8. **Initialize the ignored results file**: create `autoresearch/results.tsv` with the nine-column header shown below. The baseline is its first data row.
9. **Confirm and go**: report the branch, baseline commit, fixed settings, database path, prompt filename, interpreter, and run timeout to the user.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is one combined train-and-inference execution. With the fixed autoresearch configuration, it rebuilds the database, trains on **5000 OSCAR sequences**, evaluates the held-out prompt selected by `inferencePromptFileName`, and prints top-1 accuracy plus resource measurements.

Run from the repository root with the configured project interpreter. Replace `python` with the explicit environment interpreter path when required by the local setup:

```bash
cd proto
python GIAANNcmn_main.py
```

**What you CAN do:**

- Modify relevant `proto/*.py` model and optimization code.
- Change architecture and hyperparameters that do not alter the frozen comparison semantics.
- Add a focused helper function or `proto` file when the repository rules call for it.

**What you CANNOT do:**

- Modify files outside `proto` during the experiment loop.
- Modify `GIAANNproto*.nlc` under any circumstances.
- Modify the evaluation inputs, output-summary implementation, fixed data budget, dataset start/slice, evaluation-prompt selection, top-1 metric settings, database reset behavior, or any other frozen comparison setting.
- Change tokenization or sentence/sequence composition within an existing campaign, because that changes the population over which top-1 accuracy is measured.
- Install packages or add dependencies. Use only the environment installed during setup.
- Touch unrelated modality code or unrelated user changes.

**The primary goal is the highest held-out `averageTop1Accuracy` under the frozen 5000-sequence comparison.** The code must finish without error and emit a complete summary. Resource use and simplicity are secondary decision criteria.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome - that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.0001 top-1 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A top-1 0.0001 accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the code as is.

## Output format

The current accuracy path prints one final summary with these fields:

```
---
averageTop1Accuracy: 0.8108882521489972
databaseMemoryGb: 2.4
training_seconds: 298.4125513479953
total_seconds: 307.1849021180244
peak_vram_mb: 11872.53125
peak_ram_mb: 18463.75
```

Do not hard-code an evaluation filename. `GIAANNnlp_globalDefs.py` derives it from the frozen sentence, benchmark, and test-set settings. Read the `inferencePromptFileName` printed in `run.log` and require it to match the campaign baseline.

Extract the summary from the repository root with:

```bash
grep -E "^(averageTop1Accuracy|databaseMemoryGb|training_seconds|total_seconds|peak_vram_mb|peak_ram_mb):" autoresearch/run.log
```

## Logging results

When an experiment is done, append it to the ignored `autoresearch/results.tsv` file. Use tabs, not commas or spaces, as separators.

The TSV has nine columns matching the current emitted summary:

```
commit	averageTop1Accuracy	databaseMemoryGb	training_seconds	total_seconds	peak_vram_mb	peak_ram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. `averageTop1Accuracy` achieved; use `0.000000` for crashes
3. `databaseMemoryGb`; use `0.0` for crashes
4. `training_seconds`; use `0.0` for crashes
5. `total_seconds`; use `0.0` for crashes
6. `peak_vram_mb`; use `0.0` for crashes
7. `peak_ram_mb`; use `0.0` for crashes
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	averageTop1Accuracy	databaseMemoryGb	training_seconds	total_seconds	peak_vram_mb	peak_ram_mb	status	description
a1b2c3d	0.810888	2.4	298.413	307.185	11872.53	18463.75	keep	baseline
b2c3d4e	0.823200	2.8	310.200	319.450	12004.00	19002.25	keep	increase segment count to 18
c3d4e5f	0.805000	2.4	294.100	303.800	11872.53	18460.00	discard	decrease segment count to 4
d4e5f6g	0.000000	0.0	0.0	0.0	0.0	0.0	crash	modify segment sequentiality check
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `GIAANNpy/mar5` or `GIAANNpy/mar5-gpu0`).

LOOP FOREVER:

1. Verify the dedicated branch, retained start commit, and clean working tree. Hard-fail if unrelated or unexplained changes are present.

2. Form one testable idea and modify only the relevant `proto` implementation. Preserve every frozen campaign setting.

3. Review the diff, run focused static or unit checks where available, and commit the experiment. The committed tree must exactly match the code being measured.

4. Run the combined experiment from the repository root, recording the process exit status immediately:

```bash
cd proto
python GIAANNcmn_main.py > ../autoresearch/run.log 2>&1
```

Redirect everything; do not use `tee` or stream the full output into the agent context.

5. Require a zero exit status and exactly one complete six-field summary. Extract it with the `grep -E` command above. Also require the printed `useAutoresearch`, `executionMode`, `datasetType`, `trainMaxSequences`, and `inferencePromptFileName` values to match the baseline.

6. Treat a nonzero exit, timeout, missing/duplicate field, non-finite value, or changed invariant as a failed run. Inspect `tail -n 50 autoresearch/run.log`. Fix only a clear implementation defect; commit the fix before rerunning so the measured commit remains exact. Otherwise record the failure and discard the experiment.

7. Append all measurements and the measured commit to `autoresearch/results.tsv`. The file is ignored and must never be committed.

8. Keep the commit when accuracy improves enough to justify its complexity. Keep an equal-accuracy result only when it materially simplifies the code or improves a secondary resource measurement without a meaningful regression elsewhere.

9. Otherwise discard it by resetting the dedicated experiment branch to the recorded start commit. Use `git reset --hard <start-commit>` only after verifying that all experiment code is committed and no pre-existing or unrelated work can be lost. Never reset beyond the current experiment's start commit.


The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: The fixed input budget does not guarantee a fixed runtime. Record the baseline runtime, but retain the campaign's 20-minute hard limit unless the human explicitly sets another limit during setup. Kill an over-limit run, record it as a failure, and discard it.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder; read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

The user may leave the campaign running unattended. Throughput depends on the measured baseline and each retained configuration; do not assume a fixed number of experiments per hour.
