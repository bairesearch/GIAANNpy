# autoresearch

This is an experiment to have GIAANN do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `GIAANNpy/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b GIAANNpy/<tag>` from current main [master].
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `../README.md` - repository summary.
   - `../proto/*.py` - the files you modify. Model architecture, parameters, etc.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed data budget of 5000 sequences**. 

You launch train phase, and then inference phase using these commands:

```
cd ../proto/
python GIAANNproto_main.py
cd ../autoresearch/
```

**What you CAN do:**
- Modify all `../proto/*.py` files - these are the only file you edit. Everything is fair game: model architecture, hyperparameters, etc.

**What you CANNOT do:**
- Modify any files not contained in `../proto'.
- Install new packages or add dependencies. You can only use what's already specified in `../README.md`.
- Modify the evaluation harness (useAutoresearch=True): Top-1 accuracy of datasetOscar is the eval ground truth metric (useBenchmarkDefaultsTestSet=True) after training for trainMaxSequences=5000 sequences.

**The goal is simple: get the highest top-1 accuracy. Everything is fair game: change the architecture, the hyperparameters, etc. The only constraint is that the code runs without crashing and finishes executing (trainMaxSequences=5000 sequences).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome - that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.0001 top-1 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A top-1 0.0001 accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the code as is.

## Output format

Once the code finishes it prints a summary like this:

```
---
averageTop1Accuracy: 0.8108882521489972
memory_gb: 2.4
```

Note that the code (when useAutoresearch=True) is configured to always train the first 5000 oscar dataset sequences, and test using the sequences within inference_prompt.txt.longTestOscar (during inferenceMode).

You can extract the key metric from the log file:

grep "^averageTop1Accuracy:" run.log

## Logging results

When an experiment is done, log the results to `results.tsv` (tab-separated, NOT comma-separated - commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	averageTop1Accuracy	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. averageTop1Accuracy achieved (e.g. 0.8108882) - use 0.000000 for crashes
3. memory_gb (database memory in GB), round to .1f (e.g. 2.4) - use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	averageTop1Accuracy	memory_gb	status	description
a1b2c3d	0.810888	2.4	keep	baseline
b2c3d4e	0.823200	2.8	keep	increase segment count to 18
c3d4e5f	0.805000	2.4	discard	decrease segment count to 4
d4e5f6g	0.000000	0.0	crash	modify segment sequentiality check
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `GIAANNpy/mar5` or `GIAANNpy/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on

2. Tune `../proto/*.py` with an experimental idea by directly hacking the code.

3. git commit

4. Run the experiment: 

```
cd ../proto/
python GIAANNproto_main.py > ../autoresearch/run.log 2>&1
cd ../autoresearch/
```

> (redirect everything - do NOT use tee or let output flood your context)

5. Read out the results: `grep "^averageTop1Accuracy:\|^memory_gb:" run.log`

6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.

7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)

8. If averageTop1Accuracy improved (higher), you "advance" the branch, keeping the git commit

9. If averageTop1Accuracy is equal or worse, you git reset back to where you started


The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder; read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
