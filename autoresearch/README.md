# autoresearch

This directory contains the autonomous experiment protocol for GIAANN. See [`../README.md`](../README.md) for the repository overview and `../GIAANNproto*.nlc` for the current requirements.

The human researcher gives an AI agent a real GIAANN NLP training and evaluation setup. The agent changes the in-scope prototype code, runs the fixed-budget experiment, records the emitted measurements, and retains only justified improvements. `program.md` defines that protocol and is edited by the human researcher; the agent experiments in `../proto`.

## How it works

With `useAutoresearch=True`, the current common configuration selects `executionMode="trainAndInference"`. One execution of `GIAANNcmn_main.py` clears the autoresearch database, trains it, evaluates it, and emits the summary consumed by the experiment log.

The primary files are:

- `../proto/GIAANNcmn_globalDefs.py`: common execution, database, inference, and autoresearch settings.
- `../proto/GIAANNnlp_globalDefs.py`: NLP dataset, sequence, tokenizer, and evaluation-prompt settings.
- `../proto/GIAANNcmn_main.py`: the combined training/inference entry point.
- `../proto/*.py`: the implementation the agent may optimize within the limits in `program.md`.
- `../database/inference_prompt.txt.*`: versioned evaluation inputs. The active filename is derived from the current NLP configuration and printed as `inferencePromptFileName`.
- `program.md`: the experiment instructions maintained by the human researcher.

The built-in autoresearch path currently uses `datasetType="oscar"`, `trainMaxSequences=5000`, a held-out evaluation prompt, and top-1 next-token feature accuracy. The input budget is fixed, but elapsed time is platform- and configuration-dependent. Tokenizer or sequence-composition changes alter the evaluated token population and therefore require a separate campaign baseline.

## Documentation and requirements

- Repository overview and installation: [`../README.md`](../README.md)
- Agent coding rules: [`../AGENTS.md`](../AGENTS.md)
- Current requirements: `../GIAANNproto*.nlc`
- Background paper: `../paper/GIAANN-paper-WIP/*.tex`

The requirements files are read-only during autoresearch.

## Running the agent

Start Codex in the repository root with the required workspace and execution permissions, then prompt:

```
Hi have a look at autoresearch/program.md and let's kick off a new experiment! let's do the setup first.
```

The setup phase protects existing work, creates a dedicated branch, validates the fixed comparison settings, and establishes a baseline before autonomous experimentation begins.

## Project structure

```
../AGENTS.md                         repository coding rules
../GIAANNproto*.nlc                  read-only requirements
../proto/GIAANNcmn_globalDefs.py     common/autoresearch configuration
../proto/GIAANNnlp_globalDefs.py     NLP configuration
../proto/GIAANNor_globalDefs.py      OR configuration (not used by the NLP campaign)
../proto/GIAANNcmn_main.py           train-and-inference entry point
../proto/*.py                        prototype implementation
.gitignore                           local experiment artifacts
program.md                           autonomous experiment protocol
```

## Design choices

- **Limited mutation scope.** The agent changes only relevant files under `../proto`; the evaluation harness, requirements, prompts, and autoresearch protocol remain fixed during a campaign.
- **Fixed comparison context.** The training budget, dataset slice, evaluation prompt, tokenizer/sequence interpretation, metric, base revision, software environment, and hardware must remain comparable. A change to one of these starts a new campaign and baseline.
- **Complete measurements.** Each run records accuracy, database size, training and total time, and peak GPU/CPU memory from the current autoresearch summary.
- **Isolated database.** `trainLoadExistingDatabase=False` clears and rebuilds the configured autoresearch database for every run.
- **Single-process execution.** Autoresearch does not use distributed training. CPU/GPU placement follows the current global definitions; an enabled GPU device requires CUDA.

## Platform support

The active global definitions determine hardware use. The current code supports CPU placement and optional CUDA placement; use the same hardware and software environment throughout a campaign so measurements remain comparable.

## License

MIT

- Copyright (c) 2026 karpathy (https://github.com/karpathy/autoresearch)
- Copyright (c) 2026 BAI Research Pty Ltd (bairesearch.com.au)
