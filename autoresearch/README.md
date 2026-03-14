# autoresearch

This is a readme for the GIAANN autoresearch project. See ../README.md for the repository readme file.

The idea: give an AI agent a small but real GIAANN model training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code is GIAANN/proto. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc.

## How it works

The primary GIAANN files used by autoresearch:

- ** proto/*.py the files the agent edits. Contains the full GIAANN proto model. Everything is fair game: architecture, hyperparameters, etc. **These files are edited and iterated on by the agent**.
- **`program.md`** baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for exactly 5000 sequences (**approx 5-minute time budget**). The eval metric is top-1 accuracy (next word prediction).

## Documentation

**Repository Summary and Requirements:**
See ../README.md

**Background:**
See paper/GIAANN-paper-WIP/*.tex

## Running the agent

Simply spin up your Codex in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
../proto/GIAANNproto_globalDefs.py      all configurable parameters
../proto/GIAANNproto_main.py       entry point (train/inference)
../proto/*.py 	all other optimisable code
program.md      agent instructions
```

## Design choices

- **Limited files to modify.** The agent only touches the ../proto files. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** With useAutoresearch=True, training always runs for exactly trainMaxSequences=5000 sequences (approx 5 minutes) and datasetOscar=True, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform for that input data budget.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU.

## License

MIT

- Copyright (c) 2026 karpathy (https://github.com/karpathy/autoresearch)
- Copyright (c) 2026 baxterai


