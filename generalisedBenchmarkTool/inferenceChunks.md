# Chunked CPU inference

Set these options in `proto/GIAANNcmn_globalDefs.py`:

```python
optimisationInferenceChunks = True
optimisationInferenceWorkers = True
```

The dependent settings default to `inferenceChunksSequenceCount = 25`, `inferenceWorkersCount = 12`, and `inferenceChunksWorkerThreads = 1`. Change the chunk size or worker count in their existing conditional blocks. Set both flags to `False` to use normal execution. The current defaults enable them for inference mode; `useGeneralisedBenchmarkTool` does not disable them.

Run the existing inference entry point from `proto`, using the Python environment containing the project dependencies:

```bash
python GIAANNcmn_main.py
```

1. `optimisationInferenceChunks` splits the configured inference prompt into independent sequences and starts a fresh process for each chunk. With workers disabled, chunks execute one at a time.
2. `optimisationInferenceWorkers` runs up to `inferenceWorkersCount` chunks concurrently, assigning another chunk whenever a process finishes. It requires chunking. All workers serve this one repository configuration. `generalisedBenchmarkTool` compares configurations sequentially and delegates each configuration to this same native worker pool.
3. Each worker uses one Torch intra-op, Torch inter-op and BLAS compute thread, loads the existing trained database, and executes the same inference implementation. Blank lines and sequences too short for prediction are filtered as in normal execution. Other length limits remain enforced by the existing inference code.
4. Results sum correct-token and scored-token counts before calculating accuracy. Both prediction-only and seed-inclusive `averageTop1Accuracy` outputs are preserved. Sequence and token prediction logic is unchanged.

The run prints its output directory. By default this is a new `giaann-inference-*` directory under the system temporary directory. To retain runs elsewhere, set `inferenceChunksOutputDirectory` to an existing directory outside the repository and database. Each run retains a source snapshot, manifest, private prompts, worker logs, per-sequence counts, and `summary.json`. Temporary directories may be removed by system cleanup.

The source snapshot fixes the code used by workers after initialization. Keep the trained database unchanged throughout evaluation. Workers disable template copying and database saves and install a write guard for the database and source directories. The parent reads the template prompt directly when normal execution would copy it into the database.

This path supports static CPU NLP inference with an existing database, independent prompt lines, one epoch, zero sequence offset, and top-1 token accuracy. Training, GPU execution, drawing, grounded evaluation, bits-per-byte reporting, autoresearch and startup generation of saved neuron tensors are rejected. Timing and memory diagnostics inside worker logs describe individual processes; `summary.json` reports total wall time and the largest individual worker RSS, not total concurrent RAM.

The worker count limits concurrency, not total memory. Each fresh process releases its caches when its chunk ends; peak memory still depends on the database, sequences and concurrent workers. A failed worker fails the run and stops remaining workers. Interrupting the parent also stops workers. Completed chunk files remain available for inspection; automatic resume is not implemented.
