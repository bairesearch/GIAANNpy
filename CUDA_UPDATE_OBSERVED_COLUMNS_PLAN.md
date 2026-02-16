# CUDA optimisation plan for `updateObservedColumnsEfficient()`

## Why `coalesce()` is likely the bottleneck

`updateObservedColumnsEfficient()` currently merges sparse updates by concatenating COO indices/values and then repeatedly calling `coalesce()`.

For large `nnz`, `coalesce()` is expensive because it effectively performs:

1. key generation for each COO coordinate,
2. global sort/group,
3. reduction of grouped values,
4. new sparse tensor materialisation.

Doing this inside per-column update paths multiplies total sorting work.

## Proposed x10 path: designated CUDA kernel with hash accumulation (no global coalesce in hot path)

### Core idea

Replace repeated COO concat+coalesce with a custom CUDA accumulation kernel that writes updates directly into a preallocated sparse storage model:

- **Static structure:** keep `(indices, values)` for each observed-column tensor in a canonical sorted order once per sequence step (or once per structural change).
- **Fast update path:** convert update coordinates to linear keys and use a CUDA hash table + atomic accumulation to merge duplicates in-kernel.
- **Materialise once:** rebuild final COO only once after all updates, not on every partial add.

### Data layout

Use a structure-of-arrays layout on GPU:

- `target_indices[D, nnz_target]` (int64)
- `target_values[nnz_target]` (float32/float16)
- `target_linear_key[nnz_target]` (int64, precomputed from multi-dim index)
- `slot_for_key` hash table (open addressing)

For each update batch:

- `update_indices[D, nnz_update]`
- `update_values[nnz_update]`
- `update_linear_key[nnz_update]`

### CUDA kernels

#### Kernel A: build/refresh target hash map

- Input: `target_linear_key`
- Output: `slot_for_key` mapping linear key -> value slot index
- Called only when target sparsity pattern changes materially.

#### Kernel B: accumulate updates

- Input: `update_linear_key`, `update_values`, `slot_for_key`
- Logic:
	- lookup key in hash map,
	- if found: `atomicAdd(target_values[slot], update_value)`,
	- if not found: append to overflow buffer (`new_keys`, `new_values`) via atomic counter.

#### Kernel C: merge overflow

- Sort-reduce only overflow keys (typically much smaller than full tensor).
- Append reduced overflow entries to target arrays.
- Rebuild hash map if overflow ratio exceeds threshold.

### Why this can achieve ~10x

The current path repeatedly sort-reduces the **entire** concatenated sparse payload. The proposed path sort-reduces only:

- rarely: when rebuilding hash map,
- usually: only the overflow (new coordinates),
- never: the entire target for every micro-update.

This changes complexity from repeated global `O((N+U) log(N+U))` behaviour toward near-linear `O(U)` atomic accumulation for stable sparsity patterns.

## Integration strategy in this codebase

1. Add a feature flag in `globalDefs.py`, e.g. `useCUDAObservedColumnUpdateKernel`.
2. Keep current Python sparse path as reference implementation.
3. Route only `updateObservedColumnsEfficient()` strength-update merge through the CUDA path when:
	- tensor device is CUDA,
	- dtype is supported,
	- feature flag enabled.
4. Validate numeric parity against current path (exact for fp32, tolerance for fp16).
5. Add microbenchmarks:
	- varying `nnz_target`, `nnz_update`, duplicate ratio, new-key ratio.

## Practical implementation options

### Option 1 (recommended): C++/CUDA extension via `torch.utils.cpp_extension`

- Best control over hash table and atomics.
- Can expose Python API:
	- `build_sparse_accumulator(indices, values, shape)`
	- `accumulate_sparse_updates(accumulator, update_indices, update_values)`
	- `export_coo(accumulator)`

### Option 2: Triton kernel prototype

- Faster iteration for prototype.
- Harder to implement robust dynamic hash/overflow logic.
- Good for proving speedup before C++ hardening.

## Guardrails

- Hard-fail when unsupported dtype/device is passed (do not silently fall back inside kernel path).
- Keep deterministic mode switch for reproducibility.
- Expose instrumentation counters:
	- hash hit rate,
	- overflow count,
	- rebuild count,
	- average update latency.

## Suggested first milestone

Implement GPU-only acceleration for the hottest operation currently equivalent to `addSparseUpdateNonNegative()`:

- Build accumulator once for `observedColumn.featureConnections` and `globalFeatureNeurons`.
- Replace concat+coalesce inside loop with kernel B accumulation.
- Export COO once per sequence at loop end.

If hit-rate is high and new-key ratio is low (common in iterative updates), this is the highest-probability path to a real 10x class speedup.
