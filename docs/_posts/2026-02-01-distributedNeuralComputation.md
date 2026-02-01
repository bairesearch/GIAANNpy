---
layout: post
title: Distributed Neural Computation on a GPU Cluster
date: 2026-02-01
image: assets/distributed-neural-computation-twitter.png
image_alt: Distributed Neural Computation on a GPU Cluster
---

*The compatibility of GIAANN Biological Artificial Intelligence and frontier‑scale GPU superclusters*

## Summary
This post describes a distributed GIAANN training and inference implementation on a large peer-to-peer GPU cluster. The design assumes 1m+ concept columns at ~1GB->1TB each, with most nodes dedicated to storing concept columns in RAM (not used for primary computation), and a smaller batch B of nodes dedicated to primary compute for training and inference.

## Storage Layout Assumptions
- Most nodes store a subset (1+) of concept columns (in RAM) and primarily serve data to/from computation nodes.
- Concept columns are large (~1GB->1TB each), so the cluster is storage-heavy relative to compute.
- GIAANNpy currently stores entire concept columns in `database/observedColumns` files (typically Ext4 SSD filesystem). In a distributed implementation, concept column features (including their outgoing connections) may be split into separate RAM regions rather than a single large sparse PyTorch tensor to speed up inference.

## Training Parallelism (Multiple Sequences)
### Description
- Each training round (sequence) is assigned to a dedicated computation node.
- Sequences are either randomly selected or preselected to minimize overlap in concept columns, reducing interconnect contention.
- The computation node builds a dense PyTorch subnetwork for the sequence connectivity (see `GIAANNproto_databaseNetworkTrainExcitation.py`).
- The computation node sends per-sequence concept column data to the storage nodes responsible for those concept columns.
- Transferred data is typically limited to sequence features when `trainSequenceObservedColumnsUseSequenceFeaturesOnly=True` (densified view of sparse columns).
- Storage nodes independently integrate updates into their sparse concept column tensors (see `updateObservedColumns()` in `GIAANNproto_sequenceObservedColumnsExcitation.py`).

### Diagram (Training)
![Training parallelism diagram]({{ "/assets/training-parallel.svg" | relative_url }})

## Inference Parallelism (Multiple Sequences)
### Description
- Each sequence corresponds to a separate user prompt/prediction (inference for a single sequence is not parallelized; it executes per token).
- Each computation node in batch B handles one prompt+prediction sequence.
- Each computation node keeps its own `globalFeatureNeurons` tensor in GPU memory (very large, 100GB+ in production).
- For each token (inference round):
	- The computation node requests the full concept column data for the token (concept column feature), including outgoing connections.
	- The computation node computes target activations.
	- The computation node updates its local `globalFeatureNeurons` tensor.
	- The process repeats for the next token in the seed/prediction sequence.

### Diagram (Inference)
![Inference parallelism diagram]({{ "/assets/inference-parallel.svg" | relative_url }})

## Notes
- The training batch size B is bounded by interconnect bandwidth and storage node parallelism.
- Sequence selection that minimizes column overlap reduces contention and improves throughput.
- Splitting concept column features into multiple RAM regions could reduce sparse tensor overhead for inference.
