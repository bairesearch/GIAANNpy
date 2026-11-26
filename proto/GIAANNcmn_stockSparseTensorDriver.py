"""GIAANNcmn_stockSparseTensorDriver.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common stock Sparse Tensor Driver

Stock PyTorch CUDA sparse arithmetic with packed GIAANN training storage.

"""


from GIAANNcmn_globalDefs import *

if(useGPUstockSparseTensorDriver and executionMode == "train"):
	from pathlib import Path
	import numpy as np
	from torch.utils.cpp_extension import load
	import GIAANNcmn_databasePacked as packedStorage
	_extension = None


	def coalesceTensor(tensor):
		result = tensor.to(stockSparseCUDADevice).coalesce().to(deviceDatabase)
		return result


	def mergeConnections(state):
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections
		keys, updates = getExtension().merge(state.columns, state.deferredSourcePreparation, state.connections, state.sourceShape, trainSequenceBatchBucketDimension, False, PackedDatabaseResidentFeatureConnections, packedStorage.getEntryDtype(), stockSparseTileEntries, trainSequenceBatchInvalidInput, reduceTile)
		updates.prepareOutstanding(state.deferredSourcePreparation, state.sourceShape, deviceDatabase, trainSequenceBatchInvalidInput)
		state.deferredSourcePreparation.clear()
		return keys, updates


	def mergeNeurons(state, shape):
		keys, updates = getExtension().merge(state.columns, {}, state.neurons, shape, trainSequenceBatchBucketDimension, True, None, np.dtype(np.int32), stockSparseTileEntries, trainSequenceBatchInvalidInput, reduceTile)
		return keys, updates.outputs


	def reduceTile(hostKeys, hostValues, extent, sources, bitPacked, segmentShift, featureBits):
		keys = hostKeys.to(stockSparseCUDADevice)
		values = hostValues.to(stockSparseCUDADevice)
		coalesced = pt.sparse_coo_tensor(keys.unsqueeze(databaseResidentPackedZero), values, (extent * sources,), device=keys.device).coalesce()
		keys = coalesced.indices()[databaseResidentPackedZero]
		values = coalesced.values()
		if(bitPacked):
			# A packed key orders source, segment, target concept and target feature.
			# Property and branch are singleton axes for this encoding.
			targetBits = extent.bit_length() - databaseResidentPackedOne
			local = keys.bitwise_and(extent - databaseResidentPackedOne)
			records = pt.empty((values.numel(), databaseResidentPackedSourceRank + databaseResidentPackedOne), dtype=databaseResidentCoordinateDtype, device=keys.device)
			records[:, :trainFusedSegmentDimension].zero_()
			records[:, trainFusedSegmentDimension] = local >> segmentShift
			records[:, databaseResidentPackedConceptDimension] = (local >> featureBits).bitwise_and((databaseResidentPackedOne << (segmentShift - featureBits)) - databaseResidentPackedOne)
			records[:, databaseResidentPackedFeatureDimension] = local.bitwise_and((databaseResidentPackedOne << featureBits) - databaseResidentPackedOne)
			records[:, databaseResidentPackedSourceRank] = values.view(databaseResidentCoordinateDtype)
			nextSourceKeys = pt.arange(databaseResidentPackedOne, sources + databaseResidentPackedOne, dtype=keys.dtype, device=keys.device) << targetBits
			ends = pt.searchsorted(keys, nextSourceKeys, out_int32=True)
			result = records.cpu(), ends.cpu()
		else:
			result = keys.cpu(), values.cpu()
		return result


	def isPending(updates):
		return _extension is not None and isinstance(updates, _extension.Pending)


	def getExtension():
		global _extension
		if(_extension is None):
			source = Path(__file__).with_name(stockSparseHostSource)
			_extension = load(name=stockSparseHostName, sources=[str(source)], extra_cflags=trainSequenceBatchCompilerFlags, extra_ldflags=trainSequenceBatchLinkerFlags, with_cuda=False, verbose=False)
		return _extension
