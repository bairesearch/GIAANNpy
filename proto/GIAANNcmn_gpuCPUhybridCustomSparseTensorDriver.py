"""GIAANNcmn_gpuCPUhybridCustomSparseTensorDriver.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common gpu CPU hybrid Custom Sparse Tensor Driver

Adaptive ordered CPU kernels and CUDA offloads for GIAANN sparse operations.

"""



from GIAANNcmn_globalDefs import *

if(useGPUCPUhybridCustomSparseTensorDriver):
	import ctypes as C
	from GIAANNcmn_gpuCustomSparseTensorDriver import canNormalise, coalesceTensor, gatherCanonical, nativeLibrary, normalise, propagateBlockSparse, sliceCanonical, supportsTensor, unitEntryStride


def addTensors(left, right, concatenateUpdates=False):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		left = coalesceTensor(left)
		right = coalesceTensor(right)
		result = None
		if(canMerge(left, right) and ordinaryValues(left, right)):
			if(min(left._nnz(), right._nnz())*gpuCustomSparseTensorDriverDeltaRatio <= max(left._nnz(), right._nnz())):
				result = mergeDelta(left, right)
			else:
				result = mergeOrdered(left, right)
		if(result is None):
			if(concatenateUpdates):
				result = pt.sparse_coo_tensor(pt.cat((left.indices(), right.indices()), dim=1), pt.cat((left.values(), right.values())), size=left.shape, dtype=left.dtype, device=left.device).coalesce()
			else:
				result = (left+right).coalesce()
	return result


def ordinaryValues(left, right):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		result = True
		for tensor in (left, right):
			values = tensor.values()
			if(nativeLibrary().coo_special_values(values.data_ptr(), values.stride(0), values.numel(), threads)):
				result = False
				break
		if(not result):
			# Preserve input validation when the CPU must choose the arithmetic order.
			dimensions = (C.c_uint64*left.dim())(*left.shape)
			for tensor in (left, right):
				indices = unitEntryStride(tensor.indices())
				if(nativeLibrary().coo_validate(indices.data_ptr(), indices.stride(0), dimensions, tensor.dim(), tensor._nnz(), threads)):
					raise ValueError(gpuCustomSparseTensorDriverInvalidKey)
	return result


def mergeDelta(left, right):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		baseFirst = left._nnz() >= right._nnz()
		base, update = (left, right) if baseFirst else (right, left)
		rank = base.dim()
		sizes = (C.c_uint64*rank)(*base.shape)
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		bi, ui = unitEntryStride(base.indices()), unitEntryStride(update.indices())
		bv, uv = base.values().contiguous(), update.values().contiguous()
		positions = pt.empty(update._nnz(), dtype=pt.int32, device=base.device)
		output = pt.empty_like(positions)
		count = nativeLibrary().coo_delta_positions(bi.data_ptr(), bi.stride(0), base._nnz(), ui.data_ptr(), ui.stride(0), update._nnz(), rank, positions.data_ptr(), output.data_ptr())
		indices = pt.empty((rank, count), dtype=pt.int64, device=base.device)
		values = pt.empty(count, dtype=base.dtype, device=base.device)
		if(nativeLibrary().coo_delta_emit(bi.data_ptr(), bi.stride(0), bv.data_ptr(), base._nnz(), ui.data_ptr(), ui.stride(0), uv.data_ptr(), update._nnz(), rank, positions.data_ptr(), output.data_ptr(), count, threads, indices.data_ptr(), values.data_ptr(), baseFirst, sizes)):
			raise ValueError(gpuCustomSparseTensorDriverInvalidKey)
		result = pt.sparse_coo_tensor(indices, values, size=base.shape, is_coalesced=True)
	return result


def mergeOrdered(left, right):
	"""Intersect canonical key streams and write disjoint ranges of the exact-size union."""
	if(useGPUCPUhybridCustomSparseTensorDriver):
		rank = left.dim()
		sizes = (C.c_uint64*rank)(*left.shape)
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		operands = []
		for tensor in (left, right):
			indices = unitEntryStride(tensor.indices())
			values = tensor.values().contiguous()
			keys = pt.empty(tensor._nnz(), dtype=pt.int64, device=tensor.device)
			if(nativeLibrary().cpu_pack_rows(indices.data_ptr(), keys.data_ptr(), sizes, rank, tensor._nnz(), threads, indices.stride(0))):
				raise ValueError(gpuCustomSparseTensorDriverInvalidKey)
			operands.append((indices, values, keys))
		(ai, av, ak), (bi, bv, bk) = operands
		tiles = (max(ak.numel(), bk.numel())-1)//gpuCustomSparseTensorDriverMergeTile+1
		partitions = pt.empty((tiles+1, gpuCustomSparseTensorDriverPartitionFields), dtype=pt.int32, device=left.device)
		count = nativeLibrary().coo_partition(ak.data_ptr(), ak.numel(), bk.data_ptr(), bk.numel(), partitions.data_ptr(), tiles, threads)
		indices = pt.empty((rank, count), dtype=pt.int64, device=left.device)
		values = pt.empty(count, dtype=left.dtype, device=left.device)
		nativeLibrary().coo_emit(ai.data_ptr(), av.data_ptr(), ak.data_ptr(), ai.stride(0), bi.data_ptr(), bv.data_ptr(), bk.data_ptr(), bi.stride(0), rank, partitions.data_ptr(), tiles, threads, indices.data_ptr(), values.data_ptr(), count)
		result = pt.sparse_coo_tensor(indices, values, size=left.shape, is_coalesced=True)
	return result


def selectLeading(tensor, index):
	"""A leading coordinate occupies one contiguous range of a canonical COO tensor."""
	if(useGPUCPUhybridCustomSparseTensorDriver):
		if(isinstance(tensor, pt.Tensor) and supportsTensor(tensor) and tensor.is_coalesced() and tensor.dim() > 1 and type(index) is int and 0 <= index < tensor.size(0)):
			result = sliceCanonical(tensor, 0, index, tensor.device)
		else:
			result = tensor[index]
	return result


def supportsActivation(tensor):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		result = supportsTensor(tensor) and tensor.is_coalesced() and tensor.dim() == inferenceLeakyIntegrateAndFireNeuronTensorRank and tensor.size(0)*tensor.size(1) <= gpuCustomSparseTensorDriverMaxLookupFibers and gpuCustomSparseTensorDriverMergeTile <= tensor._nnz() <= gpuCustomSparseTensorDriverMaxEntries
	return result


def findNeuron(tensor, column, feature):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		indices = unitEntryStride(tensor.indices())
		positions = pt.empty(tensor.size(0)*tensor.size(1), dtype=pt.int64, device=tensor.device)
		count = nativeLibrary().coo_find_neuron(indices.data_ptr(), indices.stride(0), tensor._nnz(), tensor.size(0), tensor.size(1), column, feature, positions.data_ptr())
		result = positions[:count]
	return result


def validateCandidateState(tensor):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		indices = unitEntryStride(tensor.indices())
		values = tensor.values().contiguous()
		shape = (C.c_int64*tensor.dim())(*tensor.shape)
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		if(nativeLibrary().coo_validate(indices.data_ptr(), indices.stride(0), shape, tensor.dim(), tensor._nnz(), threads) or nativeLibrary().coo_nonnegative(values.data_ptr(), values.numel(), threads)):
			raise RuntimeError(inferenceCandidateLookupInvalidState)


def candidatePositions(tensor, keys, segments):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		indices = unitEntryStride(tensor.indices())
		shape = (C.c_int64*tensor.dim())(*tensor.shape)
		keys = keys.contiguous()
		segmentIDs = sorted(set(segments))
		fibers = (C.c_int64*len(segmentIDs))(*segmentIDs)
		positions = pt.empty(min(tensor._nnz(), keys.numel()*len(fibers)), dtype=pt.long, device=tensor.device)
		count = nativeLibrary().coo_candidates(indices.data_ptr(), indices.stride(0), tensor._nnz(), shape, keys.data_ptr(), keys.numel(), fibers, len(fibers), positions.data_ptr())
		result = positions[:count]
	return result


def decaySoma(tensor, soma, rate, multiply):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		indices = unitEntryStride(tensor.indices())
		inputValues = tensor.values().contiguous()
		scaled = pt.empty_like(inputValues)
		n = tensor._nnz()
		tile = gpuCustomSparseTensorDriverMergeTile
		tiles = (n-1)//tile+1
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		offsets = pt.empty(tiles+1, dtype=pt.int32, device=tensor.device)
		count = nativeLibrary().coo_decay_plan(indices[inferenceLeakyIntegrateAndFireSegmentDimension].data_ptr(), inputValues.data_ptr(), n, soma, rate, multiply, threads, scaled.data_ptr(), offsets.data_ptr(), tiles, tile)
		if(count < 0):
			raise RuntimeError(gpuCustomSparseTensorDriverInvalidDecay)
		outIndices = pt.empty((tensor.dim(), count), dtype=pt.int64, device=tensor.device)
		values = pt.empty(count, dtype=tensor.dtype, device=tensor.device)
		if(count):
			nativeLibrary().coo_decay_emit(indices.data_ptr(), indices.stride(0), tensor.dim(), scaled.data_ptr(), n, offsets.data_ptr(), tiles, tile, threads, outIndices.data_ptr(), values.data_ptr(), count)
		result = pt.sparse_coo_tensor(outIndices, values, size=tensor.shape, is_coalesced=True)
	return result


def canMerge(left, right):
	if(useGPUCPUhybridCustomSparseTensorDriver):
		result = supportsTensor(left) and supportsTensor(right) and left.shape == right.shape and min(left._nnz(), right._nnz()) > 0 and gpuCustomSparseTensorDriverMinEntries <= left._nnz()+right._nnz() <= gpuCustomSparseTensorDriverMaxEntries
	return result
