"""GIAANNproto_cudaObservedColumnUpdate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto CUDA sparse accumulator extension loader/wrapper for observed column updates

"""

import os
import time
import torch as pt

from torch.utils.cpp_extension import load
from torch.utils.cpp_extension import CUDA_HOME

_cudaObservedColumnUpdateExtension = None


def _raiseInvalidAccumulator(message):
	raise RuntimeError(message)
	return


def _validateShapeTuple(shapeTuple):
	if(not isinstance(shapeTuple, tuple)):
		raise RuntimeError("_validateShapeTuple error: shapeTuple must be tuple")
	if(len(shapeTuple) <= 0):
		raise RuntimeError("_validateShapeTuple error: shapeTuple must contain at least one dimension")
	totalElements = 1
	for dimValue in shapeTuple:
		if(not isinstance(dimValue, int)):
			raise RuntimeError("_validateShapeTuple error: all shape dimensions must be int")
		if(dimValue <= 0):
			raise RuntimeError("_validateShapeTuple error: all shape dimensions must be > 0")
		totalElements = totalElements * dimValue
		if(totalElements > 9223372036854775807):
			raise RuntimeError("_validateShapeTuple error: flattened index overflow (total elements exceed int64)")
	return


def _validateSparseInputs(indices, values, shapeTuple, tensorName):
	if(indices.dtype != pt.int64):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.indices dtype must be int64")
	if(values.dtype != pt.float32):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.values dtype must be float32")
	if(indices.device.type != "cuda" or values.device.type != "cuda"):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName} tensors must be CUDA tensors")
	if(indices.dim() != 2):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.indices must be rank-2")
	if(values.dim() != 1):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.values must be rank-1")
	if(indices.shape[1] != values.shape[0]):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.indices nnz must equal {tensorName}.values length")
	if(indices.shape[0] != len(shapeTuple)):
		raise RuntimeError(f"_validateSparseInputs error: {tensorName}.indices first dimension must equal rank(shape)")
	return


def _computeLinearKeys(indices, shapeTuple):
	shapeTensor = pt.tensor(shapeTuple, dtype=pt.int64, device=indices.device)
	strides = pt.ones((len(shapeTuple),), dtype=pt.int64, device=indices.device)
	for dimIndex in range(len(shapeTuple)-2, -1, -1):
		strides[dimIndex] = strides[dimIndex+1] * shapeTensor[dimIndex+1]
	linearKeys = (indices * strides.unsqueeze(1)).sum(dim=0)
	return linearKeys


def _unflattenLinearKeys(linearKeys, shapeTuple):
	indicesReverse = []
	remaining = linearKeys
	for dimIndex in range(len(shapeTuple)-1, -1, -1):
		dimSize = int(shapeTuple[dimIndex])
		indicesReverse.append(pt.remainder(remaining, dimSize))
		remaining = pt.div(remaining, dimSize, rounding_mode="floor")
	indicesReverse.reverse()
	indices = pt.stack(indicesReverse, dim=0)
	return indices


def _resolveHashCapacity(targetNNZ, hashCapacityMultiplier):
	if(hashCapacityMultiplier <= 1.0):
		raise RuntimeError("_resolveHashCapacity error: hashCapacityMultiplier must be > 1.0")
	hashCapacity = int(targetNNZ * hashCapacityMultiplier)
	if(hashCapacity < 8):
		hashCapacity = 8
	return hashCapacity


def _resolveOverflowCapacity(targetNNZ, overflowCapacityMultiplier):
	if(overflowCapacityMultiplier <= 0.0):
		raise RuntimeError("_resolveOverflowCapacity error: overflowCapacityMultiplier must be > 0.0")
	overflowCapacity = int(max(1, targetNNZ) * overflowCapacityMultiplier)
	if(overflowCapacity < 1):
		overflowCapacity = 1
	return overflowCapacity

def _ensureOverflowBufferCapacity(accumulator, requiredCapacity):
	if(requiredCapacity <= 0):
		raise RuntimeError("_ensureOverflowBufferCapacity error: requiredCapacity must be > 0")
	currentCapacity = int(accumulator["overflow_keys"].shape[0])
	if(currentCapacity < requiredCapacity):
		accumulator["overflow_keys"] = pt.empty((requiredCapacity,), dtype=pt.int64, device=accumulator["values"].device)
		accumulator["overflow_values"] = pt.empty((requiredCapacity,), dtype=pt.float32, device=accumulator["values"].device)
	return


def _buildHashState(accumulator):
	extensionModule = getCUDAObservedColumnUpdateExtension()
	targetLinearKeys = _computeLinearKeys(accumulator["indices"], accumulator["shape"])
	hashCapacity = _resolveHashCapacity(accumulator["values"].shape[0], accumulator["hash_capacity_multiplier"])
	hashKeys = pt.full((hashCapacity,), -1, dtype=pt.int64, device=accumulator["values"].device)
	hashSlots = pt.full((hashCapacity,), -1, dtype=pt.int64, device=accumulator["values"].device)
	maxProbe = hashCapacity
	extensionModule.build_sparse_hash(targetLinearKeys, hashKeys, hashSlots, maxProbe)
	accumulator["hash_keys"] = hashKeys
	accumulator["hash_slots"] = hashSlots
	accumulator["max_probe"] = maxProbe
	accumulator["rebuild_count"] = accumulator["rebuild_count"] + 1
	return


def _mergeOverflow(accumulator, overflowCountValue):	
	overflowKeys = accumulator["overflow_keys"][0:overflowCountValue]
	overflowValues = accumulator["overflow_values"][0:overflowCountValue]
	sortedOverflowKeys, sortOrder = pt.sort(overflowKeys)
	sortedOverflowValues = overflowValues[sortOrder]
	uniqueKeys, inverseIndices = pt.unique(sortedOverflowKeys, sorted=True, return_inverse=True)
	reducedValues = pt.zeros((uniqueKeys.shape[0],), dtype=sortedOverflowValues.dtype, device=sortedOverflowValues.device)
	reducedValues = reducedValues.scatter_add(0, inverseIndices, sortedOverflowValues)
	newIndices = _unflattenLinearKeys(uniqueKeys, accumulator["shape"])
	accumulator["indices"] = pt.cat([accumulator["indices"], newIndices], dim=1)
	accumulator["values"] = pt.cat([accumulator["values"], reducedValues], dim=0)
	return


def _validateAccumulator(accumulator):
	if(not isinstance(accumulator, dict)):
		_raiseInvalidAccumulator("_validateAccumulator error: accumulator must be a dict")
	requiredKeys = ["indices", "values", "shape", "hash_keys", "hash_slots", "overflow_keys", "overflow_values", "overflow_count", "hash_hits", "hash_capacity_multiplier", "overflow_capacity_multiplier", "rebuild_count", "overflow_total", "hash_hits_total", "update_calls", "update_total_seconds", "max_probe"]
	for key in requiredKeys:
		if(key not in accumulator):
			_raiseInvalidAccumulator(f"_validateAccumulator error: missing key '{key}'")
	_validateShapeTuple(accumulator["shape"])
	_validateSparseInputs(accumulator["indices"], accumulator["values"], accumulator["shape"], "accumulator")
	if(accumulator["hash_keys"].dtype != pt.int64 or accumulator["hash_slots"].dtype != pt.int64):
		_raiseInvalidAccumulator("_validateAccumulator error: hash tensors must be int64")
	if(accumulator["overflow_keys"].dtype != pt.int64):
		_raiseInvalidAccumulator("_validateAccumulator error: overflow_keys must be int64")
	if(accumulator["overflow_values"].dtype != pt.float32):
		_raiseInvalidAccumulator("_validateAccumulator error: overflow_values must be float32")
	if(accumulator["overflow_count"].dtype != pt.int64 or accumulator["overflow_count"].numel() != 1):
		_raiseInvalidAccumulator("_validateAccumulator error: overflow_count must be int64 with one element")
	if(accumulator["hash_hits"].dtype != pt.int64 or accumulator["hash_hits"].numel() != 1):
		_raiseInvalidAccumulator("_validateAccumulator error: hash_hits must be int64 with one element")
	return


def getCUDAObservedColumnUpdateExtension():
	global _cudaObservedColumnUpdateExtension
	if(_cudaObservedColumnUpdateExtension is None):
		if(not pt.cuda.is_available()):
			raise RuntimeError("getCUDAObservedColumnUpdateExtension error: CUDA is not available")
		if(CUDA_HOME is None):
			raise RuntimeError("getCUDAObservedColumnUpdateExtension error: CUDA_HOME is None (CUDA toolkit / nvcc not detected)")
		moduleDirectory = os.path.dirname(os.path.abspath(__file__))
		extensionDirectory = os.path.join(moduleDirectory, "cuda_observed_column_update_ext")
		cppSource = os.path.join(extensionDirectory, "sparse_accumulator.cpp")
		cudaSource = os.path.join(extensionDirectory, "sparse_accumulator_kernel.cu")
		buildDirectory = os.path.join(extensionDirectory, "build")
		os.makedirs(buildDirectory, exist_ok=True)
		_cudaObservedColumnUpdateExtension = load(name="giaann_cuda_observed_column_update", sources=[cppSource, cudaSource], build_directory=buildDirectory, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
	return _cudaObservedColumnUpdateExtension


def build_sparse_accumulator(indices, values, shape, hashCapacityMultiplier=2.0, overflowCapacityMultiplier=0.25):
	shapeTuple = tuple(int(dimValue) for dimValue in shape)
	_validateShapeTuple(shapeTuple)
	_validateSparseInputs(indices, values, shapeTuple, "target")
	if(indices.shape[1] == 0):
		emptyIndices = pt.empty((len(shapeTuple), 0), dtype=pt.int64, device=indices.device)
		emptyValues = pt.empty((0,), dtype=pt.float32, device=values.device)
		indices = emptyIndices
		values = emptyValues
	accumulator = {}
	accumulator["indices"] = indices
	accumulator["values"] = values
	accumulator["shape"] = shapeTuple
	accumulator["hash_capacity_multiplier"] = float(hashCapacityMultiplier)
	accumulator["overflow_capacity_multiplier"] = float(overflowCapacityMultiplier)
	accumulator["rebuild_count"] = 0
	accumulator["overflow_total"] = 0
	accumulator["hash_hits_total"] = 0
	accumulator["update_calls"] = 0
	accumulator["update_total_seconds"] = 0.0
	accumulator["overflow_count"] = pt.zeros((1,), dtype=pt.int64, device=values.device)
	accumulator["hash_hits"] = pt.zeros((1,), dtype=pt.int64, device=values.device)
	overflowCapacity = _resolveOverflowCapacity(values.shape[0], accumulator["overflow_capacity_multiplier"])
	accumulator["overflow_keys"] = pt.empty((overflowCapacity,), dtype=pt.int64, device=values.device)
	accumulator["overflow_values"] = pt.empty((overflowCapacity,), dtype=pt.float32, device=values.device)
	_buildHashState(accumulator)
	_validateAccumulator(accumulator)
	return accumulator


def accumulate_sparse_updates(accumulator, updateIndices, updateValues):
	_validateAccumulator(accumulator)
	_validateSparseInputs(updateIndices, updateValues, accumulator["shape"], "update")
	updateStartTime = time.perf_counter()
	if(updateValues.shape[0] > 0):
		extensionModule = getCUDAObservedColumnUpdateExtension()
		updateLinearKeys = _computeLinearKeys(updateIndices, accumulator["shape"])
		_ensureOverflowBufferCapacity(accumulator, int(updateValues.shape[0]))
		accumulator["overflow_count"].zero_()
		accumulator["hash_hits"].zero_()
		extensionModule.accumulate_sparse_updates(accumulator["values"], updateLinearKeys, updateValues, accumulator["hash_keys"], accumulator["hash_slots"], accumulator["overflow_keys"], accumulator["overflow_values"], accumulator["overflow_count"], accumulator["hash_hits"], accumulator["max_probe"])
		overflowCountValue = int(accumulator["overflow_count"].item())
		if(overflowCountValue > accumulator["overflow_keys"].shape[0]):
			raise RuntimeError("accumulate_sparse_updates error: overflow counter exceeded allocated overflow buffer capacity")
		if(overflowCountValue > 0):
			_mergeOverflow(accumulator, overflowCountValue)
			_buildHashState(accumulator)
		accumulator["overflow_total"] = accumulator["overflow_total"] + overflowCountValue
		accumulator["hash_hits_total"] = accumulator["hash_hits_total"] + int(accumulator["hash_hits"].item())
	accumulator["update_calls"] = accumulator["update_calls"] + 1
	accumulator["update_total_seconds"] = accumulator["update_total_seconds"] + (time.perf_counter() - updateStartTime)
	_validateAccumulator(accumulator)
	return accumulator


def export_coo(accumulator):
	_validateAccumulator(accumulator)
	resultIndices = accumulator["indices"]
	resultValues = accumulator["values"]
	stats = {}
	stats["hash_hit_count"] = int(accumulator["hash_hits_total"])
	stats["overflow_count"] = int(accumulator["overflow_total"])
	stats["rebuild_count"] = int(accumulator["rebuild_count"])
	stats["update_calls"] = int(accumulator["update_calls"])
	stats["average_update_latency_seconds"] = float(accumulator["update_total_seconds"]/accumulator["update_calls"]) if accumulator["update_calls"] > 0 else 0.0
	stats["hash_hit_rate"] = float(accumulator["hash_hits_total"]/max(1, accumulator["hash_hits_total"] + accumulator["overflow_total"]))
	return resultIndices, resultValues, stats
