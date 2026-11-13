"""GIAANNcmn_branchVerificationStorage.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common branch Verification Storage

Indexed, lossless per-column storage for branch-verification connections.

"""



import os
import struct
import hashlib
import tempfile
from collections import OrderedDict
import numpy as np
import torch as pt
from GIAANNcmn_globalDefs import *

if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
	_indexCache = OrderedDict()
	_source, _offset, _count, _shape, _digest = branchVerifyPackedIndexFields
	_indices, _value = branchVerifyPackedRecordFields
	_indexDtype = np.dtype([(_source, branchVerifyPackedInt64), (_offset, branchVerifyPackedInt64), (_count, branchVerifyPackedInt64), (_shape, branchVerifyPackedInt64, (branchVerifySourceRank,)), (_digest, branchVerifyPackedDigestDtype)])
	_recordDtype = np.dtype([(_indices, branchVerifyPackedInt32, (branchVerifySourceRank,)), (_value, branchVerifyPackedFloat32)])
	_header = struct.Struct(branchVerifyPackedHeaderFormat)


def saveColumn(column, sourceFeatureIndices):
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		if(sourceFeatureIndices is not None or column.databaseNetworkObject.inferenceMode or not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			raise RuntimeError(branchVerifyPackedInvalid)
		path = getPath(column.conceptIndex)
		folder = os.path.dirname(path)
		if(any(name.endswith(pytorchTensorFileExtension) for name in os.listdir(folder))):
			raise RuntimeError(branchVerifyPackedInvalid)
		sources = sorted(column.featureConnectionsBySourceFeature)
		if(os.path.isfile(path) and not set(listSources(column.conceptIndex)).issubset(sources)):
			raise RuntimeError(branchVerifyPackedInvalid)
		stored = []
		for source in sources:
			tensor = column.featureConnectionsBySourceFeature[source]
			validateTensor(tensor)
			if(tensor._nnz() > branchVerifyZero):
				stored.append(source)
		index = np.zeros(len(stored), dtype=_indexDtype)
		payloadStart = _header.size + index.nbytes
		fileDescriptor, temporaryPath = tempfile.mkstemp(prefix=branchVerifyPackedTemporaryPrefix, dir=folder)
		try:
			with os.fdopen(fileDescriptor, branchVerifyPackedWriteMode) as handle:
				handle.seek(payloadStart)
				for position, source in enumerate(stored):
					tensor = column.featureConnectionsBySourceFeature[source]
					records = np.empty(tensor._nnz(), dtype=_recordDtype)
					records[_indices] = tensor.indices().transpose(branchVerifyZero, branchVerifyOne).numpy()
					records[_value] = tensor.values().numpy()
					payload = records.tobytes()
					index[position][_source] = source
					index[position][_offset] = handle.tell()
					index[position][_count] = tensor._nnz()
					index[position][_shape] = tuple(tensor.size())
					index[position][_digest] = np.void(hashlib.sha256(payload).digest())
					handle.write(payload)
				indexBytes = index.tobytes()
				handle.seek(branchVerifyZero)
				handle.write(_header.pack(branchVerifyPackedMagic, branchVerifySchema, len(stored), _recordDtype.itemsize, hashlib.sha256(indexBytes).digest()))
				handle.write(indexBytes)
				handle.flush()
			os.replace(temporaryPath, path)
		finally:
			if(os.path.exists(temporaryPath)):
				os.remove(temporaryPath)
		if(optimisationGetFeatureConnectionsForSourceFeatureCache):
			column.storedSourceFeatureIndicesCache = set(stored)
	return


def listSources(conceptIndex):
	result = []
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		path = getPath(conceptIndex)
		folder = os.path.dirname(path)
		if(os.path.isdir(folder) and any(name.endswith(pytorchTensorFileExtension) for name in os.listdir(folder))):
			raise RuntimeError(branchVerifyPackedInvalid)
		if(os.path.isfile(path)):
			result = getIndex(path)[_source].tolist()
		elif(os.path.isfile(os.path.join(observedColumnsDir, observedColumnFolderNamePrefix+str(conceptIndex), observedColumnMetadataFileName))):
			raise RuntimeError(branchVerifyPackedInvalid)
	return result


def loadSource(conceptIndex, sourceFeatureIndex, targetDevice):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		path = getPath(conceptIndex)
		index = getIndex(path)
		position = int(np.searchsorted(index[_source], sourceFeatureIndex))
		if(position >= len(index) or int(index[position][_source]) != sourceFeatureIndex):
			raise RuntimeError(branchVerifyPackedInvalid)
		row = index[position]
		with open(path, branchVerifyPackedReadMode) as handle:
			handle.seek(int(row[_offset]))
			payload = handle.read(int(row[_count])*_recordDtype.itemsize)
		if(hashlib.sha256(payload).digest() != row[_digest].tobytes()):
			raise RuntimeError(branchVerifyPackedInvalid)
		records = np.frombuffer(payload, dtype=_recordDtype)
		indices = pt.from_numpy(np.ascontiguousarray(records[_indices].transpose(), dtype=np.int64))
		values = pt.from_numpy(np.array(records[_value], copy=True))
		shape = tuple(int(value) for value in row[_shape])
		result = pt.sparse_coo_tensor(indices, values, shape, dtype=pt.float32, is_coalesced=True, check_invariants=True)
		if(not bool(pt.isfinite(values).all())):
			raise RuntimeError(branchVerifyPackedInvalid)
		if(targetDevice is not None):
			result = result.to(targetDevice)
	return result


def getPath(conceptIndex):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		if(type(conceptIndex) is not int or conceptIndex < branchVerifyZero):
			raise RuntimeError(branchVerifyPackedInvalid)
		result = os.path.join(observedColumnsDir, observedColumnFolderNamePrefix+str(conceptIndex), observedColumnFeatureConnectionsFolderName, branchVerifyPackedFileName)
	return result


def getIndex(path):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		stat = os.stat(path)
		cacheKey = path, stat.st_ino, stat.st_mtime_ns, stat.st_size
		if(cacheKey in _indexCache):
			result = _indexCache.pop(cacheKey)
		else:
			result = readIndex(path, stat.st_size)
		_indexCache[cacheKey] = result
		while(len(_indexCache) > branchVerifyPackedIndexCacheSize):
			_indexCache.popitem(last=False)
	return result


def readIndex(path, fileSize):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		with open(path, branchVerifyPackedReadMode) as handle:
			headerBytes = handle.read(_header.size)
			if(len(headerBytes) != _header.size):
				raise RuntimeError(branchVerifyPackedInvalid)
			magic, schema, count, recordSize, digest = _header.unpack(headerBytes)
			if(magic != branchVerifyPackedMagic or schema != branchVerifySchema or recordSize != _recordDtype.itemsize or count > (fileSize-_header.size)//_indexDtype.itemsize):
				raise RuntimeError(branchVerifyPackedInvalid)
			indexBytes = handle.read(count*_indexDtype.itemsize)
			if(hashlib.sha256(indexBytes).digest() != digest):
				raise RuntimeError(branchVerifyPackedInvalid)
			result = np.frombuffer(indexBytes, dtype=_indexDtype)
		if(np.any(result[_source] < branchVerifyZero) or np.any(result[_source][branchVerifyOne:] <= result[_source][:-branchVerifyOne]) or np.any(result[_count] <= branchVerifyZero) or np.any(result[_shape] <= branchVerifyZero) or np.any(result[_shape] > branchVerifyInt32Maximum+branchVerifyOne)):
			raise RuntimeError(branchVerifyPackedInvalid)
		payloadStart = _header.size + len(indexBytes)
		counts = result[_count]
		if(np.any(counts > fileSize//_recordDtype.itemsize) or int(counts.sum()) > (fileSize-payloadStart)//_recordDtype.itemsize):
			raise RuntimeError(branchVerifyPackedInvalid)
		expectedOffsets = payloadStart + (np.cumsum(counts)-counts)*_recordDtype.itemsize
		if(not np.array_equal(result[_offset], expectedOffsets) or payloadStart+int(counts.sum())*_recordDtype.itemsize != fileSize):
			raise RuntimeError(branchVerifyPackedInvalid)
	return result


def validateTensor(tensor):
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
		if(not pt.is_tensor(tensor) or tensor.layout != pt.sparse_coo or tensor.dim() != branchVerifySourceRank or tensor.device.type != branchVerifyCPU or tensor.dtype != pt.float32 or not tensor.is_coalesced() or any(size <= branchVerifyZero or size > branchVerifyInt32Maximum+branchVerifyOne for size in tensor.size())):
			raise RuntimeError(branchVerifyPackedInvalid)
		if(not bool(pt.isfinite(tensor.values()).all()) or bool(pt.any(tensor.indices() < branchVerifyZero)) or bool(pt.any(tensor.indices() > branchVerifyInt32Maximum))):
			raise RuntimeError(branchVerifyPackedInvalid)
		if(bool(pt.any(tensor.indices() >= pt.tensor(tensor.size(), dtype=pt.long).unsqueeze(branchVerifyOne)))):
			raise RuntimeError(branchVerifyPackedInvalid)
	return
