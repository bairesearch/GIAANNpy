"""Ordered GPU commits for CPU-resident GIAANN sparse training batches."""
from GIAANNcmn_globalDefs import *

if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
	import os
	import sysconfig
	from importlib import util
	import numpy as np
	from torch.utils.cpp_extension import load
	import GIAANNcmn_gpuCustomSparseTensorDriver as driver
	_extension = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		import GIAANNcmn_databasePacked as packed
		_recordType = packed.getEntryDtype()


def mergeConnections(state):
	result = None
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		import GIAANNcmn_trainSequenceBatch as batches
		import GIAANNcmn_trainDeferredSourcePreparation as preparation
		eligible = not debugPrintTrainSectionTimesSourceFeatureConnections and not trainVerifyConnectionNonexistentAcrossBranchesOptimisation1 and not optimiseTrainConnectionOverlay and sum(tensor._nnz() for keys, tensor in state.connections) >= gpuCustomSparseTensorDriverMinEntries
		fusedPreparation = eligible and optimiseTrainDeferredSourcePreparation
		if(optimiseTrainDeferredSourcePreparation and not fusedPreparation):
			preparation.prepare(state)
		if(eligible):
			result = merge(state, state.deferredSourcePreparation if fusedPreparation else {})
		if(fusedPreparation):
			if(result is None):
				preparation.prepare(state)
			else:
				prepareOutstanding(state, result[1])
		if(result is None):
			result = batches.mergePackets(state, referencePackets(state.connections), state.sourceShape, True)
	return result


def merge(state, deferred):
	result = None
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections
		dtype = _recordType if optimisationStoreDatabaseResidentCoordinatesAsInt32 else np.dtype(np.int32)
		with pt.cuda.device(driver.cudaDeviceIndex()):
			keys, updates = getExtension().merge((useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver), state.columns, deferred, state.connections, state.sourceShape, trainSequenceBatchBucketDimension, optimisationStoreDatabaseResidentCoordinatesAsInt32, False, PackedDatabaseResidentFeatureConnections, dtype, gpuCustomSparseTensorDriverCommitTileEntries, trainSequenceBatchInvalidInput, historyRatio(), mergeSparseHistory if useGPUCPUhybridCustomSparseTensorDriver else None)
		if(updates is not None):
			result = keys, updates
	return result


def mergeNeurons(state, shape):
	result = None
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		import GIAANNcmn_trainSequenceBatch as batches
		if(sum(tensor._nnz() for keys, tensor in state.neurons) >= gpuCustomSparseTensorDriverMinEntries):
			with pt.cuda.device(driver.cudaDeviceIndex()):
				keys, updates = getExtension().merge((useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver), state.columns, {}, state.neurons, shape, trainSequenceBatchBucketDimension, False, True, None, np.dtype(np.int32), gpuCustomSparseTensorDriverCommitTileEntries, trainSequenceBatchInvalidInput, historyRatio(), mergeSparseHistory if useGPUCPUhybridCustomSparseTensorDriver else None)
			if(updates is not None):
				result = keys, updates.outputs
		if(result is None):
			result = batches.mergePackets(state, referencePackets(state.neurons), shape, False)
	return result


def historyRatio():
	result = 0
	if(useGPUCPUhybridCustomSparseTensorDriver and not optimisationStoreDatabaseResidentCoordinatesAsInt32):
		result = gpuCustomSparseTensorDriverHistoryRatio
	return result


def mergeSparseHistory(sources, updates, mappings, target):
	result = None
	if(useGPUCPUhybridCustomSparseTensorDriver):
		import GIAANNcmn_trainSequenceBatch as batches
		result = batches.getExtension().mergeSequenceBatch(optimiseTrainSequenceBatch, sources, updates, mappings, list(target), trainSequenceBatchBucketDimension, trainSequenceBatchGrainSize, trainSequenceBatchInvalidInput)
	return result


def referencePackets(packets):
	result = None
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		result = [(keys.tolist(), tensor) for keys, tensor in packets]
	return result


def prepareOutstanding(state, updates):
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		if(not state.committing or state.failed or not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam or not state.database.observedColumnsRAMLoaded or state.database.inferenceMode or pt.device(deviceSparse).type != trainSequenceBatchDeviceType or pt.device(deviceDatabase).type != trainSequenceBatchDeviceType):
			raise RuntimeError(trainSequenceBatchInvalidState)
		updates.prepareOutstanding(state.deferredSourcePreparation, state.sourceShape, deviceSparse, trainSequenceBatchInvalidInput)
		state.deferredSourcePreparation.clear()
	return


def isPending(updates):
	result = False
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		result = _extension is not None and isinstance(updates, _extension.Pending)
	return result


def getExtension():
	global _extension
	result = None
	if(useGPUcustomSparseTensorDriver or useGPUCPUhybridCustomSparseTensorDriver):
		if(_extension is None):
			folder = os.path.dirname(os.path.abspath(__file__))
			binary = os.path.join(folder, gpuCustomSparseTensorDriverCommitName+sysconfig.get_config_var("EXT_SUFFIX"))
			if(os.path.isfile(binary)):
				spec = util.spec_from_file_location(gpuCustomSparseTensorDriverCommitName, binary)
				_extension = util.module_from_spec(spec)
				spec.loader.exec_module(_extension)
			else:
				linkFlags = gpuCustomSparseTensorDriverLinkerFlags + ["-L"+folder, "-Wl,-rpath,"+folder, "-l:"+gpuCustomSparseTensorDriverLibrary]
				_extension = load(name=gpuCustomSparseTensorDriverCommitName, sources=[os.path.join(folder, gpuCustomSparseTensorDriverCommitSource)], extra_cflags=gpuCustomSparseTensorDriverCompilerFlags, extra_ldflags=linkFlags, with_cuda=False, verbose=False)
		result = _extension
	return result
