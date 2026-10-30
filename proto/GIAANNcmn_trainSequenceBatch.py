"""Ordered CPU training batches. Stored tensors remain canonical sparse COO.

Explicit flushDatabase calls expose all completed sequences before raw tensor inspection.
Supported training, source-reader and persistence entry points flush automatically.
"""

from GIAANNcmn_globalDefs import *

if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
	if(optimiseTrainSequenceBatch):
		import GIAANNcmn_trainPackedSequenceBatch

if(optimiseTrainSequenceBatch):
	import os
	import threading
	import time
	from torch.utils.cpp_extension import load
	import GIAANNcmn_debug
	if(patchTrainColumnStepsRepeatedColumns):
		import GIAANNcmn_trainColumnSteps
	_sequenceBatchExtension = None
if(optimiseTrainDeferredSourcePreparation):
	import GIAANNcmn_trainDeferredSourcePreparation
if(optimiseTrainCommitThreads):
	import GIAANNcmn_trainCommitThreads


def queueSequence(sequence, observedColumns, inference, mode):
	if(optimiseTrainSequenceBatch):
		if(inference or mode != trainSequenceBatchUpdateMode):
			raise RuntimeError(trainSequenceBatchInvalidConfiguration)
		database = sequence.databaseNetworkObject
		state = getState(database)
		started = time.perf_counter() if debugPrintTrainSectionTimes else None
		neuronKeys, neuronUpdates, connectionKeys, connectionUpdates = prepareSequence(sequence, state.sourceShape)
		entries = neuronUpdates._nnz() + connectionUpdates._nnz()
		if(debugPrintTrainSectionTimes):
			GIAANNcmn_debug.debugTrainSectionTimesAdd(database, trainSequenceBatchPrepareTiming, time.perf_counter() - started)
		if(entries > trainSequenceBatchMaxPendingEntries):
			raise ValueError(trainSequenceBatchInvalidLimit)
		if(state.pendingEntries + entries > trainSequenceBatchMaxPendingEntries):
			flushDatabase(database)
		columns = sequence.getObservedColumnsByConceptIndex(observedColumns)
		for concept, column in columns.items():
			if(column.databaseNetworkObject is not database or not column.isDatabaseResidentConnectionStorage() or concept < trainSequenceBatchZero or concept >= database.c or (concept in state.columns and state.columns[concept] is not column)):
				raise RuntimeError(trainSequenceBatchInvalidState)
		state.columns.update(columns)
		state.neurons.append((neuronKeys.tolist(), neuronUpdates))
		state.connections.append((connectionKeys.tolist(), connectionUpdates))
		state.sequenceCount += trainSequenceBatchOne
		state.pendingEntries += entries
		if(state.sequenceCount >= trainSequenceBatchSize):
			flushDatabase(database)
	return


def flushDatabase(database):
	if(optimiseTrainSequenceBatch):
		state = database.trainSequenceBatchState
		if(state is not None):
			if(state.ownerThread != threading.get_ident()):
				raise RuntimeError(trainSequenceBatchInvalidState)
			if(state.failed):
				raise RuntimeError(trainSequenceBatchFailed)
			if(optimiseTrainDeferredSourcePreparation):
				GIAANNcmn_trainDeferredSourcePreparation.flushWithoutSequences(database)
			if(not state.committing and state.sequenceCount > trainSequenceBatchZero):
				state = getState(database)
				started = time.perf_counter() if debugPrintTrainSectionTimes else None
				state.committing = True
				if(optimiseTrainCommitThreads):
					commitThreads = None
				try:
					if(optimiseTrainCommitThreads):
						commitThreads = GIAANNcmn_trainCommitThreads.begin(database)
					if(optimiseTrainDeferredSourcePreparation):
						GIAANNcmn_trainDeferredSourcePreparation.prepare(state)
					connectionKeys, connectionUpdates = mergePackets(state, state.connections, state.sourceShape, True)
					neuronShape = state.sourceShape[:trainSequenceBatchBucketDimension] + (database.f,)
					neuronKeys, neuronUpdates = mergePackets(state, state.neurons, neuronShape, False)
					# Both native merges must succeed before any tensors are published.
					for key, tensor in zip(connectionKeys, connectionUpdates):
						concept, feature = divmod(key, database.f)
						if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
							GIAANNcmn_trainPackedSequenceBatch.publishConnection(state.columns[concept], feature, tensor)
						else:
							state.columns[concept].setFeatureConnectionsForSourceFeature(feature, tensor)
					for key, tensor in zip(neuronKeys, neuronUpdates):
						state.columns[key].featureNeurons = tensor
					state.connections.clear()
					state.neurons.clear()
					state.columns.clear()
					state.sequenceCount = trainSequenceBatchZero
					state.pendingEntries = trainSequenceBatchZero
				except BaseException:
					state.failed = True
					raise
				finally:
					if(optimiseTrainCommitThreads):
						try:
							if(commitThreads is not None):
								GIAANNcmn_trainCommitThreads.restore(commitThreads)
						except BaseException:
							state.failed = True
							raise
						finally:
							state.committing = False
					else:
						state.committing = False
				if(debugPrintTrainSectionTimes):
					GIAANNcmn_debug.debugTrainSectionTimesAdd(database, trainSequenceBatchCommitTiming, time.perf_counter() - started)
	return


def createTemporalConnections(active, columns, words, cs, fs, includeSameTime, mode, columnSegments, featureSegments, internalColumns, linkFirstColumn, linkFirstFeature, lastColumnSegment, sequence):
	result = None
	if(optimiseTrainSequenceBatch):
		columnSteps = None
		if(patchTrainColumnStepsRepeatedColumns):
			columnSteps = GIAANNcmn_trainColumnSteps.calculateColumnSteps(sequence, columns.device)
		extension = getExtension()
		result = extension.createTemporalConnections(optimiseTrainSequenceBatch, active, columns, words, cs, fs, includeSameTime, trainConnectionsAllowSelfTransitions, debugConnectNodesToNextNodesInSequenceOnly, debugConnectColumnsToNextColumnsInSequenceOnly, useSANI, arrayIndexSegmentLast, mode, parallelisation2TemporalModeColumns, parallelisation2TemporalModeFeatures, parallelisation2TemporalModeCombined, columnSegments, featureSegments, internalColumns, linkFirstColumn, linkFirstFeature, lastColumnSegment, parallelisation2MaximumSegments, trainSequenceBatchGrainSize, patchTrainColumnStepsRepeatedColumns, columnSteps, trainSequenceBatchInvalidInput)
	return result


def prepareSequence(sequence, sourceShape):
	result = None
	if(optimiseTrainSequenceBatch):
		if(optimiseTrainFusedSequenceUpdates):
			result = sequence.trainFusedSequencePacket
			if(result is None):
				raise RuntimeError(trainFusedInvalidInput)
		else:
			import GIAANNcmn_cpuParallelisation2
			database = sequence.databaseNetworkObject
			features, subset = sequence.getObservedColumnFeatureIndices()
			neurons = sequence.extractSequenceFeaturePropertySparse(database.arrayIndexPropertiesStrengthIndex).coalesce()
			connections = sequence.extractSequenceConnectionPropertySparse(database.arrayIndexPropertiesStrengthIndex).coalesce()
			neuronIndices, neuronValues = neurons.indices(), neurons.values()
			connectionIndices, connectionValues = connections.indices(), connections.values()
			if(subset.numel() != sequence.fs):
				mask = sequence.buildMaskLookup(sequence.fs, subset, features.device)
				neuronIndices, neuronValues = sequence.filterSparseByFeatureMask(neuronIndices, neuronValues, mask, trainSequenceBatchNeuronFeatureDimensions)
				connectionIndices, connectionValues = sequence.filterSparseByFeatureMask(connectionIndices, connectionValues, mask, trainSequenceBatchConnectionFeatureDimensions)
			concepts = sequence.conceptIndicesInSequenceObservedTensor
			neuronKeys = pt.unique(concepts[neuronIndices[trainSequenceBatchNeuronConceptDimension]], sorted=True)
			neuronShape = sourceShape[:trainSequenceBatchBucketDimension] + (neuronKeys.numel(), database.f)
			neuronUpdates = sequence.buildFeaturePropertyUpdateSparseBatched(neuronIndices, neuronValues, database.arrayIndexPropertiesStrengthIndex, features, concepts, neuronShape, neuronKeys).coalesce()
			connectionKeys, connectionUpdates = GIAANNcmn_cpuParallelisation2.prepareConnectionUpdates(connectionIndices, connectionValues, features, concepts, sourceShape, database.arrayIndexPropertiesStrengthIndex)
			result = (neuronKeys, neuronUpdates, connectionKeys, connectionUpdates)
	return result


def mergePackets(state, packets, shape, connections):
	result = None
	if(optimiseTrainSequenceBatch):
		if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
			result = GIAANNcmn_trainPackedSequenceBatch.mergePackets(state, packets, shape, connections)
		else:
			keys = sorted({key for packetKeys, tensor in packets for key in packetKeys})
			positions = {key: index for index, key in enumerate(keys)}
			mappings = [pt.tensor([positions[key] for key in packetKeys], dtype=pt.long) for packetKeys, tensor in packets]
			sources = []
			for key in keys:
				if(connections):
					concept, feature = divmod(key, state.database.f)
					column = state.columns[concept]
					if(feature not in column.featureConnectionsBySourceFeature or feature not in column.loadedSourceFeatureIndices):
						raise RuntimeError(trainSequenceBatchInvalidState)
					source = column.featureConnectionsBySourceFeature[feature]
				else:
					source = state.columns[key].featureNeurons
				sources.append(source.coalesce())
			extension = getExtension()
			merged = extension.mergeSequenceBatch(optimiseTrainSequenceBatch, sources, [tensor for packetKeys, tensor in packets], mappings, list(shape), trainSequenceBatchBucketDimension, trainSequenceBatchGrainSize, trainSequenceBatchInvalidInput)
			result = (keys, merged)
	return result


def getState(database):
	result = None
	if(optimiseTrainSequenceBatch):
		if(database.inferenceMode or not database.observedColumnsRAMLoaded or database.c < trainSequenceBatchOne or database.f < trainSequenceBatchOne or database.arrayNumberOfProperties != trainSequenceBatchOne):
			raise RuntimeError(trainSequenceBatchInvalidState)
		if(type(trainSequenceBatchSize) is not int or trainSequenceBatchSize < trainSequenceBatchOne or type(trainSequenceBatchMaxPendingEntries) is not int or trainSequenceBatchMaxPendingEntries < trainSequenceBatchOne):
			raise ValueError(trainSequenceBatchInvalidLimit)
		shape = (database.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, database.c, database.f)
		if(database.trainSequenceBatchState is None):
			database.trainSequenceBatchState = SequenceBatchState(database, shape)
		result = database.trainSequenceBatchState
		if(result.failed):
			raise RuntimeError(trainSequenceBatchFailed)
		if(result.database is not database or result.sourceShape != shape or result.ownerThread != threading.get_ident() or result.committing):
			raise RuntimeError(trainSequenceBatchInvalidState)
	return result


def getExtension():
	global _sequenceBatchExtension
	result = None
	if(optimiseTrainSequenceBatch):
		if(_sequenceBatchExtension is None):
			_sequenceBatchExtension = load(name=trainSequenceBatchExtensionName, sources=[os.path.join(os.path.dirname(__file__), trainSequenceBatchExtensionSource)], extra_cflags=trainSequenceBatchCompilerFlags, extra_ldflags=trainSequenceBatchLinkerFlags, with_cuda=False, verbose=False)
		result = _sequenceBatchExtension
	return result


class SequenceBatchState:
	def __init__(self, database, shape):
		if(optimiseTrainSequenceBatch):
			self.database = database
			self.ownerThread = threading.get_ident()
			self.sourceShape = shape
			self.connections = []
			self.neurons = []
			self.columns = {}
			self.sequenceCount = trainSequenceBatchZero
			self.pendingEntries = trainSequenceBatchZero
			self.committing = False
			self.failed = False
			if(optimiseTrainDeferredSourcePreparation):
				self.deferredSourcePreparation = {}
		return
