"""Merge ordered training packets directly into resident int32 connection records."""

from GIAANNcmn_globalDefs import *

if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
	if(optimiseTrainSequenceBatch):
		import os
		from torch.utils.cpp_extension import load
		import GIAANNcmn_databasePacked
		_extension = None


def mergePackets(state, packets, shape, connections):
	result = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		if(optimiseTrainSequenceBatch):
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
					if(source.shape != tuple(shape)):
						raise RuntimeError(databaseResidentPackedInvalidInput)
					sources.append(GIAANNcmn_databasePacked.getRecords(source))
				else:
					sources.append(state.columns[key].featureNeurons.coalesce())
			if(connections):
				mergedRecords = getExtension().mergePackedSequenceBatch(optimisationStoreDatabaseResidentCoordinatesAsInt32, sources, [tensor for packetKeys, tensor in packets], mappings, list(shape), trainSequenceBatchBucketDimension, trainSequenceBatchGrainSize, databaseResidentPackedZero, databaseResidentPackedOne, databaseResidentPackedRecordRank, databaseResidentPackedSourceRank, databaseResidentCoordinateMaximum, databaseResidentPackedInvalidInput)
				merged = [GIAANNcmn_databasePacked.wrapValidatedRecords(records, shape) for records in mergedRecords]
			else:
				import GIAANNcmn_trainSequenceBatch
				merged = GIAANNcmn_trainSequenceBatch.getExtension().mergeSequenceBatch(optimiseTrainSequenceBatch, sources, [tensor for packetKeys, tensor in packets], mappings, list(shape), trainSequenceBatchBucketDimension, trainSequenceBatchGrainSize, trainSequenceBatchInvalidInput)
			result = (keys, merged)
	return result


def publishConnection(column, feature, source):
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		if(optimiseTrainSequenceBatch):
			if(not column.isDatabaseResidentConnectionStorage() or type(feature) is not int or feature < databaseResidentPackedZero or feature >= column.databaseNetworkObject.f or source.shape != column.getFeatureConnectionsTargetSize()):
				raise RuntimeError(databaseResidentPackedInvalidInput)
			GIAANNcmn_databasePacked.getRecords(source)
			column.featureConnectionsBySourceFeature[feature] = source
			column.loadedSourceFeatureIndices.add(feature)
	return


def getExtension():
	global _extension
	result = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		if(optimiseTrainSequenceBatch):
			if(_extension is None):
				_extension = load(name=databaseResidentPackedExtensionName, sources=[os.path.join(os.path.dirname(__file__), databaseResidentPackedExtensionSource)], extra_cflags=databaseResidentPackedCompilerFlags, extra_ldflags=databaseResidentPackedLinkerFlags, with_cuda=False, verbose=False)
			result = _extension
	return result
