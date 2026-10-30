from GIAANNcmn_globalDefs import *

if(optimiseTrainFusedSequenceUpdates):
	import os
	from torch.utils.cpp_extension import load
	import GIAANNcmn_trainColumnSteps
	_extension = None


def prepare(sequence, concepts, starts, ends):
	result = None
	if(optimiseTrainFusedSequenceUpdates):
		if(not optimiseTrainSequenceBatch or not useSANI or not useSANIfeaturesAndColumns or not useSANIfeaturesAndColumnsInternal or multipleDendriticBranches or trainConnectionStrengthIncreaseColumnInternal or trainConnectionStrengthPOSdependence or trainConnectionStrengthLimitMax or trainConnectionStrengthLimitTanh or useTrainDuringInference or auxiliaryNeurons or SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens or SANIcolumnsLinkFirstSegmentToAllPriorTrainSeqTokens):
			raise RuntimeError(trainFusedInvalidInput)
		database = sequence.databaseNetworkObject
		features, subset = sequence.getObservedColumnFeatureIndices()
		if(subset.numel() != sequence.fs or not pt.equal(subset, pt.arange(sequence.fs))):
			raise RuntimeError(trainFusedInvalidInput)
		shape = [database.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, database.c, database.f]
		steps = GIAANNcmn_trainColumnSteps.calculateColumnSteps(sequence, pt.device(trainSequenceBatchDeviceType)) if patchTrainColumnStepsRepeatedColumns else pt.arange(sequence.cs, dtype=pt.long)
		proximity = pt.ones(sequence.fs, dtype=arrayType)
		if(trainConnectionStrengthNormaliseWrtContextLength):
			proximity = trainConnectionStrengthProximityScale / pt.pow(pt.arange(sequence.fs, dtype=arrayType) + trainConnectionStrengthProximityDistanceOffset, trainConnectionStrengthProximityExponent)
		connectionSegments = arrayNumberOfSegmentsFeatureDistance + (inferenceLeakyIntegrateAndFireSomaSegmentCount if inferenceLeakyIntegrateAndFire else trainSequenceBatchZero)
		result = getExtension().preparePackets(optimiseTrainFusedSequenceUpdates, concepts, starts, ends, sequence.conceptIndicesInSequenceObservedTensor, features, steps, proximity, shape, arrayNumberOfSegmentsColumnDistance, arrayNumberOfSegmentsFeatureDistance, connectionSegments, trainConnectionsAllowSelfTransitions, debugConnectNodesToNextNodesInSequenceOnly, debugConnectColumnsToNextColumnsInSequenceOnly, trainSequenceBatchZero, trainSequenceBatchOne, trainFusedSourceRank, trainFusedConnectionRank, trainFusedPropertyDimension, trainFusedBranchDimension, trainFusedSegmentDimension, trainFusedBucketDimension, trainFusedTargetConceptDimension, trainFusedTargetFeatureDimension, trainFusedInvalidInput)
	return result


def getExtension():
	global _extension
	result = None
	if(optimiseTrainFusedSequenceUpdates):
		if(_extension is None):
			_extension = load(name=trainFusedExtensionName, sources=[os.path.join(os.path.dirname(__file__), trainFusedExtensionSource)], extra_cflags=trainSequenceBatchCompilerFlags, extra_ldflags=trainSequenceBatchLinkerFlags, with_cuda=False, verbose=False)
		result = _extension
	return result
