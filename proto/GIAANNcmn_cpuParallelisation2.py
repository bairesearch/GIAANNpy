"""Independently selectable CPU training optimisations layered on optimiseTrainParallelisation1."""

from GIAANNcmn_globalDefs import *

if(optimiseTrainParallelisation2a or optimiseTrainParallelisation2b or optimiseTrainParallelisation2c or optimiseTrainParallelisation2d or optimiseTrainParallelisation2e or optimiseTrainParallelisation2f):
	import os
	from torch.utils.cpp_extension import load
	_parallelisation2Extension = None


def createTemporalConnections(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, includeSameTime, sequenceObservedColumns):
	result = None
	if(optimiseTrainParallelisation2a):
		if(not pt.is_tensor(featureNeuronsActive) or tuple(featureNeuronsActive.shape) != (multipleDendriticBranchesNumber, arrayNumberOfSegments, cs, fs)):
			raise RuntimeError("optimiseTrainParallelisation2a activation dimensions must match the configured branches, segments, concepts and features")
		if(sequenceObservedColumns.trainConnectionsUseSpatialDistance or sequenceObservedColumns.trainConnectionsUseSpatialAxis or sequenceObservedColumns.trainConnectionsUseSpatialAxes):
			raise RuntimeError("optimiseTrainParallelisation2a requires temporal connection distances")
		if(columnsWordOrder is None or featureNeuronsWordOrder is None or not isinstance(includeSameTime, bool)):
			raise RuntimeError("optimiseTrainParallelisation2a requires word/column order tensors and a Boolean temporal mode")
		mode = parallelisation2TemporalModeNone
		columnSegments = arrayNumberOfSegments
		featureSegments = arrayNumberOfSegments
		internalColumns = True
		linkFirstColumn = False
		linkFirstFeature = False
		lastColumnSegment = arrayNumberOfSegments - 1
		if(useSANI):
			linkFirstColumn = SANIcolumnsLinkFirstSegmentToAllPriorTrainSeqTokens
			linkFirstFeature = SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens
			if(useSANIcolumns):
				mode = parallelisation2TemporalModeColumns
				if(inferenceLeakyIntegrateAndFire):
					lastColumnSegment = arrayIndexSegmentLastColumn
			elif(useSANIfeatures):
				mode = parallelisation2TemporalModeFeatures
			elif(useSANIfeaturesAndColumns):
				mode = parallelisation2TemporalModeCombined
				columnSegments = arrayNumberOfSegmentsColumnDistance
				featureSegments = arrayNumberOfSegmentsFeatureDistance
				if(inferenceLeakyIntegrateAndFire):
					featureSegments += inferenceLeakyIntegrateAndFireSomaSegmentCount
				internalColumns = useSANIfeaturesAndColumnsInternal
		extension = getParallelisation2Extension()
		result = extension.createTemporalConnections(optimiseTrainParallelisation2a, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, cs, fs, includeSameTime, trainConnectionsAllowSelfTransitions, debugConnectNodesToNextNodesInSequenceOnly, debugConnectColumnsToNextColumnsInSequenceOnly, useSANI, arrayIndexSegmentLast, mode, parallelisation2TemporalModeColumns, parallelisation2TemporalModeFeatures, parallelisation2TemporalModeCombined, columnSegments, featureSegments, internalColumns, linkFirstColumn, linkFirstFeature, lastColumnSegment, parallelisation2MaximumSegments, parallelisation2KernelGrainSize)
	return result


def prepareConnectionUpdates(indices, values, featureIndicesInObserved, conceptIndicesTensor, sourceSize, propertyIndex):
	result = None
	if(optimiseTrainParallelisation2b):
		extension = getParallelisation2Extension()
		result = extension.prepareConnectionUpdates(optimiseTrainParallelisation2b, indices, values, featureIndicesInObserved, conceptIndicesTensor, list(sourceSize), propertyIndex, trainSequenceObservedColumnsUseSequenceFeaturesOnly, parallelisation2MappingGrainSize)
	return result


def mergeSparseSources(existingSources, updates, sourceSize, bucketDimension):
	result = None
	if(optimiseTrainParallelisation2c or optimiseTrainParallelisation2d):
		extension = getParallelisation2Extension()
		result = extension.mergeSparseSources(optimiseTrainParallelisation2c, optimiseTrainParallelisation2d, existingSources, updates, list(sourceSize), bucketDimension, parallelisation2MergeChunkEntries, parallelisation2KernelGrainSize)
	return result


def updateFeatureNeurons(sequenceObservedColumns, observedColumnsByConceptIndex, featureIndices, featureValues, featureIndicesInObserved, conceptIndicesTensor):
	if(optimiseTrainParallelisation2e):
		if(sequenceObservedColumns.databaseNetworkObject.inferenceMode):
			raise RuntimeError("optimiseTrainParallelisation2e only supports training neuron updates")
		if(featureIndices.numel() > 0):
			database = sequenceObservedColumns.databaseNetworkObject
			concepts = pt.unique(conceptIndicesTensor[featureIndices[2]], sorted=True)
			sourceSize = (database.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, database.f)
			updateSize = sourceSize[:parallelisation2NeuronBucketDimension] + (concepts.numel(),) + sourceSize[parallelisation2NeuronBucketDimension:]
			updates = sequenceObservedColumns.buildFeaturePropertyUpdateSparseBatched(featureIndices, featureValues, database.arrayIndexPropertiesStrengthIndex, featureIndicesInObserved, conceptIndicesTensor, updateSize, concepts).coalesce()
			columns = [observedColumnsByConceptIndex[concept] for concept in concepts.tolist()]
			sources = [column.featureNeurons.coalesce() for column in columns]
			if(optimiseTrainParallelisation2c or optimiseTrainParallelisation2d):
				updated = mergeSparseSources(sources, updates, sourceSize, parallelisation2NeuronBucketDimension)
			else:
				import GIAANNcmn_cpuObservedColumnUpdate
				extension = GIAANNcmn_cpuObservedColumnUpdate.getCPUConnectionUpdateExtension()
				updated = extension.mergeConnectionSources(optimiseTrainParallelisation1, sources, updates, list(sourceSize), parallelisation2NeuronBucketDimension, parallelisation1MergeGrainSize)
			for column, tensor in zip(columns, updated):
				column.featureNeurons = tensor
	return


def prepareTrainingFeatureNeurons(sequenceObservedColumns, tokens, conceptIndices, startIndices, endIndices):
	result = None
	if(optimiseTrainParallelisation2f):
		cs = sequenceObservedColumns.cs
		fs = sequenceObservedColumns.fs
		if(cs <= 0 or fs <= 0 or multipleDendriticBranches or not trainSequenceObservedColumnsUseSequenceFeaturesOnly or not trainSequenceObservedColumnsMatchSequenceWords):
			raise RuntimeError("optimiseTrainParallelisation2f requires positive sequence dimensions and single-branch sequence positions")
		for tensor in (conceptIndices, startIndices, endIndices):
			if(not pt.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != pt.long or tensor.dim() != 1 or tensor.numel() != cs):
				raise RuntimeError("optimiseTrainParallelisation2f requires one CPU int64 concept/start/end index per sequence column")
		if(bool(pt.any(conceptIndices < 0)) or bool(pt.any(conceptIndices >= fs)) or bool(pt.any(startIndices < 0)) or bool(pt.any(endIndices < startIndices)) or bool(pt.any(endIndices > min(fs, len(tokens))))):
			raise RuntimeError("optimiseTrainParallelisation2f concept or feature interval is out of range")
		if(conceptIndices.numel() > 1 and bool(pt.any(conceptIndices[1:] <= conceptIndices[:-1]))):
			raise RuntimeError("optimiseTrainParallelisation2f requires strictly increasing concept token positions")
		columnsWordOrder = pt.arange(cs, dtype=pt.long)
		positions = pt.arange(fs, dtype=pt.long)
		featureMask = (positions.unsqueeze(0) >= startIndices.unsqueeze(1)) & (positions.unsqueeze(0) < endIndices.unsqueeze(1))
		segmentPositions = pt.arange(arrayNumberOfSegments, dtype=pt.long).unsqueeze(1)
		if(useSANI):
			if(useSANIcolumns):
				segmentMask = segmentPositions <= columnsWordOrder.unsqueeze(0)
			elif(useSANIfeatures):
				segmentMask = segmentPositions <= conceptIndices.unsqueeze(0)
			elif(useSANIfeaturesAndColumns):
				columnCounts = (columnsWordOrder + int(useSANIfeaturesAndColumnsInternal)).clamp(max=arrayNumberOfSegmentsColumnDistance)
				featureCounts = (conceptIndices + 1).clamp(max=arrayNumberOfSegmentsFeatureDistance)
				segmentMask = (segmentPositions < columnCounts.unsqueeze(0)) | ((segmentPositions >= arrayNumberOfSegmentsColumnDistance) & (segmentPositions < arrayNumberOfSegmentsColumnDistance + featureCounts.unsqueeze(0)))
			else:
				raise RuntimeError("optimiseTrainParallelisation2f requires a supported SANI segment mode")
			featureNeuronsActive = (segmentMask.unsqueeze(2) & featureMask.unsqueeze(0)).to(arrayType).unsqueeze(0)
			featureNeuronsSegmentMask = segmentMask.to(arrayType)
		else:
			featureNeuronsActive = pt.zeros((multipleDendriticBranchesNumber, arrayNumberOfSegments, cs, fs), dtype=arrayType)
			featureNeuronsActive[:, arrayIndexSegmentFirst] = featureMask.to(arrayType)
			featureNeuronsSegmentMask = pt.ones((arrayNumberOfSegments, cs), dtype=arrayType)
		usedPositions = pt.nonzero(featureMask.any(dim=0), as_tuple=False).flatten()
		posValues = pt.zeros(fs, dtype=arrayType)
		posValues[usedPositions] = pt.tensor([posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, tokens[position].pos) for position in usedPositions.tolist()], dtype=arrayType)
		featureNeuronsPos = featureMask.to(arrayType) * posValues.unsqueeze(0)
		featureNeuronsWordOrder = positions.unsqueeze(0).repeat(cs, 1)
		sequenceConceptIndexMask = pt.ones((cs, fs), dtype=arrayType)
		sequenceConceptIndexMask[:, conceptIndices] = 0
		sequenceConceptIndexMask[columnsWordOrder, conceptIndices] = 1
		result = (featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask)
	return result


def getParallelisation2Extension():
	global _parallelisation2Extension
	result = None
	if(optimiseTrainParallelisation2a or optimiseTrainParallelisation2b or optimiseTrainParallelisation2c or optimiseTrainParallelisation2d or optimiseTrainParallelisation2e or optimiseTrainParallelisation2f):
		if(_parallelisation2Extension is None):
			_parallelisation2Extension = load(name=parallelisation2ExtensionName, sources=[os.path.join(os.path.dirname(__file__), parallelisation2ExtensionSource)], extra_cflags=parallelisation2CompilerFlags, extra_ldflags=parallelisation2LinkerFlags, with_cuda=False, verbose=False)
		result = _parallelisation2Extension
	return result
