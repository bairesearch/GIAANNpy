"""GIAANNcmn_trainColumnSteps.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common train Column Steps

Column-segment training distances measured by changes in database column identity.

"""



from GIAANNcmn_globalDefs import *


def calculateColumnDistances(sequenceObservedColumns, sourceConceptIndices, targetConceptIndices):
	result = None
	if(patchTrainColumnStepsRepeatedColumns):
		for occurrenceIndices in (sourceConceptIndices, targetConceptIndices):
			if(not isinstance(occurrenceIndices, pt.Tensor) or occurrenceIndices.layout != pt.strided or occurrenceIndices.dim() != patchTrainColumnStepsRank or occurrenceIndices.dtype != pt.long):
				raise RuntimeError(patchTrainColumnStepsInvalidOccurrences)
			if((occurrenceIndices < patchTrainColumnStepsFirst).any() or (occurrenceIndices >= sequenceObservedColumns.cs).any()):
				raise RuntimeError(patchTrainColumnStepsInvalidOccurrences)
		if(sourceConceptIndices.shape != targetConceptIndices.shape or sourceConceptIndices.device != targetConceptIndices.device):
			raise RuntimeError(patchTrainColumnStepsInvalidOccurrences)
		columnSteps = calculateColumnSteps(sequenceObservedColumns, sourceConceptIndices.device)
		result = pt.abs(columnSteps[targetConceptIndices] - columnSteps[sourceConceptIndices])
	return result


def calculateColumnSteps(sequenceObservedColumns, targetDevice):
	result = None
	if(patchTrainColumnStepsRepeatedColumns):
		if(not useModalityNLP or not inferenceLeakyIntegrateAndFire or not trainSequenceObservedColumnsMatchSequenceWords):
			raise RuntimeError(patchTrainColumnStepsInvalid)
		columnIndices = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor
		if(not isinstance(sequenceObservedColumns.cs, int) or isinstance(sequenceObservedColumns.cs, bool) or sequenceObservedColumns.cs < patchTrainColumnStepsFirst or not isinstance(columnIndices, pt.Tensor) or columnIndices.layout != pt.strided or columnIndices.dim() != patchTrainColumnStepsRank or columnIndices.dtype != pt.long or columnIndices.numel() != sequenceObservedColumns.cs):
			raise RuntimeError(patchTrainColumnStepsInvalid)
		if((columnIndices < patchTrainColumnStepsFirst).any() or (columnIndices >= sequenceObservedColumns.databaseNetworkObject.c).any()):
			raise RuntimeError(patchTrainColumnStepsInvalid)
		columnIndices = columnIndices.to(targetDevice)
		#Keep every occurrence and neuron coordinate; collapse only adjacent equal IDs in the timing coordinate.
		columnChanges = pt.zeros_like(columnIndices)
		columnChanges[patchTrainColumnStepsIncrement:] = (columnIndices[patchTrainColumnStepsIncrement:] != columnIndices[:-patchTrainColumnStepsIncrement]).long()
		result = pt.cumsum(columnChanges, dim=patchTrainColumnStepsDimension)
	return result
