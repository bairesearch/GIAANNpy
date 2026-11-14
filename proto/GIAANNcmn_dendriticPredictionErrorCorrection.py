"""GIAANNcmn_dendriticPredictionErrorCorrection.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common dendritic Prediction Error Correction

Local margin learning for independent LIF branches during predict-then-train.

The eligibility tape differentiates linear segment transport, multiplicative decay
and neuron resets, holding observed source firing fixed. It is not backpropagation
through firing thresholds. All predictions in a sequence use frozen weights; the
following training pass applies their accumulated projected synaptic updates.

"""

from types import SimpleNamespace
from GIAANNcmn_globalDefs import *


def initialiseSequence(databaseNetworkObject, sequenceIndex):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(executionMode != dendriticCorrectionTrainMode or type(sequenceIndex) is not int or sequenceIndex < dendriticCorrectionZero):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		previous = getattr(databaseNetworkObject, dendriticCorrectionTraceAttribute, None)
		if(previous is not None and not previous.consumed):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		databaseNetworkObject.dendriticCorrectionTrace = SimpleNamespace(sequenceIndex=sequenceIndex, tokens=None, events=[], sources={}, sourceConnections={}, updates={}, neurons=[], neuronUsage={}, diagnostics=[], columnStep=dendriticCorrectionZero, lastPosition=None, consumed=False, eligibility=createEligibilityTrace())


def prepareSequence(sequenceObservedColumns, tokensSequence, conceptMask):
	if(trainDendriticBranchesPredictionErrorCorrection):
		import GIAANNcmn_databaseNetwork
		trace = getTrace(sequenceObservedColumns.databaseNetworkObject)
		if(trace.tokens is not None):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		trace.tokens = tuple(token.word for token in tokensSequence)
		trace.targets = [GIAANNcmn_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, position, kcNetwork) for position in range(len(tokensSequence))]


def finishSequence(database, sequenceIndex):
	if(trainDendriticBranchesPredictionErrorCorrection):
		trace = getattr(database, dendriticCorrectionTraceAttribute, None)
		if(trace is None or trace.sequenceIndex != sequenceIndex):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(not trace.consumed):
			#The native parser can skip a short sequence or one with no usable concept columns.
			if(trace.tokens is not None or trace.events or trace.updates):
				raise RuntimeError(dendriticCorrectionInvalidTrace)
			trace.consumed = True


def recordPrediction(sequenceObservedColumns, activation, position, selectedColumn, allowedColumns, constraintMode, arrivalKeys):
	if(trainDendriticBranchesPredictionErrorCorrection):
		import GIAANNcmn_predictionActivate as activate
		import GIAANNcmn_predictionBeamSearch as selection
		import GIAANNcmn_predictionConstraints as constraints
		import GIAANNcmn_sparseTensors
		database = sequenceObservedColumns.databaseNetworkObject
		trace = getTrace(database)
		if(trace.tokens is None or type(position) is not int or position < numSeedTokens or position >= len(trace.tokens) or (trace.lastPosition is None and position != numSeedTokens) or (trace.lastPosition is not None and position != trace.lastPosition+dendriticCorrectionOne)):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		trace.lastPosition = position
		targetColumn, unusedNextColumn, targetFeature = trace.targets[position]
		targetColumn, targetFeature = int(targetColumn), int(targetFeature)
		scoringState = activation
		if(inferenceReviewPatch11ProspectiveColumnScoring and (useSANIcolumns or useSANIfeaturesAndColumns)):
			scoringState = selection.createReviewProspectiveColumnSelectionState(activation, int(selectedColumn))
		branchSignal = activate.calculateLeakyIntegrateAndFireSomaActivationByBranch(scoringState).coalesce()
		correctScores = getBranchScores(branchSignal, targetColumn, targetFeature)
		correctBranch = selectCorrectBranch(database, trace, correctScores, targetColumn, targetFeature)
		correctKey = activate.calculateLeakyIntegrateAndFireBranchColumnFeatureKeys(pt.tensor([correctBranch], dtype=pt.long), pt.tensor([targetColumn], dtype=pt.long), pt.tensor([targetFeature], dtype=pt.long), database.c, database.f)
		correctEligible = bool(constraints.buildSortedKeyMembershipMask(correctKey, arrivalKeys).item())
		correctRawScore = float(correctScores[correctBranch].item())
		#An excluded correct branch still needs prospective input growth, even if its older response is large.
		correctScore = correctRawScore if correctEligible else float(dendriticCorrectionZero)
		constraintState = constraints.createConstraintState(allowedColumns, constraintMode)
		#Learn context discrimination before threshold/direct-edge admission. Otherwise an incorrect
		#partial-context branch can evade depression merely by losing its most recent input edge.
		cellSignal = GIAANNcmn_sparseTensors.reduceSparseBranchMax(branchSignal).coalesce()
		columns, features, scores = cellSignal.indices()[inferenceLeakyIntegrateAndFireSomaActivationConceptDimension], cellSignal.indices()[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension], cellSignal.values()
		columns, features, scores = constraints.filterColumnFeatureCandidatesByConstraint(database, columns, features, scores, constraintState)
		wrong = None
		wrongScore = float(dendriticCorrectionZero)
		if(columns is not None):
			targetToken = tokenIdentity(database, targetColumn, targetFeature)
			wrongMask = pt.tensor([tokenIdentity(database, column, feature) != targetToken for column, feature in zip(columns.tolist(), features.tolist())], dtype=pt.bool, device=scores.device) & (scores > dendriticCorrectionZero)
			if(wrongMask.any()):
				candidateIndex = int(pt.argmax(scores*wrongMask).item())
				column, feature = int(columns[candidateIndex]), int(features[candidateIndex])
				branchScores = getBranchScores(branchSignal, column, feature)
				branch = int(pt.argmax(branchScores).item())
				wrongScore = float(branchScores[branch].item())
				wrong = (column, feature, branch)
		#An absent/subthreshold target must also learn to cross the firing threshold.
		loss = max(float(dendriticCorrectionZero), max(wrongScore, inferenceLeakyIntegrateAndFireSomaActivationThreshold)+dendriticCorrectionMargin-correctScore)
		if(loss > dendriticCorrectionZero):
			positiveIndices, positiveValues = calculateSynapseEligibility(database, trace, targetColumn, targetFeature, correctBranch, position, selectedColumn, True)
			if(positiveValues.numel() == dendriticCorrectionZero):
				raise RuntimeError(dendriticCorrectionNoEligibility)
			negativeIndices, negativeValues = None, None
			norm = float(positiveValues.square().sum().item())
			if(wrong is not None):
				negativeIndices, negativeValues = calculateSynapseEligibility(database, trace, wrong[dendriticCorrectionZero], wrong[dendriticCorrectionOne], wrong[dendriticCorrectionTwo], position, selectedColumn, False)
				norm += float(negativeValues.square().sum().item())
			step = min(dendriticCorrectionLearningRate, loss/norm)
			queueSynapseUpdate(trace, positiveIndices, positiveValues*step, targetColumn, targetFeature, correctBranch)
			if(wrong is not None):
				queueSynapseUpdate(trace, negativeIndices, -negativeValues*step, wrong[dendriticCorrectionZero], wrong[dendriticCorrectionOne], wrong[dendriticCorrectionTwo])
			trace.neurons.append((correctBranch, arrayIndexSegmentSoma, targetColumn, targetFeature))
			usage = trace.neuronUsage.setdefault((targetColumn, targetFeature), pt.zeros(multipleDendriticBranchesNumber, dtype=arrayType))
			usage[correctBranch] += dendriticCorrectionOne
		trace.diagnostics.append(SimpleNamespace(position=position, target=(targetColumn, targetFeature, correctBranch), correctScore=correctScore, correctRawScore=correctRawScore, correctEligible=correctEligible, wrong=wrong, wrongScore=wrongScore, loss=loss))


def trainSequence(sequenceObservedColumns, sequenceIndex, tokens):
	if(trainDendriticBranchesPredictionErrorCorrection):
		import GIAANNcmn_databaseNetwork
		import GIAANNcmn_databaseNetworkFiles
		database = sequenceObservedColumns.databaseNetworkObject
		trace = getTrace(database)
		if(sequenceIndex != trace.sequenceIndex or (trace.tokens is not None and tuple(token.word for token in tokens) != trace.tokens)):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(trace.tokens is not None and len(trace.tokens) > numSeedTokens and trace.lastPosition != len(trace.tokens)-dendriticCorrectionOne):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(trace.updates):
			requiredSources = sequenceObservedColumns.getTrainRequiredSourceFeatureIndicesByObservedColumn()
			GIAANNcmn_databaseNetwork.prepareObservedColumnsForTrainSequence(sequenceObservedColumns.observedColumnsDict, requiredSources)
		prepared = []
		for (sourceColumn, sourceFeature), updates in trace.updates.items():
			columnName = database.conceptColumnsList[sourceColumn]
			if(columnName not in sequenceObservedColumns.observedColumnsDict):
				raise RuntimeError(dendriticCorrectionInvalidTrace)
			observedColumn = sequenceObservedColumns.observedColumnsDict[columnName]
			original = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeature, targetDevice=deviceSparse, createMissing=True, ensureCurrentSizeOnLoad=True).coalesce()
			validateStrengthTensor(original)
			indices = pt.cat([item[dendriticCorrectionZero] for item in updates], dim=dendriticCorrectionOne)
			values = pt.cat([item[dendriticCorrectionOne] for item in updates])
			properties = pt.full((dendriticCorrectionOne, values.numel()), database.arrayIndexPropertiesStrengthIndex, dtype=pt.long, device=values.device)
			delta = pt.sparse_coo_tensor(pt.cat((properties, indices)), values, size=original.size()).coalesce()
			updated = (original+delta).coalesce()
			updatedValues = updated.values().clamp(min=dendriticCorrectionZero)
			retained = updatedValues > dendriticCorrectionZero
			updated = pt.sparse_coo_tensor(updated.indices()[:, retained], updatedValues[retained], size=original.size()).coalesce()
			validateStrengthTensor(updated)
			prepared.append((observedColumn, sourceFeature, updated))
		neuronStrength = database.globalFeatureNeurons
		if(trace.neurons):
			coordinates = pt.tensor(trace.neurons, dtype=pt.long, device=neuronStrength.device).t()
			properties = pt.full((dendriticCorrectionOne, coordinates.shape[dendriticCorrectionOne]), database.arrayIndexPropertiesStrengthIndex, dtype=pt.long, device=coordinates.device)
			counts = pt.ones(coordinates.shape[dendriticCorrectionOne], dtype=neuronStrength.dtype, device=neuronStrength.device)
			neuronStrength = (neuronStrength+pt.sparse_coo_tensor(pt.cat((properties, coordinates)), counts, size=neuronStrength.size())).coalesce()
		for observedColumn, sourceFeature, updated in prepared:
			observedColumn.setFeatureConnectionsForSourceFeature(sourceFeature, updated)
		cache = getattr(database, dendriticCorrectionUsageAttribute, None)
		if(cache is not None and cache.strength is database.globalFeatureNeurons):
			for target, counts in trace.neuronUsage.items():
				cache.usage.setdefault(target, pt.zeros(multipleDendriticBranchesNumber, dtype=arrayType)).add_(counts)
			cache.strength = neuronStrength
		database.globalFeatureNeurons = neuronStrength
		trace.consumed = True
		trace.events.clear()
		trace.sources.clear()
		trace.sourceConnections.clear()
		trace.eligibility = None
		trace.updates.clear()
		if(prepared and useSaveData and not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			GIAANNcmn_databaseNetworkFiles.saveData(database, sequenceObservedColumns.observedColumnsDict, sequenceIndex)


def recordSource(database, strength, sourceColumn, sourceFeature, predictionPosition, firing):
	if(trainDendriticBranchesPredictionErrorCorrection):
		trace = getTrace(database)
		if(predictionPosition is None or type(predictionPosition) is not int or predictionPosition <= dendriticCorrectionZero or not pt.isfinite(firing).all() or firing.numel() != dendriticCorrectionOne or float(firing.item()) not in (dendriticCorrectionZero, dendriticCorrectionOne)):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(not strength.is_sparse or strength.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or tuple(strength.shape) != (multipleDendriticBranchesNumber, arrayNumberOfSegments, database.c, database.f) or not pt.isfinite(strength.coalesce().values()).all() or (strength.coalesce().values() < dendriticCorrectionZero).any()):
			raise RuntimeError(dendriticCorrectionInvalidTensor)
		if(float(firing.item()) > dendriticCorrectionZero):
			source = (int(sourceColumn), int(sourceFeature))
			trace.sources[source] = strength.coalesce()
			if(source not in trace.sourceConnections):
				storedIndices = strength.coalesce().indices()
				keys = ((storedIndices[inferenceLeakyIntegrateAndFireConceptDimension]*database.f+storedIndices[inferenceLeakyIntegrateAndFireFeatureDimension])*multipleDendriticBranchesNumber+storedIndices[inferenceLeakyIntegrateAndFireBranchDimension])*arrayNumberOfSegments+storedIndices[inferenceLeakyIntegrateAndFireSegmentDimension]
				trace.sourceConnections[source] = set(keys[strength.coalesce().values() > dendriticCorrectionZero].tolist())
			recordEvent(database, dendriticCorrectionSource, int(sourceColumn), int(sourceFeature), predictionPosition-dendriticCorrectionOne)


def recordEvent(database, action, column=None, feature=None, position=None):
	if(trainDendriticBranchesPredictionErrorCorrection):
		trace = getTrace(database)
		if(action not in dendriticCorrectionActions or trace.tokens is None):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(action in (dendriticCorrectionSource, dendriticCorrectionReset)):
			if(type(column) is not int or type(feature) is not int or not dendriticCorrectionZero <= column < database.c or not dendriticCorrectionZero <= feature < database.f):
				raise RuntimeError(dendriticCorrectionInvalidTrace)
		if(action == dendriticCorrectionSource):
			if(type(position) is not int or not dendriticCorrectionZero <= position < len(trace.tokens)-dendriticCorrectionOne):
				raise RuntimeError(dendriticCorrectionInvalidTrace)
		trace.events.append(SimpleNamespace(action=action, column=column, feature=feature, position=position, columnStep=trace.columnStep))
		advanceEligibilityTrace(trace, action, column, feature, position)
		if(action == dendriticCorrectionAdvance):
			trace.columnStep += dendriticCorrectionOne


def createEligibilityTrace():
	if(trainDendriticBranchesPredictionErrorCorrection):
		empty = pt.empty(dendriticCorrectionZero, dtype=pt.long)
		result = SimpleNamespace(segments=empty, originalSegments=empty, columns=empty, features=empty, positions=empty, columnSteps=empty, coefficients=pt.empty(dendriticCorrectionZero, dtype=arrayType), resets={})
	return result


def advanceEligibilityTrace(trace, action, column, feature, position):
	if(trainDendriticBranchesPredictionErrorCorrection):
		eligibility = trace.eligibility
		if(action == dendriticCorrectionSource):
			segments = pt.arange(arrayNumberOfSegments, dtype=pt.long)
			eligibility.segments = pt.cat((eligibility.segments, segments))
			eligibility.originalSegments = pt.cat((eligibility.originalSegments, segments))
			eligibility.coefficients = pt.cat((eligibility.coefficients, pt.full((arrayNumberOfSegments,), j1, dtype=arrayType)))
			eligibility.columns = pt.cat((eligibility.columns, pt.full_like(segments, column)))
			eligibility.features = pt.cat((eligibility.features, pt.full_like(segments, feature)))
			eligibility.positions = pt.cat((eligibility.positions, pt.full_like(segments, position)))
			eligibility.columnSteps = pt.cat((eligibility.columnSteps, pt.full_like(segments, trace.columnStep)))
		elif(action == dendriticCorrectionReset):
			resetIndices = pt.nonzero(resetEligibilityMask(eligibility.segments) & (eligibility.coefficients > dendriticCorrectionZero), as_tuple=False).flatten()
			previous = eligibility.resets.get((column, feature))
			if(previous is not None):
				resetIndices = pt.unique(pt.cat((previous, resetIndices)), sorted=True)
			eligibility.resets[(column, feature)] = resetIndices
		else:
			transportEligibility(eligibility.segments, eligibility.coefficients, action, False)


def calculateSynapseEligibility(database, trace, targetColumn, targetFeature, branch, position, selectedColumn, allowGrowth):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(type(branch) is not int or not dendriticCorrectionZero <= branch < multipleDendriticBranchesNumber or type(targetColumn) is not int or not dendriticCorrectionZero <= targetColumn < database.c or type(targetFeature) is not int or not dendriticCorrectionZero <= targetFeature < database.f or type(allowGrowth) is not bool):
			raise RuntimeError(dendriticCorrectionInvalidTensor)
		indices, coefficients, growthCoefficients = replayEligibility(database, trace, targetColumn, targetFeature, position, selectedColumn)
		retained = coefficients > dendriticCorrectionZero
		indices, coefficients, growthCoefficients = indices[:, retained], coefficients[retained], growthCoefficients[retained]
		targetKey = ((targetColumn*database.f+targetFeature)*multipleDendriticBranchesNumber+branch)*arrayNumberOfSegments
		stored = pt.tensor([targetKey+segment in trace.sourceConnections[(column, feature)] for column, feature, segment in indices.t().tolist()], dtype=pt.bool)
		values = coefficients*stored
		if(allowGrowth):
			values = pt.where(stored, coefficients, growthCoefficients)
		retained = values > dendriticCorrectionZero
		result = indices[:, retained], values[retained]
	return result


def replayEligibility(database, trace, targetColumn, targetFeature, position, selectedColumn):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(type(position) is not int or trace.tokens is None or not dendriticCorrectionZero <= position < len(trace.tokens) or type(selectedColumn) is not int or not dendriticCorrectionZero <= selectedColumn < database.c):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		#Transport each source impulse once when its event occurs. Target-specific resets
		#only zero impulses, so their masks commute with all later linear transport.
		eligibility = trace.eligibility
		segments, coefficients = eligibility.segments.clone(), eligibility.coefficients.clone()
		resetIndices = eligibility.resets.get((targetColumn, targetFeature))
		if(resetIndices is not None):
			coefficients[resetIndices] = dendriticCorrectionZero
		originalSegments, columns, features = eligibility.originalSegments, eligibility.columns, eligibility.features
		targetColumnStep = trace.columnStep+int(int(selectedColumn) != targetColumn)
		growth = calculateProspectiveImpulseMask(originalSegments, position-eligibility.positions, targetColumnStep-eligibility.columnSteps)
		if(not trainConnectionsAllowSelfTransitions):
			growth &= (columns != targetColumn) | (features != targetFeature)
		if(inferenceReviewPatch11ProspectiveColumnScoring and (useSANIcolumns or useSANIfeaturesAndColumns) and targetColumn != int(selectedColumn)):
			transportEligibility(segments, coefficients, dendriticCorrectionAdvance, False)
		terminal = segments == arrayIndexSegmentSoma
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			terminal |= segments == arrayIndexSegmentLastColumn
		coefficients *= terminal
		keys = (columns*database.f+features)*arrayNumberOfSegments+originalSegments
		uniqueKeys, inverse = pt.unique(keys, sorted=True, return_inverse=True)
		total = pt.zeros(uniqueKeys.numel(), dtype=arrayType)
		growthTotal = pt.zeros_like(total)
		total.index_add_(dendriticCorrectionZero, inverse, coefficients)
		growthTotal.index_add_(dendriticCorrectionZero, inverse, coefficients*growth)
		sourceKeys = pt.div(uniqueKeys, arrayNumberOfSegments, rounding_mode=dendriticCorrectionFloor)
		indices = pt.stack((pt.div(sourceKeys, database.f, rounding_mode=dendriticCorrectionFloor), sourceKeys % database.f, uniqueKeys % arrayNumberOfSegments))
		result = indices, total, growthTotal
	return result


def transportEligibility(segments, coefficients, action, resetTarget):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(action == dendriticCorrectionPropagate):
			moving = segments < arrayIndexSegmentSoma
			if(useSANIcolumns or useSANIfeaturesAndColumns):
				moving &= segments > arrayIndexSegmentLastColumn
			segments[moving] += dendriticCorrectionOne
		elif(action == dendriticCorrectionAdvance):
			if(not (useSANIcolumns or useSANIfeaturesAndColumns)):
				raise RuntimeError(dendriticCorrectionInvalidTrace)
			if(inferenceDecrementActivationsLastColumnSegment):
				coefficients[segments == arrayIndexSegmentLastColumn] *= dendriticCorrectionOne-inferenceDecrementActivationsLastColumnSegmentPerPredictedColumn
			segments[segments < arrayIndexSegmentLastColumn] += dendriticCorrectionOne
		elif(action == dendriticCorrectionDecay):
			if(inferenceDecrementActivationsSoma):
				coefficients[segments == arrayIndexSegmentSoma] *= dendriticCorrectionOne-inferenceDecrementActivationsSomaPerPredictedToken
		elif(action == dendriticCorrectionReset):
			if(resetTarget):
				coefficients[resetEligibilityMask(segments)] = dendriticCorrectionZero
		else:
			raise RuntimeError(dendriticCorrectionInvalidTrace)


def resetEligibilityMask(segments):
	if(trainDendriticBranchesPredictionErrorCorrection):
		result = pt.zeros_like(segments, dtype=pt.bool)
		if(inferenceDeactivateSomaUponPrediction):
			result |= segments == arrayIndexSegmentSoma
		if(inferenceDeactivateSegmentsUponPrediction):
			result |= segments < arrayIndexSegmentSoma
		if(inferenceDeactivateLastColumnSegmentUponPrediction):
			result |= segments == arrayIndexSegmentLastColumn
	return result


def prospectiveSegments(segments, tokenDistance, columnDistance):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(type(tokenDistance) is not int or type(columnDistance) is not int or tokenDistance <= dendriticCorrectionZero or columnDistance < dendriticCorrectionZero):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		result = calculateProspectiveImpulseMask(segments, pt.full_like(segments, tokenDistance), pt.full_like(segments, columnDistance))
	return result


def calculateProspectiveImpulseMask(segments, tokenDistances, columnDistances):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(segments.shape != tokenDistances.shape or segments.shape != columnDistances.shape or (tokenDistances <= dendriticCorrectionZero).any() or (columnDistances < dendriticCorrectionZero).any()):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
		result = pt.zeros_like(segments, dtype=pt.bool)
		if(useSANIfeatures or useSANIfeaturesAndColumns):
			firstFeature = arrayNumberOfSegmentsColumnDistance if useSANIfeaturesAndColumns else dendriticCorrectionZero
			featureSegments = arrayNumberOfSegments-tokenDistances
			if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
				featureSegments = featureSegments.clamp(min=firstFeature)
			result |= (segments == featureSegments) & (segments >= firstFeature)
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			columnSegments = arrayIndexSegmentLastColumn-columnDistances
			validColumns = pt.ones_like(segments, dtype=pt.bool)
			if(useSANIfeaturesAndColumns and not useSANIfeaturesAndColumnsInternal):
				columnSegments += dendriticCorrectionOne
				validColumns = columnDistances > dendriticCorrectionZero
			if(SANIcolumnsLinkFirstSegmentToAllPriorTrainSeqTokens):
				columnSegments = columnSegments.clamp(min=dendriticCorrectionZero)
			result |= (segments == columnSegments) & (segments <= arrayIndexSegmentLastColumn) & validColumns
	return result


def queueSynapseUpdate(trace, indices, values, targetColumn, targetFeature, branch):
	if(trainDendriticBranchesPredictionErrorCorrection):
		for sourceColumn, sourceFeature in pt.unique(indices[:dendriticCorrectionTwo].t(), dim=dendriticCorrectionZero).tolist():
			mask = (indices[dendriticCorrectionZero] == sourceColumn) & (indices[dendriticCorrectionOne] == sourceFeature)
			segments = indices[dendriticCorrectionTwo, mask]
			coordinates = pt.stack((pt.full_like(segments, branch), segments, pt.full_like(segments, targetColumn), pt.full_like(segments, targetFeature)))
			trace.updates.setdefault((sourceColumn, sourceFeature), []).append((coordinates, values[mask]))


def selectCorrectBranch(database, trace, scores, column, feature):
	if(trainDendriticBranchesPredictionErrorCorrection):
		branches = pt.nonzero(scores == scores.max(), as_tuple=False).flatten()
		result = int(branches[dendriticCorrectionZero].item())
		if(branches.numel() > dendriticCorrectionOne):
			#Deterministic symmetry breaking only on equal responses: prefer the least-trained branch.
			usage = getCorrectBranchUsage(database, column, feature).clone()
			pending = trace.neuronUsage.get((column, feature))
			if(pending is not None):
				#Preserve the original float32 increment order, including counts beyond exact integer precision.
				for usedBranch in branches.tolist():
					for unused in range(int(pending[usedBranch].item())):
						usage[usedBranch] += dendriticCorrectionOne
			result = int(branches[pt.argmin(usage[branches])].item())
	return result


def getCorrectBranchUsage(database, column, feature):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(type(column) is not int or not dendriticCorrectionZero <= column < database.c or type(feature) is not int or not dendriticCorrectionZero <= feature < database.f):
			raise RuntimeError(dendriticCorrectionInvalidTensor)
		cache = getattr(database, dendriticCorrectionUsageAttribute, None)
		if(cache is None or cache.strength is not database.globalFeatureNeurons):
			strength = database.globalFeatureNeurons[database.arrayIndexPropertiesStrengthIndex].coalesce()
			if(not pt.isfinite(strength.values()).all() or (strength.values() < dendriticCorrectionZero).any()):
				raise RuntimeError(dendriticCorrectionInvalidTensor)
			indices = strength.indices()
			mask = indices[inferenceLeakyIntegrateAndFireSegmentDimension] == arrayIndexSegmentSoma
			usage = {}
			for coordinate, value in zip(indices[:, mask].t().tolist(), strength.values()[mask].tolist()):
				branch, segment, targetColumn, targetFeature = coordinate
				usage.setdefault((targetColumn, targetFeature), pt.zeros(multipleDendriticBranchesNumber, dtype=arrayType))[branch] += value
			cache = SimpleNamespace(strength=database.globalFeatureNeurons, usage=usage)
			database.dendriticCorrectionBranchUsage = cache
		result = cache.usage.get((column, feature))
		if(result is None):
			result = pt.zeros(multipleDendriticBranchesNumber, dtype=arrayType)
	return result


def getBranchScores(branchSignal, column, feature):
	if(trainDendriticBranchesPredictionErrorCorrection):
		indices = branchSignal.indices()
		mask = (indices[inferenceLeakyIntegrateAndFireSomaActivationByBranchConceptDimension] == column) & (indices[inferenceLeakyIntegrateAndFireSomaActivationByBranchFeatureDimension] == feature)
		result = pt.zeros(multipleDendriticBranchesNumber, dtype=branchSignal.dtype, device=branchSignal.device)
		result.index_add_(dendriticCorrectionZero, indices[inferenceLeakyIntegrateAndFireSomaActivationByBranchBranchDimension, mask], branchSignal.values()[mask])
		if(not pt.isfinite(result).all() or (result < dendriticCorrectionZero).any()):
			raise RuntimeError(dendriticCorrectionInvalidTensor)
	return result


def tokenIdentity(database, column, feature):
	if(trainDendriticBranchesPredictionErrorCorrection):
		result = database.conceptColumnsList[column] if feature == featureIndexPrimeConceptNeuron else database.conceptFeaturesList[feature]
	return result


def validateStrengthTensor(tensor):
	if(trainDendriticBranchesPredictionErrorCorrection):
		if(not tensor.is_sparse or tensor.dim() != dendriticCorrectionConnectionRank or tensor.shape[dendriticCorrectionZero] != dendriticCorrectionOne or tensor.shape[dendriticCorrectionOne] != multipleDendriticBranchesNumber or tensor.shape[dendriticCorrectionTwo] != arrayNumberOfSegments or not pt.isfinite(tensor.values()).all() or (tensor.values() < dendriticCorrectionZero).any()):
			raise RuntimeError(dendriticCorrectionInvalidTensor)
		limits = pt.tensor(tensor.shape, dtype=pt.long, device=tensor.device).unsqueeze(dendriticCorrectionOne)
		if((tensor.indices() < dendriticCorrectionZero).any() or (tensor.indices() >= limits).any()):
			raise RuntimeError(dendriticCorrectionInvalidTensor)


def getTrace(database):
	if(trainDendriticBranchesPredictionErrorCorrection):
		result = getattr(database, dendriticCorrectionTraceAttribute, None)
		if(executionMode != dendriticCorrectionTrainMode or result is None or result.consumed):
			raise RuntimeError(dendriticCorrectionInvalidTrace)
	return result
