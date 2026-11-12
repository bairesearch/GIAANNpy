"""Likelihood beam search with isolated speculative state and bounded candidate admission.

For depth greater than one, score the root log probability plus the configured
weight times the mean continuation log probability. Normalisation uses all
admitted output tokens before beam truncation. Each token retains the native
strongest-neuron representative for its speculative state transition.
An optional root-score margin restricts which first-token alternatives may win;
it is a prediction policy, separate from exact probability-bound pruning.
"""

from dataclasses import dataclass
import math
import torch as pt
from GIAANNcmn_globalDefs import *
import GIAANNcmn_predictionContextLikelihood as likelihood
import GIAANNcmn_predictionColumnLikelihood as columnLikelihood
import GIAANNcmn_predictionConstraints as constraints


@dataclass
class BeamCandidate:
	column: int
	feature: int
	token: int
	logProbability: float
	activation: float
	eligible: bool


@dataclass
class BeamPath:
	state: dict
	context: dict
	constraint: object
	connectedColumns: object
	connectedFeatures: object
	root: BeamCandidate
	score: float
	future: float


def search(database, observedColumns, features, time, conceptState, arrival, selectedColumn, deactivated, allowedColumns, constraintMode, position, connectedColumns, connectedFeatures):
	if(inferenceBeamSearchLikelihood):
		import GIAANNcmn_predictionBeamSearch as native
		validateConfiguration()
		if(type(position) is not int or position < inferenceBeamSearchZero or database.contextLikelihoodPosition != position):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		originalContext = {name:getattr(database, name) for name in inferenceBeamSearchContextAttributes}
		initialState = native.initialiseBeamActivationState(features, None, time, conceptState, arrival, selectedColumn)
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
			if(deactivated is None):
				raise RuntimeError(inferenceBeamSearchInvalidState)
			initialState[inferenceBeamSearchDeactivatedKey] = set(deactivated)
		initialConstraint = constraints.createConstraintState(allowedColumns, constraintMode)
		localColumns = dict(observedColumns)
		database.inferenceBeamSearchLastTrace = []
		try:
			rootCandidates = candidates(database, initialState, initialConstraint, position, connectedColumns, connectedFeatures)
			if(not rootCandidates):
				if(debugTerminateInferenceOnNoPredictionCandidatesAvailable):
					raise RuntimeError(inferenceBeamSearchNoCandidates)
				raise constraints.InferenceStopSequenceNoPredictionCandidatesAvailable(inferenceBeamSearchNoCandidates)
			if(inferenceBeamSearchLimitRootGap):
				rootCandidates = filterRootCandidates(rootCandidates)
			rootOrder = {candidate.token:index for index, candidate in enumerate(rootCandidates)}
			pruningWeight = inferenceBeamSearchDiagnosticMaximumWeight if inferenceBeamSearchRecordDiagnostics else inferenceBeamSearchContinuationWeight
			searchDepth = inferenceBeamDepth
			if(inferenceBeamSearchPruneDominated and not inferenceBeamSearchRecordDiagnostics):
				if(len(rootCandidates) == inferenceBeamSearchOne or inferenceBeamSearchContinuationWeight == inferenceBeamSearchZero):
					searchDepth = inferenceBeamSearchOne
				elif(inferenceBeamSearchContinuationFloor <= inferenceBeamSearchOne / database.f):
					# A nonempty continuation has a maximum token probability >= 1/vocabulary; an empty one uses the configured floor.
					guaranteedScore = rootCandidates[inferenceBeamSearchZero].logProbability + inferenceBeamSearchContinuationWeight * math.log(inferenceBeamSearchContinuationFloor)
					if(all(candidate.logProbability < guaranteedScore for candidate in rootCandidates[inferenceBeamSearchOne:])):
						searchDepth = inferenceBeamSearchOne
			paths = [BeamPath(initialState, copyContext(originalContext), initialConstraint, connectedColumns, connectedFeatures, candidate, candidate.logProbability, float(inferenceBeamSearchZero)) for candidate in rootCandidates]
			transitionCache = None
			if(inferenceBeamSearchPruneDominated and not inferenceBeamSearchRecordDiagnostics and searchDepth == inferenceBeamSearchThree):
				transitionCache = {}
				if(proveLeadingRoot(database, localColumns, paths, position, transitionCache)):
					searchDepth = inferenceBeamSearchOne
				installContext(database, originalContext)
			frontier = [(path, path.root) for path in paths]
			completed = []
			rootBestFuture = {candidate.token:float(inferenceBeamSearchZero) if inferenceBeamDepth == inferenceBeamSearchOne else inferenceBeamSearchMinimumScore for candidate in rootCandidates}
			for depth in range(inferenceBeamSearchOne, searchDepth):
				expanded = []
				bestCompletedScore = inferenceBeamSearchMinimumScore
				for path, action in frontier:
					# Every normalised continuation log probability is non-positive, so this is an exact upper bound.
					if(inferenceBeamSearchPruneDominated and depth == inferenceBeamDepth - inferenceBeamSearchOne and (inferenceBeamDepth == inferenceBeamSearchTwo or not inferenceBeamSearchRecordDiagnostics) and path.score < bestCompletedScore):
						continue
					state, context, nextConstraint, nextColumns, nextFeatures, nextCandidates = advancePath(database, localColumns, path, action, position + depth, depth == inferenceBeamDepth - inferenceBeamSearchOne, transitionCache)
					if(nextCandidates):
						for candidate in nextCandidates:
							future = path.future + candidate.logProbability
							score = path.root.logProbability + inferenceBeamSearchContinuationWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne)
							child = BeamPath(state, context, nextConstraint, nextColumns, nextFeatures, path.root, score, future)
							expanded.append((child, candidate))
							if(depth == inferenceBeamDepth - inferenceBeamSearchOne):
								rootBestFuture[path.root.token] = max(rootBestFuture[path.root.token], future)
								bestCompletedScore = max(bestCompletedScore, path.root.logProbability + pruningWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne))
					else:
						# A missing continuation is an explicit floor-probability event, not an unpenalised shorter path.
						future = path.future + (inferenceBeamDepth - depth) * math.log(inferenceBeamSearchContinuationFloor)
						score = path.root.logProbability + inferenceBeamSearchContinuationWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne)
						completed.append(BeamPath(state, context, nextConstraint, nextColumns, nextFeatures, path.root, score, future))
						rootBestFuture[path.root.token] = max(rootBestFuture[path.root.token], future)
						bestCompletedScore = max(bestCompletedScore, path.root.logProbability + pruningWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne))
					installContext(database, originalContext)
				frontier = sorted(expanded, key=lambda entry:(entry[inferenceBeamSearchZero].score, -rootOrder[entry[inferenceBeamSearchZero].root.token]), reverse=True)[:inferenceBeamWidth]
				if(not frontier):
					break
			available = [entry[inferenceBeamSearchZero] for entry in frontier] + completed
			if(not available):
				raise RuntimeError(inferenceBeamSearchInvalidState)
			best = max(available, key=lambda path:(path.score, -rootOrder[path.root.token]))
			if(inferenceBeamSearchRecordDiagnostics):
				# None marks an unexpanded branch. Two-step diagnostic pruning permits replay up to the configured maximum weight.
				database.inferenceBeamSearchLastTrace = [(candidate.token, candidate.column, candidate.feature, candidate.logProbability, None if rootBestFuture[candidate.token] == inferenceBeamSearchMinimumScore else rootBestFuture[candidate.token], candidate.activation, candidate.eligible) for candidate in rootCandidates]
			result = best.root.column, best.root.feature
		finally:
			for name, value in originalContext.items():
				setattr(database, name, value)
	return result


def proveLeadingRoot(database, observedColumns, paths, position, transitionCache):
	if(inferenceBeamSearchLikelihood and inferenceBeamSearchPruneDominated):
		if(inferenceBeamDepth != inferenceBeamSearchThree):
			raise RuntimeError(inferenceBeamSearchInvalidConfiguration)
		path = paths[inferenceBeamSearchZero]
		action = path.root
		future = float(inferenceBeamSearchZero)
		for depth in range(inferenceBeamSearchOne, inferenceBeamDepth):
			state, context, nextConstraint, nextColumns, nextFeatures, nextCandidates = advancePath(database, observedColumns, path, action, position + depth, depth == inferenceBeamDepth - inferenceBeamSearchOne, transitionCache)
			if(not nextCandidates):
				future += (inferenceBeamDepth - depth) * math.log(inferenceBeamSearchContinuationFloor)
				break
			action = nextCandidates[inferenceBeamSearchZero]
			future += action.logProbability
			path = BeamPath(state, context, nextConstraint, nextColumns, nextFeatures, path.root, path.root.logProbability + inferenceBeamSearchContinuationWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne), future)
		completedScore = path.root.logProbability + inferenceBeamSearchContinuationWeight * future / (inferenceBeamDepth - inferenceBeamSearchOne)
		# The first child is its root's best and outranks every competing root's upper bound, so it survives the intermediate beam. Its fully scored continuation then proves the winning root.
		result = all(other.root.logProbability < completedScore for other in paths[inferenceBeamSearchOne:])
	return result


def advancePath(database, observedColumns, path, action, nextPosition, terminal, transitionCache):
	if(inferenceBeamSearchLikelihood):
		import GIAANNcmn_predictionBeamSearch as native
		if(inferenceBeamSearchPruneDominated and transitionCache is not None):
			key = id(path.state), action.column, action.feature, nextPosition
		if(inferenceBeamSearchPruneDominated and transitionCache is not None and key in transitionCache):
			result = transitionCache[key]
		else:
			installContext(database, path.context)
			state = native.cloneBeamActivationState(path.state)
			nextColumns, nextFeatures = constraints.buildConnectedColumnsLookupForBeamNodes(database, observedColumns, [(action.column, action.feature)])
			if(inferenceBeamSearchProjectTerminalState and terminal):
				state[inferenceBeamSearchFeaturesKey] = projectTerminalState(database, observedColumns, state[inferenceBeamSearchFeaturesKey], action.column, action.feature)
			likelihood.prepareContextLikelihoodStep(database, nextPosition)
			columnLikelihood.prepareColumnLikelihoodStep(database, nextPosition, action.column)
			native.executeBeamNodeActivation(database, observedColumns, state, action.column, action.feature, nextPosition, None)
			nextConstraint = native.updateConstraintStateAfterNodes(database, path.constraint, [(action.column, action.feature)])
			context = copyContext({name:getattr(database, name) for name in inferenceBeamSearchContextAttributes})
			nextCandidates = candidates(database, state, nextConstraint, nextPosition, nextColumns, nextFeatures)
			result = state, context, nextConstraint, nextColumns, nextFeatures, nextCandidates
			if(inferenceBeamSearchPruneDominated and transitionCache is not None):
				transitionCache[key] = result
	return result


def filterRootCandidates(candidates):
	if(inferenceBeamSearchLikelihood and inferenceBeamSearchLimitRootGap):
		if(not candidates):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		# Optional accuracy policy: lookahead may only rerank roots close to the immediate leader.
		minimumScore = candidates[inferenceBeamSearchZero].logProbability - inferenceBeamSearchMaximumRootLogGap
		result = [candidate for candidate in candidates if candidate.logProbability >= minimumScore]
	return result


def projectTerminalState(database, observedColumns, features, sourceColumn, sourceFeature):
	if(inferenceBeamSearchLikelihood and inferenceBeamSearchProjectTerminalState):
		if(multipleDendriticBranchesNumber != inferenceBeamSearchOne or not predictionEnsureConnectedToPreviousPredictionStrict or not useSANIfeaturesAndColumns):
			raise RuntimeError(inferenceBeamSearchInvalidConfiguration)
		# LIF propagation is neuron-local. The final readout can only see direct targets, and source activation also needs its own segments.
		observedColumn = constraints.getObservedColumn(database, observedColumns, sourceColumn)
		if(observedColumn is None):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		connections = observedColumn.prepareFeatureConnectionsForSourceFeature(sourceFeature, targetDevice=features.device, createMissing=False)[database.arrayIndexPropertiesStrengthIndex].coalesce()
		connectionIndices = connections.indices()
		direct = connectionIndices[inferenceContextLikelihoodSegmentDimension] == arrayIndexSegmentSoma
		neurons = connectionIndices[inferenceContextLikelihoodColumnDimension, direct] * database.f + connectionIndices[inferenceContextLikelihoodFeatureDimension, direct]
		neurons = pt.unique(pt.cat((neurons, pt.tensor([sourceColumn * database.f + sourceFeature], dtype=pt.long, device=features.device))), sorted=True)
		result = selectTerminalNeuronState(database, features, neurons)
	return result


def selectTerminalNeuronState(database, features, neurons):
	if(inferenceBeamSearchLikelihood and inferenceBeamSearchProjectTerminalState):
		if(not isinstance(features, pt.Tensor) or not features.is_sparse or features.sparse_dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or tuple(features.shape) != (inferenceBeamSearchOne, arrayNumberOfSegments, database.c, database.f) or not features.is_floating_point()):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		if(not isinstance(neurons, pt.Tensor) or neurons.dim() != inferenceBeamSearchOne or neurons.dtype != pt.long or neurons.device != features.device or (neurons < inferenceBeamSearchZero).any() or (neurons >= database.c * database.f).any() or (neurons[inferenceBeamSearchOne:] <= neurons[:-inferenceBeamSearchOne]).any()):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		segments = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=features.device)
		query = (segments[:, None] * database.c * database.f + neurons[None, :]).flatten()
		activation = features.coalesce()
		indices = activation.indices()
		values = activation.values()
		selected = pt.empty((inferenceBeamSearchZero,), dtype=pt.long, device=features.device)
		if(values.numel() > inferenceBeamSearchZero):
			keys = (indices[inferenceLeakyIntegrateAndFireSegmentDimension] * database.c + indices[inferenceLeakyIntegrateAndFireConceptDimension]) * database.f + indices[inferenceLeakyIntegrateAndFireFeatureDimension]
			positions = pt.searchsorted(keys, query)
			valid = positions < keys.numel()
			positions = positions.clamp(max=keys.numel() - inferenceBeamSearchOne)
			selected = positions[valid & (keys[positions] == query)]
		result = pt.sparse_coo_tensor(indices[:, selected], values[selected], size=activation.size(), dtype=activation.dtype, device=activation.device, is_coalesced=True)
	return result


def validateConfiguration():
	if(inferenceBeamSearchLikelihood):
		if(not inferenceBeamSearch or inferenceBeamScoreStrategy != inferenceBeamSearchLikelihoodStrategy or not inferenceContextLikelihood or not inferenceColumnLikelihood or not inferenceReviewPatch16TokenAggregation or not predictionEnsureConnectedToPreviousPredictionStrict or not inferenceLeakyIntegrateAndFire or multipleDendriticBranches or useTrainDuringInference or inferenceTrainFirstSequences or useSequenceContext or auxiliaryNeurons or inferenceUseNeuronFeaturePropertiesTime or useGPUdense or useGPUsparse):
			raise RuntimeError(inferenceBeamSearchInvalidConfiguration)
		for value, maximum in ((inferenceBeamWidth, inferenceBeamSearchMaximumWidth), (inferenceBeamDepth, inferenceBeamSearchMaximumDepth)):
			if(type(value) is not int or not inferenceBeamSearchOne <= value <= maximum):
				raise ValueError(inferenceBeamSearchInvalidParameter)
		if(type(inferenceBeamSearchExtraCandidates) is not int or not inferenceBeamSearchZero <= inferenceBeamSearchExtraCandidates <= inferenceBeamSearchMaximumExtraCandidates):
			raise ValueError(inferenceBeamSearchInvalidParameter)
		if(any(type(value) is not bool for value in (inferenceBeamSearchAdmitSubthreshold, inferenceBeamSearchRecordDiagnostics, inferenceBeamSearchPruneDominated, inferenceBeamSearchProjectTerminalState, inferenceBeamSearchPreserveEligibleScores, inferenceBeamSearchLimitRootGap))):
			raise ValueError(inferenceBeamSearchInvalidParameter)
		if(inferenceBeamSearchLimitRootGap):
			if(type(inferenceBeamSearchMaximumRootLogGap) not in (int, float) or not math.isfinite(inferenceBeamSearchMaximumRootLogGap) or inferenceBeamSearchMaximumRootLogGap < inferenceBeamSearchZero):
				raise ValueError(inferenceBeamSearchInvalidParameter)
		for value in (inferenceBeamSearchMinimumActivationRatio, inferenceBeamSearchContinuationWeight, inferenceBeamSearchSubthresholdPenalty, inferenceBeamSearchDiagnosticMaximumWeight):
			if(type(value) not in (int, float) or not math.isfinite(value) or not inferenceBeamSearchZero <= value <= inferenceBeamSearchOne):
				raise ValueError(inferenceBeamSearchInvalidParameter)
		if(inferenceBeamSearchRecordDiagnostics and inferenceBeamSearchContinuationWeight > inferenceBeamSearchDiagnosticMaximumWeight):
			raise ValueError(inferenceBeamSearchInvalidParameter)
		if(type(inferenceBeamSearchContinuationFloor) not in (int, float) or not math.isfinite(inferenceBeamSearchContinuationFloor) or not inferenceBeamSearchZero < inferenceBeamSearchContinuationFloor < inferenceBeamSearchOne):
			raise ValueError(inferenceBeamSearchInvalidParameter)
		if(type(inferenceLeakyIntegrateAndFireSomaActivationThreshold) not in (int, float) or not math.isfinite(inferenceLeakyIntegrateAndFireSomaActivationThreshold) or inferenceLeakyIntegrateAndFireSomaActivationThreshold <= inferenceBeamSearchZero):
			raise ValueError(inferenceBeamSearchInvalidParameter)
	return


def installContext(database, context):
	if(inferenceBeamSearchLikelihood):
		for name, value in copyContext(context).items():
			setattr(database, name, value)
	return


def copyContext(context):
	if(inferenceBeamSearchLikelihood):
		result = {}
		for name, value in context.items():
			if(isinstance(value, list)):
				result[name] = list(value)
			elif(isinstance(value, dict)):
				result[name] = {key:set(origins) for key, origins in value.items()}
			else:
				result[name] = value
	return result


def candidates(database, state, constraint, position, connectedColumns, connectedFeatures):
	if(inferenceBeamSearchLikelihood):
		import GIAANNcmn_predictionBeamSearch as native
		columns, features, activation = native.calculateSelectionActivationDistribution(database, state[inferenceBeamSearchFeaturesKey], state[inferenceBeamSearchTimeKey], constraint, connectedColumns, connectedFeatures, position, None, True, state[inferenceBeamSearchArrivalKey], selectedColumnIndex=state[inferenceBeamSearchColumnKey], beamSearchAdmission=True)
		result = []
		if(columns is not None):
			if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
				columns, features, activation = native.filterCandidatesByDeactivatedNeuronState(database, columns, features, activation, state[inferenceBeamSearchDeactivatedKey])
			if(columns is not None):
				allowed = pt.tensor([constraints.constraintAllowsNode(database, column, feature, constraint) and (inferenceColumnConstraintsAllowExternalPrimeConceptTransitionsBeamCandiateFilteringPatch or constraints.constraintAllowsColumn(column, constraint)) for column, feature in zip(columns.tolist(), features.tolist())], dtype=pt.bool, device=activation.device)
				columns, features, activation = columns[allowed], features[allowed], activation[allowed]
				if(activation.numel() > inferenceBeamSearchZero):
					scores = scoreCandidates(database, columns, features, activation, constraint)
					if(inferenceBeamSearchSubthresholdPenalty > inferenceBeamSearchZero):
						scores = scores * (activation / inferenceLeakyIntegrateAndFireSomaActivationThreshold).clamp(max=inferenceBeamSearchOne).pow(inferenceBeamSearchSubthresholdPenalty)
					result = aggregateCandidates(database, columns, features, activation, scores)
	return result


def scoreCandidates(database, columns, features, activation, constraint):
	if(inferenceBeamSearchLikelihood):
		scoredColumns, scoredFeatures, scores = likelihood.scoreContextLikelihoodCandidates(database, columns, features, activation, constraint)
		if(not pt.equal(columns, scoredColumns) or not pt.equal(features, scoredFeatures)):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		if(inferenceBeamSearchPreserveEligibleScores):
			eligible = activation >= inferenceLeakyIntegrateAndFireSomaActivationThreshold
			if(eligible.any() and not eligible.all()):
				eligibleColumns, eligibleFeatures, eligibleScores = likelihood.scoreContextLikelihoodCandidates(database, columns[eligible], features[eligible], activation[eligible], constraint)
				if(not pt.equal(columns[eligible], eligibleColumns) or not pt.equal(features[eligible], eligibleFeatures)):
					raise RuntimeError(inferenceBeamSearchInvalidState)
				# Keep the original eligible readout; the union model supplies the added candidates' relative mass.
				scores[~eligible] *= eligibleScores.sum() / scores[eligible].sum()
				scores[eligible] = eligibleScores
	return scores


def admitCandidates(columns, features, activation):
	if(inferenceBeamSearchLikelihood):
		if(columns is None or features is None or activation is None or columns.shape != features.shape or features.shape != activation.shape or activation.dim() != inferenceBeamSearchOne or columns.dtype != pt.long or features.dtype != pt.long or not activation.is_floating_point() or columns.device != activation.device or features.device != activation.device or (columns < inferenceBeamSearchZero).any() or (features < inferenceBeamSearchZero).any() or not pt.isfinite(activation).all() or (activation < inferenceBeamSearchZero).any()):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		selected = activation >= inferenceLeakyIntegrateAndFireSomaActivationThreshold
		if(inferenceBeamSearchAdmitSubthreshold and inferenceBeamSearchExtraCandidates > inferenceBeamSearchZero):
			weak = pt.nonzero((activation > inferenceBeamSearchZero) & (activation >= inferenceLeakyIntegrateAndFireSomaActivationThreshold * inferenceBeamSearchMinimumActivationRatio) & ~selected).flatten()
			if(weak.numel() > inferenceBeamSearchZero):
				extra = weak[pt.topk(activation[weak], min(inferenceBeamSearchExtraCandidates, weak.numel())).indices]
				selected[extra] = True
		if(selected.any()):
			result = columns[selected], features[selected], activation[selected]
		else:
			result = None, None, None
	return result


def aggregateCandidates(database, columns, features, activation, scores):
	if(inferenceBeamSearchLikelihood):
		tokens = features.clone()
		prime = features == featureIndexPrimeConceptNeuron
		if(prime.any()):
			tokens[prime] = pt.tensor([database.conceptFeaturesDict[database.conceptColumnsList[column]] for column in columns[prime].tolist()], dtype=tokens.dtype, device=tokens.device)
		unique, inverse = pt.unique(tokens, sorted=True, return_inverse=True)
		totals = pt.zeros_like(unique, dtype=scores.dtype).scatter_add_(inferenceBeamSearchZero, inverse, scores)
		if(not pt.isfinite(totals).all() or (totals <= inferenceBeamSearchZero).any()):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		# Preserve native top-one tie handling even when more than one token is retained.
		first = pt.topk(totals, inferenceBeamSearchOne).indices
		order = first
		if(unique.numel() > inferenceBeamSearchOne and inferenceBeamWidth > inferenceBeamSearchOne):
			remaining = totals.clone()
			remaining[first] = inferenceBeamSearchMinimumScore
			order = pt.cat((first, pt.topk(remaining, min(inferenceBeamWidth - inferenceBeamSearchOne, unique.numel() - inferenceBeamSearchOne)).indices))
		logProbabilities = totals.log() - totals.sum().log()
		result = []
		for index in order.tolist():
			neuronScores = pt.where(inverse == index, scores, inferenceBeamSearchMinimumScore)
			neuron = int(pt.topk(neuronScores, inferenceBeamSearchOne).indices.item())
			result.append(BeamCandidate(int(columns[neuron]), int(features[neuron]), int(unique[index]), float(logProbabilities[index]), float(activation[neuron]), bool(pt.any(activation[inverse == index] >= inferenceLeakyIntegrateAndFireSomaActivationThreshold))))
	return result
