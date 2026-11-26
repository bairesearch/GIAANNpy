"""GIAANNcmn_predictionUpgrades.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common prediction Upgrades

Sequence-local readouts over the trained physical connections.

"""


from collections import Counter, deque
import numpy as np
from GIAANNcmn_globalDefs import *
if(inferenceUpgradeObservedSources):
	import GIAANNcmn_predictionBeamSearchObservedSources as observedSources
if(inferenceUpgradeCalibratedVotes):
	import GIAANNcmn_calibratedVotes as calibratedVotes
if(trainUpgradeLearnEightLagInputs):
	import GIAANNcmn_eightLagInputs as eightLagInputs
if(inferenceUpgradeBranchBank):
	import GIAANNcmn_sharedInputBranches as branches


class TokenScores:
	"""Token mass and the strongest contributing physical cell for each token."""
	def __init__(self, words=(), weights=(), cells=()):
		if(inferenceUpgradeReadout):
			words = np.asarray(words, dtype=np.int64)
			weights = np.asarray(weights, dtype=np.float64)
			cells = np.asarray(cells, dtype=np.int64)
			if(words.ndim != upgradeOne or np.any(words <= upgradeZero) or np.any(cells < upgradeZero) or words.shape != weights.shape or weights.shape != cells.shape or not np.isfinite(weights).all() or np.any(weights < upgradeZero)):
				raise ValueError(upgradeInvalidScores)
			order = np.lexsort((cells, -weights, words))
			orderedWords = words[order]
			first = np.flatnonzero(np.r_[True, orderedWords[upgradeOne:] != orderedWords[:-upgradeOne]]) if order.size else order
			self.words = orderedWords[first]
			self.cells = cells[order[first]]
			self.weights = np.add.reduceat(weights[order], first)
		return

	def normalise(self):
		if(inferenceUpgradeReadout and self.weights.size):
			total = self.weights.sum()
			if(total <= upgradeZero or not np.isfinite(total)):
				raise ValueError(upgradeInvalidScores)
			self.weights /= total
		return self

	def winner(self, preferred=None):
		result = None
		if(inferenceUpgradeReadout and self.weights.size):
			position = self.weights.argmax()
			if(preferred is not None):
				index = self.words.searchsorted(preferred)
				if(index < self.words.size and self.words[index] == preferred and self.weights[index] == self.weights[position]):
					position = index
			result = int(self.words[position]), int(self.cells[position])
		return result

	def lookup(self, words):
		if(inferenceUpgradeReadout):
			result = np.zeros(np.shape(words), dtype=np.float64)
			if(self.words.size):
				indices = self.words.searchsorted(words).clip(max=self.words.size - upgradeOne)
				result = np.where(self.words[indices] == words, self.weights[indices], result)
		return result


def fromCells(database, cells, weights):
	if(inferenceUpgradeReadout):
		columns, features = np.divmod(cells, database.f)
		words = np.where(features == featureIndexPrimeConceptNeuron, columns + upgradeOne, features)
		result = TokenScores(words, weights, cells)
	return result


def combine(*weighted):
	if(inferenceUpgradeReadout):
		words = np.concatenate([part.words for part, scale in weighted])
		cells = np.concatenate([part.cells for part, scale in weighted])
		weights = np.concatenate([part.weights * scale for part, scale in weighted])
		result = TokenScores(words, weights, cells).normalise()
	return result


def fromOrdinals(index, ordinals, weights):
	if(inferenceUpgradeObservedSources):
		packed = index.keys[ordinals]
		words, columns = np.divmod(packed // upgradeTwo, index.database.c)
		features = np.where(packed % upgradeTwo, featureIndexPrimeConceptNeuron, words)
		result = TokenScores(words, weights, columns * index.database.f + features)
	return result


def strongestTokenScores(scores, limit):
	if(inferenceUpgradeCalibratedVotes or trainUpgradeLearnEightLagInputs):
		if(type(limit) is not int or limit < upgradeOne):
			raise ValueError(upgradeInvalidScores)
		order = np.argsort(-scores.weights, kind=upgradeStableSort)[:limit]
		result = TokenScores(scores.words[order], scores.weights[order].astype(np.float32), scores.cells[order])
	return result


def initialiseSequence(database):
	if(inferenceUpgradeReadout):
		if(not tokeniserSubword or not useDedicatedFeatureListsSubword or not inferenceContextLikelihood or not inferenceColumnLikelihood or inferenceBeamSearch or not inferenceLeakyIntegrateAndFire or multipleDendriticBranchesNumber != upgradeOne or useTrainDuringInference or inferenceTrainFirstSequences or numSeedTokens < upgradeOne or database.f != database.c + upgradeOne):
			raise ValueError(upgradeInvalidConfiguration)
		if(inferenceUpgradeObservedSources and not hasattr(database, inferenceBeamSearchObservedSourceIndexAttribute)):
			database.observedSourceIndex = observedSources.SourceIndex(database)
		database.upgradeReadout = SequenceReadout()
		if(inferenceUpgradeCalibratedVotes):
			database.upgradeReadout.sourceHistory = calibratedVotes.SourceHistory(database.observedSourceIndex)
	return


class SequenceReadout:
	def __init__(self):
		if(inferenceUpgradeReadout):
			self.previousWord = self.word = self.posterior = self.localCandidates = None
			self.nativeWinner = self.observedWinner = None
			self.advantage = upgradeZero
			self.previousTraceSource = None
			self.exactContacts, self.sharedContacts = Counter(), Counter()
			self.expiry = deque()
			if(trainUpgradeLearnEightLagInputs):
				self.inputHistory = deque(maxlen=upgradeInputLags)
		return

	def observe(self, database, column, feature, seed):
		if(inferenceUpgradeReadout):
			if(not upgradeZero <= column < database.c or not upgradeZero <= feature < database.f):
				raise ValueError(upgradeInvalidSource)
			self.previousWord, self.word = self.word, feature or column + upgradeOne
			self.localCandidates = None
			if(trainUpgradeLearnEightLagInputs):
				self.inputHistory.append(self.word)
			if(inferenceUpgradeAdaptLocalAndObservedSourceWeights and not inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
				self.advantage = upgradeAdvantageDecay * self.advantage + int(self.observedWinner == self.word) - int(self.nativeWinner == self.word)
			if(inferenceUpgradeObservedSources):
				index = database.observedSourceIndex
				self.posterior = observedSources.observe(index, self.posterior, self.word, upgradeSourceWidth, inferenceBeamSearchObservedSourcePoolWeight, inferenceBeamSearchObservedSourcePriorWeight)
				if(inferenceUpgradeCalibratedVotes):
					self.sourceHistory.observe(self.posterior, column, feature)
				if(not seed and (inferenceUpgradeAdaptLocalAndObservedSourceWeights or inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeCalibratedVotes)):
					self.advanceTraces(index, column, feature)
		return

	def advanceTraces(self, index, column, feature):
		if(inferenceUpgradeAdaptLocalAndObservedSourceWeights or inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeCalibratedVotes):
			key = int(index.encode(np.int64(column), np.int64(feature)))
			ordinal = int(index.keys.searchsorted(key))
			if(ordinal >= index.keys.size or index.keys[ordinal] != key):
				ordinal = None
			if(self.previousTraceSource is not None):
				lastOrdinal, lastWord = self.previousTraceSource
				exact = traversedContact(index.row, lastOrdinal, ordinal) if inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeCalibratedVotes else None
				shared = traversedContact(index.pool, lastWord, ordinal) if inferenceUpgradeAdaptLocalAndObservedSourceWeights else None
				self.expiry.append((exact, shared))
				for trace, contact in zip((self.exactContacts, self.sharedContacts), (exact, shared), strict=True):
					if(contact is not None):
						trace[contact] += upgradeOne
				if(len(self.expiry) > upgradeTraceWindow):
					for trace, contact in zip((self.exactContacts, self.sharedContacts), self.expiry.popleft(), strict=True):
						if(contact is not None):
							trace[contact] -= upgradeOne
							if(not trace[contact]):
								del trace[contact]
			self.previousTraceSource = ordinal, self.word
		return

	def select(self, database, nativeCell):
		if(inferenceUpgradeReadout):
			nativeWord = None if nativeCell is None else nativeCell[upgradeOne] or nativeCell[upgradeZero] + upgradeOne
			local = self.localDistribution(database, nativeCell)
			nativeCandidates = local
			if(inferenceUpgradeCalibratedVotes):
				cells, mass = self.sourceHistory.score(self.posterior)
				observed = fromOrdinals(database.observedSourceIndex, cells, mass).normalise()
			else:
				observed = self.observedDistribution(database, nativeCell)
			observedPrimary = inferenceUpgradeObservedSourceInference or inferenceUpgradeApplyBranchEvidenceByCandidateToken
			if(inferenceUpgradeBranchBank):
				cells, weights = branches.response(database, self.word, self.previousWord)
				drive = fromCells(database, cells, weights)
				if(inferenceUpgradeApplyBranchEvidenceByCandidateToken and drive.words.size):
					observed.weights *= upgradeOne + np.minimum(upgradeBranchGainCap, drive.lookup(observed.words) / upgradeBranchSmoothing)
					observed.normalise()
				if(inferenceUpgradeLearnSharedInputDendriticBranches and drive.words.size):
					peak = drive.weights.max()
					alpha = upgradeBranchHalf * peak / (upgradeBranchSmoothing + peak)
					if(observedPrimary and not inferenceUpgradeAdaptLocalAndObservedSourceWeights and not inferenceUpgradeApplyBranchVoteOnce):
						observed = combine((observed, upgradeOne - alpha), (drive.normalise(), alpha))
					else:
						drive.normalise()
					if(observedPrimary and not inferenceUpgradeAdaptLocalAndObservedSourceWeights):
						observed = combine((observed, upgradeOne - alpha), (drive, alpha))
					else:
						local = combine((local, upgradeOne - alpha), (drive, alpha))
			# Exact traversal boosts belong to the local vote before adaptive mixing.
			if((inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeCalibratedVotes) and nativeCell is not None and self.previousTraceSource is not None):
				exact = traceResponse(database.observedSourceIndex, self.exactContacts, self.previousTraceSource[upgradeZero])
				if(exact.words.size):
					local = combine((local, upgradeOne), (exact, upgradeTraceStrength))
			if(inferenceUpgradeCalibratedVotes):
				pair = self.selectCalibrated(database, nativeCandidates, local, observed)
			else:
				result = observed if observedPrimary else local
				if(inferenceUpgradeAdaptLocalAndObservedSourceWeights):
					result = self.adaptiveDistribution(database, local, observed, nativeWord)
				# An explicit abstention fill uses the distributed forecast even when adaptation is enabled.
				if((inferenceUpgradeObservedSourcesOnAbstention or inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions) and nativeCell is None and observed.words.size):
					result = observed
				# The local readout keeps its native tie break. Distributed readouts break ties by token ID.
				preferNative = not (observedPrimary or inferenceUpgradeAdaptLocalAndObservedSourceWeights or (inferenceUpgradeLearnSharedInputDendriticBranches and drive.words.size))
				if(trainUpgradeLearnEightLagInputs):
					bank = eightLagInputs.response(database, self.inputHistory)
					if(predictionEnsureConnectedToPreviousPredictionStrict):
						result = TokenScores(nativeCandidates.words, (upgradeOne - upgradeInputVoteWeight) * result.lookup(nativeCandidates.words) + upgradeInputVoteWeight * bank.lookup(nativeCandidates.words), nativeCandidates.cells)
					else:
						result = combine((result, upgradeOne - upgradeInputVoteWeight), (bank, upgradeInputVoteWeight))
					preferNative = False
				selected = result.winner(preferred=nativeWord if preferNative else None)
				if(nativeCell is not None and preferNative and not inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions):
					pair = nativeCell
				elif(selected is None):
					pair = None
				elif(selected[upgradeZero] == nativeWord and preferNative):
					pair = nativeCell
				else:
					pair = divmod(selected[upgradeOne], database.f)
		return pair

	def selectCalibrated(self, database, nativeCandidates, local, observed):
		pair = None
		if(inferenceUpgradeCalibratedVotes):
			# Combine takes precedence when both options are enabled. A bounded vote retains its original probability mass.
			if(not inferenceUpgradeCombineNativeAndObservedSourceVotes or trainUpgradeLearnEightLagInputs):
				observed = strongestTokenScores(observed, upgradeCalibratedTokenLimit)
			weight = upgradeCalibratedVoteWeight
			inputVote = TokenScores()
			inputWeight = upgradeZero
			if(trainUpgradeLearnEightLagInputs):
				inputVote = eightLagInputs.response(database, self.inputHistory)
				inputWeight = upgradeInputVoteWeight
			if(inferenceUpgradeCombineNativeAndObservedSourceVotes):
				winner = combine((local, upgradeOne - weight), (observed, weight - inputWeight), (inputVote, inputWeight)).winner()
			elif(inferenceUpgradeRerankStrictCandidates):
				words = nativeCandidates.words
				mass = (upgradeOne - weight) * local.lookup(words) + (weight - inputWeight) * observed.lookup(words) + inputWeight * inputVote.lookup(words)
				winner = TokenScores(words, mass, nativeCandidates.cells).winner()
			if(winner is not None):
				pair = divmod(winner[upgradeOne], database.f)
		return pair

	def localDistribution(self, database, nativeCell):
		if(inferenceUpgradeReadout):
			result = TokenScores()
			if(nativeCell is not None):
				if(self.localCandidates is None):
					raise RuntimeError(upgradeInvalidScores)
				columns, features, weights = (array.detach().cpu().numpy() for array in self.localCandidates)
				result = fromCells(database, columns * database.f + features, weights).normalise()
		return result

	def observedDistribution(self, database, nativeCell):
		result = TokenScores()
		if(inferenceUpgradeObservedSources and self.posterior is not None and (nativeCell is None or inferenceUpgradeObservedSourceInference or inferenceUpgradeAdaptLocalAndObservedSourceWeights or inferenceUpgradeApplyBranchEvidenceByCandidateToken)):
			index = database.observedSourceIndex
			cells, mass = observedSources.outgoing(index, self.posterior)
			if(cells.size):
				mass = observedSources.applyDendriticEvidence(index, cells, mass)
				result = fromOrdinals(index, cells, mass).normalise()
		return result

	def adaptiveDistribution(self, database, local, observed, nativeWord):
		if(inferenceUpgradeAdaptLocalAndObservedSourceWeights):
			winner = observed.winner()
			self.nativeWinner = nativeWord
			self.observedWinner = winner[upgradeZero] if winner is not None else None
			weight = upgradeOne / (upgradeOne + np.exp(-np.clip(self.advantage / upgradeAdvantageScale, -upgradeAdvantageLimit, upgradeAdvantageLimit)))
			shared = traceResponse(database.observedSourceIndex, self.sharedContacts, self.word)
			result = combine((local, upgradeOne - weight), (observed, weight), (shared, upgradeTraceStrength))
		return result


def traversedContact(row, source, destination):
	result = None
	if((inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeAdaptLocalAndObservedSourceWeights or inferenceUpgradeCalibratedVotes) and source is not None and destination is not None):
		targets, strengths, total = row(source)
		position = targets.searchsorted(destination)
		if(position < targets.size and targets[position] == destination):
			result = source, destination
	return result


def traceResponse(index, contacts, source):
	if(inferenceUpgradeFacilitateRecentSynapsesAndFillAbstentions or inferenceUpgradeAdaptLocalAndObservedSourceWeights or inferenceUpgradeCalibratedVotes):
		targets = [(destination, count) for (origin, destination), count in contacts.items() if origin == source]
		ordinals = np.asarray([destination for destination, count in targets], dtype=np.int64)
		result = fromOrdinals(index, ordinals, [count for destination, count in targets])
	return result
