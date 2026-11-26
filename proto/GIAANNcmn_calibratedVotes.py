"""GIAANNcmn_calibratedVotes.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common calibrated Votes

Calibrate positive outgoing contacts using earlier observed source populations.

"""

from collections import deque
from functools import lru_cache
import numpy as np
from GIAANNcmn_globalDefs import *
from GIAANNcmn_predictionBeamSearchObservedSources import outgoing, applyDendriticEvidence
import GIAANNcmn_predictionContextLikelihood as likelihood


class SourceHistory:
	def __init__(self, index):
		if(inferenceUpgradeCalibratedVotes):
			self.index = index
			lags = min(upgradeSourceHistoryLags, arrayIndexSegmentSoma - arrayNumberOfSegmentsColumnDistance, inferenceContextLikelihoodHistoryLength - upgradeOne)
			self.populations = deque(maxlen=lags + upgradeOne)
			self.transitions = deque(maxlen=lags)
			if(not hasattr(index, upgradeCalibratedSourceFrameAttribute)):
				index.calibratedSourceFrame = lru_cache(maxsize=upgradeSourceFrameCacheSize)(lambda ordinal: sourceFrame(index, ordinal))
		return

	def observe(self, posterior, column, feature):
		if(inferenceUpgradeCalibratedVotes):
			index = self.index
			word, sources, weights = posterior
			keep = np.argsort(-weights, kind=upgradeStableSort)[:upgradeRetainedSources]
			candidates, probabilities = sources[keep], weights[keep]
			key = index.encode(column, feature)
			actual = index.keys.searchsorted(key)
			if(actual < index.keys.size and index.keys[actual] == key and actual not in candidates):
				position = sources.searchsorted(actual)
				if(position < sources.size and sources[position] == actual):
					mass = weights[position]
				else:
					start, end = index.lex[word:word + upgradeTwo]
					mass = inferenceBeamSearchObservedSourcePriorWeight * index.counts[actual] / index.counts[start:end].sum(dtype=np.float64)
				candidates = np.append(candidates, actual)
				probabilities = np.append(probabilities, mass)
			if(probabilities.size):
				probabilities = probabilities / probabilities.sum()
			population = word, candidates, probabilities
			if(self.populations):
				self.transitions.append(reverseTransitions(index, self.populations[-upgradeOne], candidates))
			self.populations.append(population)
		return

	def score(self, posterior):
		if(inferenceUpgradeCalibratedVotes):
			index = self.index
			cells, raw = outgoing(index, posterior)
			adjusted = raw
			if(cells.size):
				database = index.database
				base = index.counts[cells] / index.counts[cells].sum(dtype=np.float64)
				words, columns = np.divmod(index.keys[cells] // upgradeTwo, database.c)
				physical = columns * database.f + np.where(index.keys[cells] % upgradeTwo, featureIndexPrimeConceptNeuron, words)
				native = applyDendriticEvidence(index, cells, raw)
				logMass = np.log(native) - (upgradeOne - upgradeImmediateStrengthExponent) * np.log(raw)
				back = self.populations[-upgradeOne][upgradeTwo]
				coefficient = inferenceContextLikelihoodContextWeight * (inferenceLikelihoodCalibrationExponent if inferenceLikelihoodCalibration else upgradeOne)
				for lag in range(upgradeOne, len(self.populations)):
					_, sources, forward = self.populations[-lag - upgradeOne]
					back = self.transitions[-lag] @ back if back.size and forward.size else forward
					query = (arrayIndexSegmentSoma - lag) * database.c * database.f + physical
					original = database.contextLikelihoodHistory[-lag - upgradeOne]
					nativeCounts = frameValues(original, query)
					mixed = np.zeros(cells.size)
					keep = np.argsort(-back, kind=upgradeStableSort)[:upgradeTransmittingSources]
					for source, weight in zip(sources[keep], back[keep] / back[keep].sum(), strict=True):
						mixed += weight * frameValues(index.calibratedSourceFrame(int(source)), query)
					smoothing = inferenceContextLikelihoodSmoothing * base
					logMass += coefficient * (upgradeDelayedEvidenceGain * np.log1p(mixed / smoothing) - np.log1p(nativeCounts / smoothing))
				adjusted = np.exp(logMass - logMass.max())
		return cells, adjusted


def reverseTransitions(index, previous, destinations):
	if(inferenceUpgradeCalibratedVotes):
		word, sources, prior = previous
		keys, strengths, total = index.pool(word)
		shared = inferenceBeamSearchObservedSourcePriorWeight * index.counts[destinations] / index.total
		if(total > upgradeZero):
			shared += (upgradeOne - inferenceBeamSearchObservedSourcePriorWeight) * lookup(keys, strengths, destinations) / total
		matrix = np.empty((sources.size, destinations.size))
		for row, source in enumerate(sources):
			keys, strengths, total = index.row(int(source))
			direct = lookup(keys, strengths, destinations) / total if total > upgradeZero else shared
			matrix[row] = prior[row] * ((upgradeOne - upgradeReversePoolWeight) * direct + upgradeReversePoolWeight * shared)
		if(sources.size and destinations.size):
			matrix /= matrix.sum(axis=upgradeZero)
	return matrix


def sourceFrame(index, source):
	frame = None
	if(inferenceUpgradeCalibratedVotes):
		tensor = index.loadSourceTensor(source)
		if(tensor is not None):
			connections = tensor[index.database.arrayIndexPropertiesStrengthIndex].coalesce()
			frame = likelihood.getContextLikelihoodFeatureFrame(index.database, connections)
	return frame


def frameValues(frame, keys):
	if(inferenceUpgradeCalibratedVotes):
		result = np.zeros(keys.size)
		if(frame is not None):
			result = lookup(frame[upgradeZero].cpu().numpy(), frame[upgradeOne].cpu().numpy(), keys)
	return result


def lookup(keys, values, query):
	if(inferenceUpgradeCalibratedVotes):
		result = np.zeros(query.size)
		if(keys.size and query.size):
			positions = np.minimum(keys.searchsorted(query), keys.size - upgradeOne)
			matches = keys[positions] == query
			result[matches] = values[positions[matches]]
	return result
