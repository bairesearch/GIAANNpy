"""GIAANNcmn_predictionBeamSearchObservedSources.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description: 
GIA ANN common prediction Beam Search Observed Sources

Infer the source cells of observed subwords on a frozen single-branch checkpoint.

Each observation admits every trained cell emitting that subword. The transition
law mixes each retained source's normalised row with the pooled row for its word
and the training occupancy prior. Predictions use the outgoing synapses of the
retained cells. Earlier dendritic frames can reweight their destination cells.

"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles as files
import GIAANNcmn_predictionConstraints as constraints
import GIAANNcmn_predictionContextLikelihood as likelihood
import GIAANNnlp_sequenceTokens as sequenceTokens


def predict(database, tokens, position):
	if(inferenceBeamSearchObservedSources):
		if(type(position) is not int or not inferenceBeamSearchZero < position <= len(tokens) or position < database.observedSourcePosition or position != database.contextLikelihoodPosition):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		if(not hasattr(database, inferenceBeamSearchObservedSourceIndexAttribute)):
			database.observedSourceIndex = SourceIndex(database)
		index = database.observedSourceIndex
		posterior = database.observedSourcePosterior
		for offset in range(database.observedSourcePosition, position):
			token = sequenceTokens.getTokeniserSubwordFeatureIndex(tokens[offset])
			posterior = observe(index, posterior, token, inferenceBeamWidth, inferenceBeamSearchObservedSourcePoolWeight, inferenceBeamSearchObservedSourcePriorWeight)
		cells, masses = outgoing(index, posterior)
		if(cells.size):
			if(inferenceBeamSearchObservedSourceContext):
				masses = applyDendriticEvidence(index, cells, masses)
			result = winner(index, cells, masses)
		database.observedSourcePosterior, database.observedSourcePosition = posterior, position
		if(not cells.size):
			if(debugTerminateInferenceOnNoPredictionCandidatesAvailable):
				raise RuntimeError(inferenceBeamSearchNoCandidates)
			raise constraints.InferenceStopSequenceNoPredictionCandidatesAvailable(inferenceBeamSearchNoCandidates)
	return result


def observe(index, posterior, token, width, poolWeight, priorWeight):
	if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
		if(type(token) is not int or not inferenceBeamSearchZero < token < index.database.f or type(width) is not int or not inferenceBeamSearchOne <= width <= inferenceBeamSearchMaximumWidth or type(poolWeight) not in (int, float) or not inferenceBeamSearchZero <= poolWeight <= inferenceBeamSearchOne or type(priorWeight) not in (int, float) or not inferenceBeamSearchZero < priorWeight <= inferenceBeamSearchOne):
			raise ValueError(inferenceBeamSearchInvalidParameter)
		start, end = index.lex[token:token + inferenceBeamSearchTwo]
		# Condition only after normalising each transition across its entire destination space.
		mass = index.counts[start:end] / index.total
		if(posterior is not None and posterior[inferenceBeamSearchOne].size and start < end):
			previous, sources, probabilities = posterior
			pooledTargets, pooledStrengths, pooledTotal = index.pool(previous)
			if(pooledTotal > inferenceBeamSearchZero):
				mass *= priorWeight
				addObservedMass(mass, start, end, pooledTargets, pooledStrengths, (inferenceBeamSearchOne - priorWeight) / pooledTotal)
			shared = poolWeight
			direct = np.zeros_like(mass)
			for source, probability in zip(sources, probabilities):
				targets, strengths, total = index.row(int(source))
				if(total > inferenceBeamSearchZero):
					addObservedMass(direct, start, end, targets, strengths, (inferenceBeamSearchOne - poolWeight) * probability / total)
				else:
					shared += (inferenceBeamSearchOne - poolWeight) * probability
			mass = direct + shared * mass
		# Equal terminal cells have identical future transition laws. Sum their histories before pruning.
		positive = np.flatnonzero(mass > inferenceBeamSearchZero)
		if(positive.size > width):
			cutoff = np.partition(mass[positive], -width)[-width]
			above = positive[mass[positive] > cutoff]
			ties = positive[mass[positive] == cutoff]
			positive = np.sort(np.concatenate((above, ties[:width - above.size])))
		weights = mass[positive]
		if(weights.size):
			weights /= weights.sum()
		result = token, positive + start, weights
	return result


def addObservedMass(mass, start, end, targets, strengths, scale):
	if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
		left, right = targets.searchsorted((start, end))
		mass[targets[left:right] - start] += strengths[left:right].astype(np.float64, copy=False) * scale
	return


def outgoing(index, posterior):
	if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
		_, sources, probabilities = posterior
		parts, weights = [], []
		for source, probability in zip(sources, probabilities):
			targets, strengths, total = index.row(int(source))
			if(total > inferenceBeamSearchZero):
				parts.append(targets)
				weights.append(strengths.astype(np.float64, copy=False) * (probability / total))
		cells, masses = sumByCell(parts, weights)
	return cells, masses


def applyDendriticEvidence(index, cells, masses):
	if((inferenceBeamSearchObservedSources and inferenceBeamSearchObservedSourceContext) or inferenceUpgradeObservedSources):
		database = index.database
		if(database.columnLikelihoodPosition != database.contextLikelihoodPosition or database.columnLikelihoodColumn is None):
			raise RuntimeError(inferenceBeamSearchInvalidState)
		packed = index.keys[cells]
		tokens, columns = np.divmod(packed // inferenceBeamSearchTwo, database.c)
		features = np.where(packed % inferenceBeamSearchTwo, featureIndexPrimeConceptNeuron, tokens)
		physical = columns * database.f + features
		order = np.argsort(physical)
		keys = physical[order]
		frequencies = index.counts[cells[order]]
		base = frequencies / frequencies.sum()
		stride = database.c * database.f
		local = np.zeros_like(base)
		# The beam supplies the immediate evidence. Start at the preceding dendritic frame.
		for offset, frame in enumerate(reversed(database.contextLikelihoodHistory)):
			if(offset > inferenceBeamSearchZero and frame is not None):
				addSegmentEvidence(local, keys, base, frame, arrayIndexSegmentSoma - offset, stride, inferenceContextLikelihoodSmoothing)
		older = np.zeros_like(base)
		frameCount = inferenceBeamSearchZero
		boundary = database.columnLikelihoodColumn * database.f
		left, right = keys.searchsorted((boundary, boundary + database.f))
		for epoch, position, frameKeys, weights in database.columnLikelihoodHistory:
			if(database.columnLikelihoodPosition - position >= inferenceContextLikelihoodHistoryLength and frameKeys.numel() > inferenceBeamSearchZero):
				segment = arrayIndexSegmentLastColumn - (database.columnLikelihoodEpoch - epoch)
				frame = frameKeys, weights
				addSegmentEvidence(older[left:right], keys[left:right], base[left:right], frame, segment, stride, inferenceBeamSearchOne)
				addSegmentEvidence(older[:left], keys[:left], base[:left], frame, segment - inferenceBeamSearchOne, stride, inferenceBeamSearchOne)
				addSegmentEvidence(older[right:], keys[right:], base[right:], frame, segment - inferenceBeamSearchOne, stride, inferenceBeamSearchOne)
				frameCount += inferenceBeamSearchOne
		# The native context probability normalisers are constant across the candidate cells and cancel.
		evidence = inferenceContextLikelihoodContextWeight * local + inferenceColumnLikelihoodWeight * older / max(frameCount, inferenceBeamSearchOne)
		if(inferenceLikelihoodCalibration):
			evidence *= inferenceLikelihoodCalibrationExponent
		logMass = np.log(masses)
		logMass[order] += evidence
		result = np.exp(logMass - logMass.max())
	return result


def addSegmentEvidence(result, keys, base, frame, segment, stride, smoothing):
	if((inferenceBeamSearchObservedSources and inferenceBeamSearchObservedSourceContext) or inferenceUpgradeObservedSources):
		if(keys.size and segment >= inferenceBeamSearchZero):
			frameKeys, weights = frame
			offset = segment * stride
			frameKeysCPU = frameKeys.cpu().numpy()
			left, right = frameKeysCPU.searchsorted((offset + keys[inferenceBeamSearchZero], offset + keys[-inferenceBeamSearchOne] + inferenceBeamSearchOne))
			if(left < right):
				available = frameKeysCPU[left:right] - offset
				values = weights[left:right].cpu().numpy()
				# Probe the smaller sorted set. Every stored key identifies exactly one physical candidate.
				if(available.size < keys.size):
					candidates = keys.searchsorted(available)
					matches = keys[candidates] == available
					candidates, values = candidates[matches], values[matches]
				else:
					locations = available.searchsorted(keys)
					candidates = np.flatnonzero(locations < available.size)
					locations = locations[candidates]
					matches = available[locations] == keys[candidates]
					candidates, values = candidates[matches], values[locations[matches]]
				result[candidates] += np.log1p(values / (smoothing * base[candidates]))
	return


def winner(index, cells, masses):
	if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
		# Cell ordinals are sorted by emitted token. Ties select the lowest token, then the lowest cell key.
		tokens = index.keys[cells] // (inferenceBeamSearchTwo * index.database.c)
		starts = np.r_[inferenceBeamSearchZero, np.flatnonzero(tokens[inferenceBeamSearchOne:] != tokens[:-inferenceBeamSearchOne]) + inferenceBeamSearchOne]
		best = int(np.argmax(np.add.reduceat(masses, starts)))
		start = starts[best]
		end = starts[best + inferenceBeamSearchOne] if best + inferenceBeamSearchOne < starts.size else cells.size
		result = index.decode(int(cells[start + np.argmax(masses[start:end])]))
	return result


def sumByCell(parts, weights):
	if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
		if(parts):
			keys = np.concatenate(parts)
			order = np.argsort(keys, kind=inferenceBeamSearchObservedSourceSort)
			keys = keys[order]
			starts = np.r_[inferenceBeamSearchZero, np.flatnonzero(keys[inferenceBeamSearchOne:] != keys[:-inferenceBeamSearchOne]) + inferenceBeamSearchOne]
			result = keys[starts], np.add.reduceat(np.concatenate(weights)[order], starts, dtype=np.float64)
		else:
			result = np.empty(inferenceBeamSearchZero, dtype=np.int64), np.empty(inferenceBeamSearchZero, dtype=np.float64)
	return result


class SourceIndex:
	"""Cell ordinals sort first by emitted subword, then by column and prime status."""
	def __init__(self, database):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			if(any(type(size) is not int or size <= inferenceBeamSearchZero for size in (inferenceBeamSearchObservedSourceRowCache, inferenceBeamSearchObservedSourcePoolCache))):
				raise ValueError(inferenceBeamSearchInvalidParameter)
			if(inferenceBeamSearchObservedSources):
				import GIAANNcmn_predictionBeamSearchLikelihood as beam
				beam.validateConfiguration()
				if(inferenceBeamDepth != inferenceBeamSearchOne or inferenceBeamSearchRecordDiagnostics or type(inferenceBeamSearchObservedSourceContext) is not bool):
					raise ValueError(inferenceBeamSearchInvalidParameter)
			self.database = database
			if(database.f != database.c + inferenceBeamSearchOne or any(database.conceptFeaturesDict[word] != col + inferenceBeamSearchOne for col, word in enumerate(database.conceptColumnsList))):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			self.mapped = inferenceBeamSearchObservedSourceIndexDirectory is not None
			if(self.mapped):
				self.loadMapped(Path(inferenceBeamSearchObservedSourceIndexDirectory))
			else:
				self.loadNative()
			self.total = self.counts.sum(dtype=np.float64)
			if(not np.isfinite(self.total) or self.total <= inferenceBeamSearchZero or np.any(self.counts <= inferenceBeamSearchZero) or np.any(self.keys[inferenceBeamSearchOne:] <= self.keys[:-inferenceBeamSearchOne])):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			self.row = lru_cache(maxsize=inferenceBeamSearchObservedSourceRowCache)(self.readRow)
			self.pool = lru_cache(maxsize=inferenceBeamSearchObservedSourcePoolCache)(self.readPool)
		return

	def loadMapped(self, directory):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			digests = [hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in (conceptColumnsDictFile, conceptFeaturesDictFile)]
			expected = dict(zip(inferenceBeamSearchObservedSourceManifestKeys, (inferenceBeamSearchObservedSourceIndexVersion, str(Path(databaseFolder).resolve()), self.database.c, self.database.f, arrayIndexSegmentSoma, digests)))
			if(json.loads((directory / inferenceBeamSearchObservedSourceIndexFile).read_text()) != expected):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			for name, dtype in zip(inferenceBeamSearchObservedSourceArrays, (np.integer, np.floating, np.integer, np.integer, np.integer, np.floating, np.integer, np.integer, np.floating), strict=True):
				array = np.asarray(np.load(directory / (name + inferenceBeamSearchObservedSourceArraySuffix), mmap_mode=inferenceBeamSearchObservedSourceMapMode, allow_pickle=False))
				if(array.ndim != inferenceBeamSearchOne or not np.issubdtype(array.dtype, dtype)):
					raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
				setattr(self, name, array)
			if(self.counts.shape != self.keys.shape or self.lex.size != self.database.f + inferenceBeamSearchOne or self.rows.size != self.keys.size + inferenceBeamSearchOne or self.targets.shape != self.strengths.shape or self.pooledRows.shape != self.lex.shape or self.pooledTargets.shape != self.pooledStrengths.shape):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			for offsets, size in ((self.lex, self.keys.size), (self.rows, self.targets.size), (self.pooledRows, self.pooledTargets.size)):
				if(offsets[inferenceBeamSearchZero] != inferenceBeamSearchZero or offsets[-inferenceBeamSearchOne] != size or np.any(offsets[inferenceBeamSearchOne:] < offsets[:-inferenceBeamSearchOne])):
					raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			wordBoundaries = np.arange(self.database.f + inferenceBeamSearchOne, dtype=np.int64) * (inferenceBeamSearchTwo * self.database.c)
			if(not np.array_equal(self.lex, self.keys.searchsorted(wordBoundaries))):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
		return

	def loadNative(self):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			keys, counts = [], []
			for column in files.listPersistedObservedColumnConceptIndices():
				features, frequencies = likelihood.loadContextLikelihoodColumnFrequencies(self.database, column)
				frequencies = frequencies.cpu().numpy()
				active = frequencies > inferenceBeamSearchZero
				keys.append(self.encode(column, features.cpu().numpy()[active]))
				counts.append(frequencies[active])
			if(not keys):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			physical = np.concatenate(keys)
			order = np.argsort(physical)
			self.keys = physical[order]
			self.counts = np.concatenate(counts)[order]
			self.lex = self.keys.searchsorted(np.arange(self.database.f + inferenceBeamSearchOne, dtype=np.int64) * (inferenceBeamSearchTwo * self.database.c))
		return

	def encode(self, columns, features):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			prime = features == featureIndexPrimeConceptNeuron
			tokens = np.where(prime, columns + inferenceBeamSearchOne, features)
			result = (tokens * self.database.c + columns) * inferenceBeamSearchTwo + prime
		return result

	def decode(self, source):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			if(not isinstance(source, (int, np.integer)) or not inferenceBeamSearchZero <= source < self.keys.size):
				raise ValueError(inferenceBeamSearchInvalidParameter)
			value, prime = divmod(int(self.keys[source]), inferenceBeamSearchTwo)
			token, column = divmod(value, self.database.c)
			result = column, featureIndexPrimeConceptNeuron if prime else token
		return result

	def readRow(self, source):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			if(not isinstance(source, (int, np.integer)) or not inferenceBeamSearchZero <= source < self.keys.size):
				raise ValueError(inferenceBeamSearchInvalidParameter)
			if(self.mapped):
				start, end = self.rows[source:source + inferenceBeamSearchTwo]
				targets, strengths = self.targets[start:end], self.strengths[start:end]
			else:
				targets, strengths = self.nativeRow(source)
			result = self.distribution(targets, strengths)
		return result

	def nativeRow(self, source):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			tensor = self.loadSourceTensor(source)
			if(tensor is not None):
				prop, branch, segment, columns, features = tensor.indices().cpu().numpy()
				values = tensor.values().cpu().numpy()
				mask = (prop == self.database.arrayIndexPropertiesStrengthIndex) & (branch == inferenceBeamSearchZero) & (segment == arrayIndexSegmentSoma)
				if(not np.isfinite(values[mask]).all() or np.any(values[mask] < inferenceBeamSearchZero)):
					raise RuntimeError(inferenceBeamSearchInvalidState)
				mask &= values > inferenceBeamSearchZero
				keys = self.encode(columns[mask], features[mask])
				targets = self.keys.searchsorted(keys)
				if(np.any(targets >= self.keys.size) or not np.array_equal(self.keys[targets], keys)):
					raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
				order = np.argsort(targets)
				result = targets[order], values[mask][order]
			else:
				result = sumByCell([], [])
		return result

	def loadSourceTensor(self, source):
		tensor = None
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			column, feature = self.decode(source)
			path = Path(files.getObservedColumnFeatureConnectionsFolder(column)) / (files.getObservedColumnSourceFeatureConnectionsFileBaseName(feature) + pytorchTensorFileExtension)
			if(inferenceUpgradeObservedSources and trainVerifyConnectionNonexistentAcrossBranchesOptimisation2):
				present = feature in files.listObservedColumnSourceFeatureIndices(column)
			else:
				present = path.is_file()
			if(present):
				tensor = files.loadObservedColumnSourceFeatureConnectionsTensor(self.database, column, feature, inferenceContextLikelihoodDevice).coalesce()
		return tensor

	def readPool(self, token):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			if(not isinstance(token, (int, np.integer)) or not inferenceBeamSearchZero < token < self.database.f):
				raise ValueError(inferenceBeamSearchInvalidParameter)
			if(self.mapped):
				start, end = self.pooledRows[token:token + inferenceBeamSearchTwo]
				targets, strengths = self.pooledTargets[start:end], self.pooledStrengths[start:end]
			else:
				parts, weights = [], []
				for source in range(*self.lex[token:token + inferenceBeamSearchTwo]):
					targets, strengths, total = self.row(source)
					if(total > inferenceBeamSearchZero):
						parts.append(targets)
						weights.append(strengths)
				targets, strengths = sumByCell(parts, weights)
			result = self.distribution(targets, strengths)
		return result

	def distribution(self, targets, strengths):
		if(inferenceBeamSearchObservedSources or inferenceUpgradeObservedSources):
			total = strengths.sum(dtype=np.float64)
			if(self.mapped and targets.size and (targets[inferenceBeamSearchZero] < inferenceBeamSearchZero or targets[-inferenceBeamSearchOne] >= self.keys.size or not np.isfinite(total) or np.any(strengths <= inferenceBeamSearchZero) or np.any(targets[inferenceBeamSearchOne:] <= targets[:-inferenceBeamSearchOne]))):
				raise RuntimeError(inferenceBeamSearchObservedSourceInvalidIndex)
			result = targets, strengths, total
		return result
