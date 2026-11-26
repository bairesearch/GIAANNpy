"""GIAANNcmn_eightLagInputs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common eight Lag Inputs

Learn single-token afferents on lag-specific compartments of physical target cells.

"""


from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import numpy as np
from GIAANNcmn_globalDefs import *
from GIAANNcmn_sharedInputBranches import trainingWordsAndCells


def initialise(database, loaded):
	if(trainUpgradeLearnEightLagInputs):
		if(not useModalityNLP or not tokeniserSubword or not useDedicatedFeatureListsSubword or database.f != database.c + upgradeOne):
			raise ValueError(upgradeInvalidConfiguration)
		if(upgradeInputPackedDirectory is not None):
			if(not database.inferenceMode or not loaded):
				raise ValueError(upgradeInputInvalidStore)
			store = PackedInputs(database, Path(upgradeInputPackedDirectory))
		else:
			store = DatabaseInputs(database, loaded)
		database.eightLagInputs = store
		if(database.inferenceMode):
			store.row = lru_cache(maxsize=upgradeInputRowCache)(store.readRow)
			store.total = store.frequencies.sum(dtype=np.float64)
			if(not np.isfinite(store.total) or store.total <= upgradeZero):
				raise ValueError(upgradeInputInvalidStore)
	return


def learnSequence(sequence, tokens):
	if(trainUpgradeLearnEightLagInputs):
		database = sequence.databaseNetworkObject
		words, cells = trainingWordsAndCells(sequence, tokens)
		contacts = Counter()
		for lag in range(upgradeOne, min(len(cells), upgradeInputLags + upgradeOne)):
			contacts.update((lag, source, target) for source, target in zip(words[:-lag], cells[lag:], strict=True))
		connection = database.eightLagInputs.connection
		if(not connection.in_transaction):
			connection.execute(upgradeInputBegin)
		connection.execute(upgradeInputSavepoint)
		try:
			connection.executemany(upgradeInputAddCell, Counter(cells).items())
			connection.executemany(upgradeInputAddWord, Counter(words).items())
			connection.executemany(upgradeInputAddContact, ((*contact, count) for contact, count in contacts.items()))
		except Exception:
			connection.execute(upgradeInputRollback)
			raise
		finally:
			connection.execute(upgradeInputRelease)
	return


def response(database, history):
	if(trainUpgradeLearnEightLagInputs):
		import GIAANNcmn_predictionUpgrades as readout
		store = database.eightLagInputs
		if(not upgradeZero < len(history) <= upgradeInputLags or any(not upgradeZero < word < database.f for word in history)):
			raise ValueError(upgradeInputInvalidHistory)
		rows, proposals = [], []
		# The benchmark uses one common nomination budget for the ordered and reversed lag controls.
		for offset, word in enumerate(reversed(history)):
			row = store.row(offset + upgradeOne, word)
			rows.append(row)
			width = upgradeInputProposals // upgradeInputLags if offset else upgradeInputProposals
			proposals.append(row[upgradeTwo][:width])
			if(offset):
				reverseLag = upgradeInputLags - offset + upgradeOne
				proposals.append(store.row(reverseLag, word)[upgradeTwo][:width])
		candidates = np.unique(np.concatenate(proposals))
		immediate = contactWeights(rows[upgradeZero], candidates)
		positive = immediate > upgradeZero
		candidates = candidates[positive]
		result = readout.TokenScores()
		if(candidates.size):
			physical, occupancy = store.cellInformation(candidates)
			evidence = np.zeros(len(candidates))
			for row, word in zip(rows[upgradeOne:], list(reversed(history))[upgradeOne:], strict=True):
				prior = store.frequencies[word] / store.total
				offset = upgradeInputPrior + upgradeInputSupport * prior
				evidence += np.log((contactWeights(row, candidates) + offset) / (upgradeInputPrior + (occupancy + upgradeInputSupport) * prior))
			mass = immediate[positive] * np.exp(upgradeInputEvidenceExponent * evidence)
			result = readout.strongestTokenScores(readout.fromCells(database, physical, mass).normalise(), upgradeInputTokenLimit)
	return result


def contactWeights(row, candidates):
	if(trainUpgradeLearnEightLagInputs):
		keys, weights, nominees = row
		result = np.zeros(len(candidates))
		if(keys.size and candidates.size):
			positions = np.minimum(keys.searchsorted(candidates), keys.size - upgradeOne)
			matches = keys[positions] == candidates
			result[matches] = weights[positions[matches]]
	return result


def validateInputRow(lag, word, features):
	if(trainUpgradeLearnEightLagInputs):
		if(not isinstance(lag, (int, np.integer)) or not upgradeOne <= lag <= upgradeInputLags or not isinstance(word, (int, np.integer)) or not upgradeZero < word < features):
			raise ValueError(upgradeInvalidSource)
	return


def rankedRow(keys, weights):
	if(trainUpgradeLearnEightLagInputs):
		# Primary keys are ascending physical targets, also the tie break in the frozen packed store.
		order = np.argsort(-weights.astype(np.int64), kind=upgradeStableSort)[:upgradeInputProposals]
	return keys, weights, keys[order]


class DatabaseInputs:
	def __init__(self, database, loaded):
		if(trainUpgradeLearnEightLagInputs):
			self.features = database.f
			path = (Path(databaseFolder) / upgradeInputStorageName).resolve()
			persist = useSaveData and not (debugDisableDatabaseFileWrite and storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam)
			metadata = (upgradeInputStorageVersion, database.c, database.f, upgradeInputLags, int(trainUpgradeWithObservableNounAnchors))
			if(loaded):
				if(not path.is_file()):
					raise RuntimeError(upgradeInputInvalidStore)
				readOnly = database.inferenceMode or not persist
				connection = sqlite3.connect(path.as_uri() + (upgradeInputReadOnly if readOnly else upgradeInputReadWrite), uri=True)
				if(connection.execute(upgradeInputReadMetadata).fetchall() != [metadata]):
					connection.close()
					raise RuntimeError(upgradeInputInvalidStore)
				if(not database.inferenceMode and not persist):
					memory = sqlite3.connect(upgradeInputMemoryDatabase)
					connection.backup(memory)
					connection.close()
					connection = memory
			else:
				if(database.inferenceMode or (persist and path.exists())):
					raise RuntimeError(upgradeInputInvalidStore)
				if(persist):
					path.parent.mkdir(parents=True, exist_ok=True)
				connection = sqlite3.connect(path if persist else upgradeInputMemoryDatabase)
				with connection:
					connection.execute(upgradeInputBegin)
					for statement in (upgradeInputMetadataSchema, upgradeInputCellsSchema, upgradeInputWordsSchema, upgradeInputContactsSchema):
						connection.execute(statement)
					connection.execute(upgradeInputInsertMetadata, metadata)
			self.connection = connection
			if(database.inferenceMode):
				self.frequencies = np.zeros(database.f)
				for word, count in connection.execute(upgradeInputReadWords):
					if(not upgradeZero < word < database.f or count <= upgradeZero):
						raise ValueError(upgradeInputInvalidStore)
					self.frequencies[word] = count
		return

	def readRow(self, lag, word):
		if(trainUpgradeLearnEightLagInputs):
			validateInputRow(lag, word, self.features)
			records = np.asarray(self.connection.execute(upgradeInputReadRow, (lag, word, upgradeInputMinimumCount)).fetchall(), dtype=np.int64).reshape(-upgradeOne, upgradeTwo)
			result = rankedRow(*records.T)
		return result

	def cellInformation(self, candidates):
		if(trainUpgradeLearnEightLagInputs):
			counts = np.empty(candidates.size, dtype=np.float64)
			for start in range(upgradeZero, candidates.size, upgradeInputQueryBatch):
				batch = candidates[start:start + upgradeInputQueryBatch].tolist()
				query = upgradeInputReadCells.format(upgradeInputSQLSeparator.join([upgradeInputSQLParameter] * len(batch)))
				found = np.asarray(self.connection.execute(query, batch).fetchall(), dtype=np.int64).reshape(-upgradeOne, upgradeTwo)
				if(not np.array_equal(found[:, upgradeZero], batch) or np.any(found[:, upgradeOne] <= upgradeZero)):
					raise RuntimeError(upgradeInputInvalidStore)
				counts[start:start + len(batch)] = found[:, upgradeOne]
		return candidates, counts

	def close(self):
		if(trainUpgradeLearnEightLagInputs):
			self.connection.close()
		return


class PackedInputs:
	"""Read the frozen, repeated-contact physical bank without importing the research programs."""
	def __init__(self, database, directory):
		if(trainUpgradeLearnEightLagInputs):
			with (directory / upgradeInputTrainingFile).open() as metadata:
				training = json.load(metadata)
			# The original packed format predates the ownership field and used ordinary training columns.
			if(training[upgradeInputPackedSequenceCount] != trainMaxSequences or training.get(upgradeInputPackedOwnership, False) != trainUpgradeWithObservableNounAnchors):
				raise RuntimeError(upgradeInputInvalidStore)
			with (directory / upgradeInputReadyFile).open() as metadata:
				manifest = json.load(metadata)
			self.frequencies = np.load(directory / upgradeInputFrequenciesFile, mmap_mode=upgradeInputReadMode)
			cellType = [(upgradeInputCellAddress, upgradeInputUInt64), (upgradeInputCellCount, upgradeInputUInt32)]
			self.cells = np.memmap(directory / upgradeInputCellsFile, mode=upgradeInputReadMode, dtype=cellType)
			if(self.frequencies.shape != (database.f,) or np.any(self.frequencies < upgradeZero) or self.frequencies.sum() != manifest[upgradeInputPackedEvents] or self.cells.size != manifest[upgradeInputPackedTargets] or training[upgradeInputPackedObservations] != manifest[upgradeInputPackedEvents] or manifest[upgradeInputPackedLags] != upgradeInputLags or self.cells[-upgradeOne][upgradeInputCellAddress] >= database.c * database.f):
				raise RuntimeError(upgradeInputInvalidStore)
			self.lags = []
			for lag in range(upgradeOne, upgradeInputLags + upgradeOne):
				stem = directory / upgradeInputLagFile.format(lag)
				offsets = np.memmap(stem.with_suffix(upgradeInputRowSuffix), mode=upgradeInputReadMode, dtype=upgradeInputUInt64)
				contacts = np.memmap(stem.with_suffix(upgradeInputContactsSuffix), mode=upgradeInputReadMode, dtype=upgradeInputUInt32).reshape(-upgradeOne, upgradeTwo)
				if(offsets.size != database.f + upgradeOne or offsets[upgradeZero] != upgradeZero or offsets[-upgradeOne] != len(contacts) or np.any(offsets[upgradeOne:] < offsets[:-upgradeOne])):
					raise RuntimeError(upgradeInputInvalidStore)
				ranks = np.memmap(stem.with_suffix(upgradeInputRankOffsetsSuffix), mode=upgradeInputReadMode, dtype=upgradeInputUInt64)
				nominees = np.memmap(stem.with_suffix(upgradeInputNomineesSuffix), mode=upgradeInputReadMode, dtype=upgradeInputUInt32)
				if(ranks.shape != offsets.shape or ranks[upgradeZero] != upgradeZero or ranks[-upgradeOne] != nominees.size or np.any(ranks[upgradeOne:] < ranks[:-upgradeOne])):
					raise RuntimeError(upgradeInputInvalidStore)
				self.lags.append((offsets, contacts, ranks, nominees))
		return

	def readRow(self, lag, word):
		if(trainUpgradeLearnEightLagInputs):
			validateInputRow(lag, word, self.frequencies.size)
			offsets, contacts, ranks, nominees = self.lags[lag - upgradeOne]
			start, end = offsets[word:word + upgradeTwo]
			indices, weights = contacts[start:end].T
			if(indices.size and (indices[-upgradeOne] >= self.cells.size or np.any(indices[upgradeOne:] <= indices[:-upgradeOne]) or np.any(weights < upgradeInputMinimumCount))):
				raise RuntimeError(upgradeInputInvalidStore)
			first, last = ranks[word:word + upgradeTwo]
			selected = nominees[first:min(int(last), int(first) + upgradeInputProposals)]
			if(selected.size != min(len(indices), upgradeInputProposals)):
				raise RuntimeError(upgradeInputInvalidStore)
			result = indices, weights, selected
		return result

	def cellInformation(self, candidates):
		if(trainUpgradeLearnEightLagInputs):
			rows = self.cells[candidates]
		return rows[upgradeInputCellAddress].astype(np.int64), rows[upgradeInputCellCount].astype(np.float64)

	def close(self):
		if(trainUpgradeLearnEightLagInputs):
			self.row.cache_clear()
			self.lags.clear()
			del self.cells, self.frequencies
		return
