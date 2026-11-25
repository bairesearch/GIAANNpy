"""Shared lexical afferents on two compartments of a physical target branch."""

from collections import Counter
from functools import lru_cache
import sqlite3
from pathlib import Path
import numpy as np
from GIAANNcmn_globalDefs import *
import GIAANNnlp_sequenceTokens as sequenceTokens


def initialise(database, loaded):
	if(inferenceUpgradeBranchBank):
		path = (Path(databaseFolder) / upgradeBranchStorageName).resolve()
		persistTraining = useSaveData and not (debugDisableDatabaseFileWrite and storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam)
		if(not tokeniserSubword or not useDedicatedFeatureListsSubword or database.f != database.c + upgradeOne):
			raise ValueError(upgradeInvalidConfiguration)
		if(loaded and not path.is_file()):
			raise RuntimeError(upgradeInvalidBranchStore)
		if(not loaded and persistTraining and path.exists()):
			raise RuntimeError(upgradeInvalidBranchStore)
		if(loaded):
			readOnly = database.inferenceMode or not persistTraining
			connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if readOnly else "?mode=rw"), uri=True)
			if(not database.inferenceMode and not persistTraining):
				memory = sqlite3.connect(":memory:")
				connection.backup(memory)
				connection.close()
				connection = memory
		else:
			if(database.inferenceMode):
				raise RuntimeError(upgradeInvalidBranchStore)
			if(persistTraining):
				path.parent.mkdir(parents=True, exist_ok=True)
			connection = sqlite3.connect(path if persistTraining else ":memory:")
		metadata = (upgradeBranchStorageVersion, database.c, database.f, upgradeBranchCount, upgradeBranchSlots, upgradeBranchRecruitment, int(trainUpgradeWithObservableNounAnchors))
		if(not loaded):
			with connection:
				connection.execute("BEGIN")
				connection.execute("CREATE TABLE metadata(version, columns, features, branches, slots, recruitment, observableAnchors)")
				connection.execute("INSERT INTO metadata VALUES(?,?,?,?,?,?,?)", metadata)
				connection.execute("CREATE TABLE exposures(target INTEGER PRIMARY KEY, count INTEGER NOT NULL)")
				connection.execute("CREATE TABLE contacts(target INTEGER, branch INTEGER, compartment INTEGER, source INTEGER, weight INTEGER NOT NULL, PRIMARY KEY(target,branch,compartment,source)) WITHOUT ROWID")
				connection.execute("CREATE INDEX incoming ON contacts(compartment,source,target,branch,weight)")
		elif(connection.execute("SELECT * FROM metadata").fetchall() != [metadata]):
			connection.close()
			raise RuntimeError(upgradeInvalidBranchStore)
		database.sharedInputBranches = connection
		if(database.inferenceMode):
			database.sharedInputAfferents = lru_cache(maxsize=upgradeBranchAfferentCache)(lambda compartment, source: readAfferents(connection, compartment, source))
	return


def learnSequence(sequence, tokens):
	if(inferenceUpgradeLearnSharedInputDendriticBranches):
		database = sequence.databaseNetworkObject
		connection = database.sharedInputBranches
		words = [sequenceTokens.getTokeniserSubwordFeatureIndex(token) for token in tokens]
		cells = []
		for position, (column, word) in enumerate(zip(sequence.tokenConceptColumnIndexList, words, strict=True)):
			if(column is None or not upgradeZero <= column < database.c or not upgradeZero < word < database.f):
				raise ValueError(upgradeInvalidSource)
			feature = featureIndexPrimeConceptNeuron if position in sequence.columnsIndexSequenceWordIndexDict else word
			cells.append(int(column) * database.f + int(feature))
		if(not connection.in_transaction):
			connection.execute("BEGIN")
		connection.execute("SAVEPOINT learn_sequence")
		try:
			updateAfferents(connection, cells, words)
		except BaseException:
			connection.execute("ROLLBACK TO learn_sequence")
			raise
		finally:
			connection.execute("RELEASE learn_sequence")
	return


def updateAfferents(connection, cells, words):
	if(inferenceUpgradeLearnSharedInputDendriticBranches):
		increments = Counter(cells)
		exposures = Counter(dict(trainingRows(connection, upgradeBranchReadExposures, list(increments))))
		eligible = [cell for cell, count in increments.items() if exposures[cell] + count >= upgradeBranchRecruitment]
		population = {cell: [[{}, {}] for branch in range(upgradeBranchCount)] for cell in eligible}
		for cell, branch, compartment, source, weight in trainingRows(connection, upgradeBranchReadContacts, eligible):
			population[cell][branch][compartment][source] = weight

		# Repeated firings in this sequence see every earlier update, including recruitment.
		changes = Counter()
		for position, cell in enumerate(cells):
			exposures[cell] += upgradeOne
			if(position >= upgradeTwo and exposures[cell] >= upgradeBranchRecruitment):
				near, far = words[position - upgradeOne], words[position - upgradeTwo]
				branch = selectBranch(population[cell], near, far)
				if(branch is not None):
					for compartment, source in ((upgradeProximal, near), (upgradeDistal, far)):
						contacts = population[cell][branch][compartment]
						contacts[source] = contacts.get(source, upgradeZero) + upgradeOne
						changes[cell, branch, compartment, source] += upgradeOne
		connection.executemany(upgradeBranchAddExposures, increments.items())
		connection.executemany(upgradeBranchAddContacts, ((*address, count) for address, count in changes.items()))
	return


def trainingRows(connection, query, cells):
	if(inferenceUpgradeLearnSharedInputDendriticBranches):
		for offset in range(upgradeZero, len(cells), upgradeBranchQueryBatch):
			batch = cells[offset:offset + upgradeBranchQueryBatch]
			parameters = ",".join("?" for cell in batch)
			yield from connection.execute(query.format(parameters), batch)


def selectBranch(branches, near, far):
	if(inferenceUpgradeLearnSharedInputDendriticBranches):
		best = None
		bestRank = None
		for branch, (proximal, distal) in enumerate(branches):
			a, b = proximal.get(near, upgradeZero), distal.get(far, upgradeZero)
			rank = None
			if(a and b):
				rank = (upgradeTwo, min(a, b))
			elif(not proximal and not distal):
				rank = (upgradeOne, upgradeZero)
			elif((a and len(distal) < upgradeBranchSlots) or (b and len(proximal) < upgradeBranchSlots)):
				rank = (upgradeZero, max(a, b))
			if(rank is not None and (bestRank is None or rank > bestRank)):
				best, bestRank = branch, rank
	return best


def response(database, near, far):
	if(inferenceUpgradeBranchBank):
		cells = np.empty(upgradeZero, dtype=np.int64)
		weights = np.empty(upgradeZero, dtype=np.float64)
		if(near is not None and far is not None):
			proximal, nearWeight = database.sharedInputAfferents(upgradeProximal, near)
			distal, farWeight = database.sharedInputAfferents(upgradeDistal, far)
			if(proximal.size and distal.size):
				if(proximal.size > distal.size):
					proximal, distal, nearWeight, farWeight = distal, proximal, farWeight, nearWeight
				positions = distal.searchsorted(proximal).clip(max=distal.size - upgradeOne)
				coincident = distal[positions] == proximal
				targets = proximal[coincident] // upgradeBranchCount
				if(targets.size):
					# Both axons must reach the same physical branch. A soma fires at its strongest branch's drive.
					strength = np.minimum(nearWeight[coincident], farWeight[positions[coincident]])
					first = np.flatnonzero(np.r_[True, targets[upgradeOne:] != targets[:-upgradeOne]])
					cells = targets[first]
					weights = np.maximum.reduceat(strength, first)
	return cells, weights


def readAfferents(connection, compartment, source):
	if(inferenceUpgradeBranchBank):
		rows = connection.execute("SELECT target,branch,weight FROM contacts WHERE compartment=? AND source=? ORDER BY target,branch", (compartment, source)).fetchall()
		units = np.fromiter((target * upgradeBranchCount + branch for target, branch, weight in rows), dtype=np.int64, count=len(rows))
		weights = np.fromiter((weight for target, branch, weight in rows), dtype=np.float64, count=len(rows))
	return units, weights
