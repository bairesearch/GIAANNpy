"""Batch source preparation; no historical weights are read by fused sequence construction."""

from GIAANNcmn_globalDefs import *


def queue(columns, required):
	if(optimiseTrainDeferredSourcePreparation):
		import GIAANNcmn_trainSequenceBatch
		if(required is None):
			raise RuntimeError(trainDeferredPreparationInvalidInput)
		validated = []
		database = None
		for column in columns.values():
			concept = column.conceptIndex
			if(database is None):
				database = column.databaseNetworkObject
			if(column.databaseNetworkObject is not database or not column.isDatabaseResidentConnectionStorage() or concept < trainSequenceBatchZero or concept >= database.c or concept not in required or required[concept] is None or not required[concept]):
				raise RuntimeError(trainDeferredPreparationInvalidInput)
			features = set(required[concept])
			if(any(type(feature) is not int or feature < trainSequenceBatchZero or feature >= database.f for feature in features)):
				raise RuntimeError(trainDeferredPreparationInvalidInput)
			state = GIAANNcmn_trainSequenceBatch.getState(database)
			if(concept in state.deferredSourcePreparation and state.deferredSourcePreparation[concept][trainSequenceBatchZero] is not column):
				raise RuntimeError(trainDeferredPreparationInvalidInput)
			validated.append((state, concept, column, features))
		for state, concept, column, features in validated:
			if(concept not in state.deferredSourcePreparation):
				state.deferredSourcePreparation[concept] = (column, set(), set())
			storedColumn, union, last = state.deferredSourcePreparation[concept]
			union.update(features)
			last.clear()
			last.update(features)
	return


def flushWithoutSequences(database):
	if(optimiseTrainDeferredSourcePreparation):
		import GIAANNcmn_trainSequenceBatch
		state = database.trainSequenceBatchState
		if(not state.committing and state.sequenceCount == trainSequenceBatchZero and state.deferredSourcePreparation):
			state = GIAANNcmn_trainSequenceBatch.getState(database)
			state.committing = True
			try:
				prepare(state)
			except BaseException:
				state.failed = True
				raise
			finally:
				state.committing = False
	return


def prepare(state):
	if(optimiseTrainDeferredSourcePreparation):
		if(not state.committing or state.failed):
			raise RuntimeError(trainSequenceBatchInvalidState)
		for column, union, last in state.deferredSourcePreparation.values():
			column.prepareRequiredSourceFeatureConnectionsTrain(union, deviceSparse, createMissing=False)
			column.setTrainPreparedSourceFeatureIndices(last)
		state.deferredSourcePreparation.clear()
	return
