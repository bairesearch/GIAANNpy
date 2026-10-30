"""Temporarily borrow the preparation pool's allocated CPUs for an ordered commit."""

from GIAANNcmn_globalDefs import *

if(optimiseTrainCommitThreads):
	import os
	from pathlib import Path


def begin(database):
	result = None
	if(optimiseTrainCommitThreads):
		budget = parallelisation3TrainingThreads + parallelisation3WorkerCount
		cpus = database.trainCommitCPUs
		if(cpus is None):
			# Explicit flushes outside a preparation pool use the same configured CPU budget.
			cpus = tuple(sorted(os.sched_getaffinity(trainSequenceBatchZero))[:budget])
		if(type(budget) is not int or budget < trainSequenceBatchOne or len(cpus) != budget or len(set(cpus)) != budget or any(type(cpu) is not int or cpu < trainSequenceBatchZero for cpu in cpus)):
			raise RuntimeError(trainCommitThreadsInvalidState)
		threads = pt.get_num_threads()
		mainAffinity = os.sched_getaffinity(trainSequenceBatchZero)
		affinities = {int(path.name):os.sched_getaffinity(int(path.name)) for path in Path(trainCommitThreadDirectory).iterdir()}
		result = (threads, mainAffinity, affinities)
		try:
			for thread in affinities:
				os.sched_setaffinity(thread, cpus)
			pt.set_num_threads(budget)
		except BaseException:
			restore(result)
			raise
	return result


def restore(saved):
	if(optimiseTrainCommitThreads):
		threads, mainAffinity, affinities = saved
		pt.set_num_threads(threads)
		for path in Path(trainCommitThreadDirectory).iterdir():
			thread = int(path.name)
			# Threads created during the commit inherit the owner's original CPU allocation.
			os.sched_setaffinity(thread, affinities.get(thread, mainAffinity))
	return
