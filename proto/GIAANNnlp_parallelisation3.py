"""Bounded CPU article preparation; all learning remains ordered in the parent."""

from GIAANNcmn_globalDefs import *

if(optimiseTrainParallelisation3b):
	import os
	import sys
	import time
	import pickle
	import socket
	import select
	import struct
	import subprocess
	import traceback
	from pathlib import Path
	from queue import Queue
	from collections import deque
	from concurrent.futures import Future, ThreadPoolExecutor


def processDatasetPrefetched(databaseNetworkObject, sequenceCount, dataset):
	result = None
	if(optimiseTrainParallelisation3b):
		import GIAANNnlp_main as nlp
		validatePrefetchConfiguration(sequenceCount)
		with ArticlePreparationPool() as pool:
			articles = iter(enumerate(dataset))
			pending = deque()
			exhausted = False
			while(sequenceCount < trainMaxSequences and (pending or not exhausted)):
				while(not exhausted and len(pending) < parallelisation3PrefetchArticles):
					try:
						articleIndex, entry = next(articles)
					except StopIteration:
						exhausted = True
					except Exception as exception:
						future = Future()
						future.set_exception(exception)
						pending.append((None, future))
						exhausted = True
					else:
						# Extract text inside the job so an error in unused lookahead is never raised early.
						pending.append((articleIndex, pool.executor.submit(pool.prepare, articleIndex, entry)))
						pool.maximumPendingArticles = max(pool.maximumPendingArticles, len(pending))
				if(pending):
					articleIndex, future = pending.popleft()
					prepared, statistics = future.result(timeout=parallelisation3WorkerTimeoutSeconds)
					pool.articleCount += 1
					pool.workerCPUSeconds += statistics[0]
					pool.workerWallSeconds += statistics[1]
					sequenceCount = consumePreparedArticle(nlp, databaseNetworkObject, sequenceCount, articleIndex, prepared)
			databaseNetworkObject.parallelisation3PipelineStatistics = pool.statistics()
		result = sequenceCount
	return result


def validatePrefetchConfiguration(sequenceCount):
	if(optimiseTrainParallelisation3b):
		if(not useModalityNLP or sentencePredictions or not tokeniserSubword or executionMode not in ("train", "trainAndInference") or trainSetStartOffsetSequences != 0):
			raise RuntimeError("optimiseTrainParallelisation3b requires NLP subword training, sentencePredictions=False and trainSetStartOffsetSequences=0")
		if(useGPUdense or useGPUsparse or useGPUdatabase or not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			raise RuntimeError("optimiseTrainParallelisation3b requires CPU training with RAM-resident connections")
		if(not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity")):
			raise RuntimeError("optimiseTrainParallelisation3b requires CPU affinity support")
		for value in (parallelisation3WorkerCount, parallelisation3TrainingThreads, parallelisation3PrefetchArticles, parallelisation3WorkerThreads, parallelisation3WorkerTimeoutSeconds, parallelisation3MaximumMessageBytes):
			if(not isinstance(value, int) or isinstance(value, bool) or value <= 0):
				raise RuntimeError("optimiseTrainParallelisation3b worker, thread, queue, timeout and message limits must be positive integers")
		if(parallelisation3WorkerThreads != 1 or parallelisation3PrefetchArticles < parallelisation3WorkerCount):
			raise RuntimeError("optimiseTrainParallelisation3b requires one thread per worker and at least one queued article per worker")
		if(parallelisation3TrainingThreads + parallelisation3WorkerCount > len(os.sched_getaffinity(0))):
			raise RuntimeError("optimiseTrainParallelisation3b training and worker budgets exceed the available CPU affinity")
		if(not isinstance(sequenceCount, int) or isinstance(sequenceCount, bool) or sequenceCount < 0 or sequenceCount >= trainMaxSequences):
			raise RuntimeError("optimiseTrainParallelisation3b sequenceCount must be within the training range")
	return


def consumePreparedArticle(nlp, databaseNetworkObject, sequenceCount, articleIndex, prepared):
	if(optimiseTrainParallelisation3b):
		for sequenceIndex, sequence, sequenceRaw, sequenceWordLength, accepted, error in prepared:
			if(sequenceCount >= trainMaxSequences):
				break
			if(error is not None):
				raise RuntimeError("optimiseTrainParallelisation3b sequence preparation failed: " + error)
			if(accepted):
				nlp.processSequence(databaseNetworkObject, False, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw, sequenceWordLength=sequenceWordLength)
			else:
				nlp.updateExecutionProgressForSkippedSequence(False, sequenceCount)
			sequenceCount += 1
	return sequenceCount


class ArticlePreparationPool:
	def __init__(self):
		if(optimiseTrainParallelisation3b):
			self.processes = []
			self.connections = []
			self.available = Queue()
			self.executor = None
			self.originalAffinities = {int(path.name):os.sched_getaffinity(int(path.name)) for path in Path("/proc/self/task").iterdir()}
			self.availableCPUs = sorted(os.sched_getaffinity(0))
			self.originalThreads = pt.get_num_threads()
			self.maximumPendingArticles = 0
			self.articleCount = 0
			self.workerCPUSeconds = 0.0
			self.workerWallSeconds = 0.0
			self.startupSeconds = 0.0
			self.workerIdentities = []

	def __enter__(self):
		if(optimiseTrainParallelisation3b):
			try:
				self.start()
			except BaseException:
				self.close(False)
				raise
		return self

	def __exit__(self, exceptionType, exception, exceptionTraceback):
		if(optimiseTrainParallelisation3b):
			self.close(exceptionType is None)
		return False

	def start(self):
		if(optimiseTrainParallelisation3b):
			start = time.perf_counter()
			trainingCPUs = self.availableCPUs[:parallelisation3TrainingThreads]
			workerCPUs = self.availableCPUs[parallelisation3TrainingThreads:parallelisation3TrainingThreads+parallelisation3WorkerCount]
			setCurrentProcessAffinity(trainingCPUs)
			pt.set_num_threads(parallelisation3TrainingThreads)
			for workerCPU in workerCPUs:
				parentConnection, childConnection = socket.socketpair()
				self.connections.append(parentConnection)
				environment = dict(os.environ)
				for name in parallelisation3ThreadEnvironmentVariables:
					environment[name] = str(parallelisation3WorkerThreads)
				try:
					process = subprocess.Popen([sys.executable, "-m", parallelisation3WorkerModule, str(childConnection.fileno()), str(workerCPU)], cwd=os.path.dirname(__file__), pass_fds=(childConnection.fileno(),), stdin=subprocess.DEVNULL, stdout=sys.stderr, env=environment)
					self.processes.append(process)
				finally:
					childConnection.close()
			for index, connection in enumerate(self.connections):
				message = receiveMessage(connection)
				if(message[0] != parallelisation3WorkerReady or message[1] != self.processes[index].pid or message[2] != [workerCPUs[index]] or message[3] != (optimiseTrainParallelisation3a, optimiseTrainParallelisation3b, optimiseTrainParallelisation3c, optimiseTrainParallelisation3d)):
					raise RuntimeError("optimiseTrainParallelisation3b worker configuration/identity mismatch: " + repr(message))
				self.workerIdentities.append(message[1:])
				self.available.put(index)
			self.executor = ThreadPoolExecutor(max_workers=parallelisation3WorkerCount)
			self.startupSeconds = time.perf_counter()-start
		return

	def prepare(self, articleIndex, entry):
		result = None
		if(optimiseTrainParallelisation3b):
			import GIAANNnlp_main as nlp
			text = nlp.GIAANNnlp_datasets.getDatasetEntryText(entry, articleIndex)
			if(datasetSanitiseNullCharacters):
				text = nlp.sanitiseDatasetNullCharacters(text, articleIndex)
			index = self.available.get(timeout=parallelisation3WorkerTimeoutSeconds)
			try:
				sendMessage(self.connections[index], (articleIndex, text))
				message = receiveMessage(self.connections[index])
				if(message[0] == parallelisation3WorkerError):
					raise RuntimeError("optimiseTrainParallelisation3b article preparation failed: " + message[1])
				if(message[0] != parallelisation3WorkerResult or message[1] != articleIndex):
					raise RuntimeError("optimiseTrainParallelisation3b worker article order/protocol mismatch")
				result = (message[2], message[3])
			finally:
				self.available.put(index)
		return result

	def statistics(self):
		result = None
		if(optimiseTrainParallelisation3b):
			result = {"workers":parallelisation3WorkerCount,"training_threads":parallelisation3TrainingThreads,"maximum_pending_articles":self.maximumPendingArticles,"articles_consumed":self.articleCount,"worker_cpu_s_consumed":self.workerCPUSeconds,"worker_wall_s_consumed":self.workerWallSeconds,"startup_s":self.startupSeconds,"worker_identities":self.workerIdentities}
		return result

	def close(self, requireSuccess):
		if(optimiseTrainParallelisation3b):
			failures = []
			try:
				if(self.executor is not None):
					self.executor.shutdown(wait=True, cancel_futures=True)
				for process, connection in zip(self.processes, self.connections):
					try:
						if(process.poll() is None):
							sendMessage(connection, parallelisation3WorkerStop)
					except (BrokenPipeError, ConnectionError, OSError) as exception:
						failures.append(repr(exception))
				for connection in self.connections:
					connection.close()
				for process in self.processes:
					try:
						code = process.wait(timeout=parallelisation3WorkerTimeoutSeconds)
						if(code != 0):
							failures.append("worker exit code " + str(code))
					except subprocess.TimeoutExpired as exception:
						failures.append(repr(exception))
			finally:
				pt.set_num_threads(self.originalThreads)
				for path in Path("/proc/self/task").iterdir():
					thread = int(path.name)
					os.sched_setaffinity(thread, self.originalAffinities.get(thread, self.availableCPUs))
			if(failures and requireSuccess):
				raise RuntimeError("optimiseTrainParallelisation3b worker shutdown failed: " + repr(failures))
		return


def runWorker(connectionDescriptor, cpu):
	if(optimiseTrainParallelisation3b):
		if(not isinstance(connectionDescriptor, int) or connectionDescriptor < 0 or not isinstance(cpu, int) or cpu < 0):
			raise RuntimeError("optimiseTrainParallelisation3b invalid worker descriptor or CPU")
		setCurrentProcessAffinity([cpu])
		pt.set_num_threads(parallelisation3WorkerThreads)
		pt.set_num_interop_threads(parallelisation3WorkerThreads)
		connection = socket.socket(fileno=connectionDescriptor)
		try:
			import GIAANNnlp_main as nlp
			nlp.loadPOSdatabase()
			nlp.GIAANNnlp_sequenceTokens.getTokeniserSubwordEncoding()
			sendMessage(connection, (parallelisation3WorkerReady, os.getpid(), sorted(os.sched_getaffinity(0)), (optimiseTrainParallelisation3a, optimiseTrainParallelisation3b, optimiseTrainParallelisation3c, optimiseTrainParallelisation3d)))
			while(True):
				message = receiveWorkerCommand(connection)
				if(message == parallelisation3WorkerStop):
					break
				articleIndex, text = message
				wallStart = time.perf_counter()
				cpuStart = time.process_time()
				try:
					prepared = prepareArticle(nlp, text)
				except Exception:
					# Deliver errors at their original article position, including unused lookahead at the training limit.
					sendMessage(connection, (parallelisation3WorkerError, traceback.format_exc()))
				else:
					sendMessage(connection, (parallelisation3WorkerResult, articleIndex, prepared, (time.process_time()-cpuStart, time.perf_counter()-wallStart)))
		except Exception:
			sendMessage(connection, (parallelisation3WorkerError, traceback.format_exc()))
			raise
		finally:
			connection.close()
	return


def prepareArticle(nlp, text):
	result = None
	if(optimiseTrainParallelisation3b):
		if(ignoreNewlineCharacters):
			text = text.replace('\n', ' ')
		sequences, rawSequences = nlp.generateSeqencesBatchOrSerial(nlp.nlpArticle(text), False)
		result = []
		for sequenceIndex, (sequence, raw) in enumerate(zip(sequences, rawSequences)):
			if(sequencesCropToMaxLength and len(sequence) > maxSequenceLength):
				sequence = sequence[:maxSequenceLength]
				raw = sequence.text
			if(len(sequence) <= maxSequenceLength):
				wordLength = len(sequence)
				try:
					prepared, preparedRaw, accepted = nlp.enforceSequenceLengthTokenLimits(sequence, raw)
				except Exception:
					result.append((sequenceIndex, None, raw, wordLength, False, traceback.format_exc()))
				else:
					result.append((sequenceIndex, prepared if accepted else None, preparedRaw, wordLength, accepted, None))
	return result


def setCurrentProcessAffinity(cpus):
	if(optimiseTrainParallelisation3b):
		if(not cpus or any(not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in cpus)):
			raise RuntimeError("optimiseTrainParallelisation3b CPU list must contain non-negative integers")
		for path in Path("/proc/self/task").iterdir():
			os.sched_setaffinity(int(path.name), cpus)
	return


def sendMessage(connection, value):
	if(optimiseTrainParallelisation3b):
		payload = pickle.dumps(value, protocol=parallelisation3PickleProtocol)
		if(len(payload) > parallelisation3MaximumMessageBytes):
			raise RuntimeError("optimiseTrainParallelisation3b message exceeds parallelisation3MaximumMessageBytes")
		connection.settimeout(parallelisation3WorkerTimeoutSeconds)
		connection.sendall(struct.pack(parallelisation3MessageHeaderFormat, len(payload)))
		connection.sendall(payload)
	return


def receiveWorkerCommand(connection):
	result = None
	if(optimiseTrainParallelisation3b):
		# Training may leave workers idle indefinitely; start the transfer deadline only when a command arrives.
		connection.settimeout(None)
		poller = select.poll()
		poller.register(connection, select.POLLIN)
		poller.poll()
		# Peer closure also wakes the poll and must fail explicitly in receiveMessage.
		result = receiveMessage(connection)
	return result


def receiveMessage(connection):
	result = None
	if(optimiseTrainParallelisation3b):
		deadline = time.monotonic()+parallelisation3WorkerTimeoutSeconds
		header = receiveBytes(connection, struct.calcsize(parallelisation3MessageHeaderFormat), deadline)
		size = struct.unpack(parallelisation3MessageHeaderFormat, header)[0]
		if(size <= 0 or size > parallelisation3MaximumMessageBytes):
			raise RuntimeError("optimiseTrainParallelisation3b invalid message size")
		result = pickle.loads(receiveBytes(connection, size, deadline))
	return result


def receiveBytes(connection, count, deadline):
	result = None
	if(optimiseTrainParallelisation3b):
		chunks = []
		remaining = count
		while(remaining):
			timeout = deadline-time.monotonic()
			if(timeout <= 0):
				raise TimeoutError("optimiseTrainParallelisation3b worker message timed out")
			connection.settimeout(timeout)
			chunk = connection.recv(remaining)
			if(not chunk):
				raise RuntimeError("optimiseTrainParallelisation3b worker connection closed before the complete result")
			chunks.append(chunk)
			remaining -= len(chunk)
		result = b"".join(chunks)
	return result
