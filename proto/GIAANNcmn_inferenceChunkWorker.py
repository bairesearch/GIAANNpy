"""GIAANNcmn_inferenceChunkWorker.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common inference Chunk Worker

Fresh read-only inference process launched by GIAANNcmn_inferenceChunks.
"""


import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from GIAANNcmn_globalDefs import *
import GIAANNcmn_globalDefs as defs
import GIAANNcmn_inferenceChunks as chunks


def main():
	if(optimisationInferenceChunks):
		started = time.perf_counter()
		if(len(sys.argv) != inferenceChunksWorkerArgumentCount):
			raise ValueError(inferenceChunksErrorJob)
		job = json.loads(Path(sys.argv[inferenceChunksOne]).read_text(encoding=inferenceChunksEncoding))
		validateInferenceChunkJob(job)
		chunks.validateInferenceChunkConfiguration()
		if(chunks.getInferenceChunkSettings() != job["settings"]):
			raise RuntimeError(inferenceChunksErrorChanged)
		sys.dont_write_bytecode = True
		installInferenceChunkWriteProtection([Path(job["database"]), Path(job["source_directory"]).parent, Path(job["snapshot_directory"])])
		import torch
		torch.set_num_threads(inferenceChunksWorkerThreads)
		torch.set_num_interop_threads(inferenceChunksWorkerThreads)
		overrides = dict(inferenceChunksOperationalOverrides)
		overrides[inferenceChunksPromptSetting] = str(Path(job["directory"]) / inferenceChunksPromptFilename)
		for moduleName in (inferenceChunksCommonModule, inferenceChunksNLPModule):
			module = sys.modules.get(moduleName)
			if(module is not None):
				for name, value in overrides.items():
					setattr(module, name, value)
		import GIAANNcmn_main
		import GIAANNcmn_prediction
		import GIAANNcmn_debug
		import GIAANNnlp_main
		recorder = InferenceChunkSequenceRecorder(GIAANNcmn_prediction, GIAANNnlp_main.processSequence, job["start"])
		GIAANNnlp_main.processSequence = recorder.processSequence
		with torch.no_grad():
			GIAANNcmn_main.mainStandard()
		metrics = {name:int(getattr(GIAANNcmn_prediction, name)) for name in inferenceChunksMetricNames}
		chunks.validateInferenceChunkCounts(metrics)
		debugMetrics = {name:int(getattr(GIAANNcmn_debug, name)) for name in inferenceChunksDebugMetricNames}
		result = dict(index=job["index"], start=job["start"], stop=job["stop"], prompt_sha256=job["prompt_sha256"], metrics=metrics, debug_metrics=debugMetrics, sequences=recorder.sequences, elapsed=time.perf_counter()-started, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		chunks.writeInferenceChunkJSON(Path(job["directory"]) / inferenceChunksResultFilename, result)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def validateInferenceChunkJob(job):
	if(optimisationInferenceChunks):
		if(any(type(job[name]) is not int or job[name] < inferenceChunksZero for name in ("index", "start", "stop")) or not job["start"] < job["stop"] <= job["start"] + inferenceChunksSequenceCount):
			raise ValueError(inferenceChunksErrorJob)
		directory = Path(job["directory"]).resolve()
		snapshot = Path(job["snapshot_directory"]).resolve()
		database = Path(job["database"]).resolve()
		if(snapshot != Path(__file__).resolve().parent or database != Path(databaseFolder).resolve() or Path(job["working_directory"]).resolve() != Path.cwd().resolve() or directory.is_relative_to(database) or directory.is_relative_to(Path(job["source_directory"]).resolve().parent) or directory.is_relative_to(snapshot)):
			raise ValueError(inferenceChunksErrorJob)
		prompt = directory / inferenceChunksPromptFilename
		data = prompt.read_bytes()
		if(hashlib.sha256(data).hexdigest() != job["prompt_sha256"] or len(data.decode(inferenceChunksEncoding).splitlines()) != job["stop"] - job["start"]):
			raise ValueError(inferenceChunksErrorJob)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


class InferenceChunkSequenceRecorder:
	def __init__(self, prediction, original, start):
		if(optimisationInferenceChunks):
			self.prediction = prediction
			self.original = original
			self.start = start
			self.sequences = []
		else:
			raise RuntimeError(inferenceChunksErrorDisabled)
		return


	def processSequence(self, *args, **kwargs):
		if(optimisationInferenceChunks):
			before = {name:int(getattr(self.prediction, name)) for name in inferenceChunksMetricNames}
			result = self.original(*args, **kwargs)
			metrics = {name:int(getattr(self.prediction, name)) - before[name] for name in inferenceChunksMetricNames}
			chunks.validateInferenceChunkCounts(metrics)
			self.sequences.append(dict(sequence=self.start + args[inferenceChunksSequenceIndexArgument], metrics=metrics))
		else:
			raise RuntimeError(inferenceChunksErrorDisabled)
		return result


def installInferenceChunkWriteProtection(protected, writable=()):
	if(optimisationInferenceChunks):
		def audit(event, arguments):
			if(optimisationInferenceChunks):
				paths = []
				if(event == inferenceChunksAuditOpenEvent and arguments[inferenceChunksAuditOpenFlagsIndex] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
					paths.append((arguments[inferenceChunksZero], None))
				elif(event in inferenceChunksAuditSinglePathEvents):
					descriptorIndex = inferenceChunksAuditSinglePathEvents[event]
					paths.append((arguments[inferenceChunksZero], arguments[descriptorIndex] if descriptorIndex is not None else None))
				elif(event in inferenceChunksAuditTwoPathEvents):
					paths.extend(((arguments[inferenceChunksZero], arguments[inferenceChunksAuditSourceDirectoryIndex]), (arguments[inferenceChunksOne], arguments[inferenceChunksAuditTargetDirectoryIndex])))
				elif(event == inferenceChunksAuditSymlinkEvent):
					paths.append((arguments[inferenceChunksOne], arguments[inferenceChunksAuditSourceDirectoryIndex]))
				for path, descriptor in paths:
					if(isinstance(path, int)):
						path = Path(inferenceChunksAuditProcDescriptors) / str(path)
					if(isinstance(path, (str, bytes, os.PathLike))):
						resolved = Path(os.fsdecode(path))
						if(not resolved.is_absolute() and isinstance(descriptor, int) and descriptor >= inferenceChunksZero):
							resolved = Path(inferenceChunksAuditProcDescriptors) / str(descriptor) / resolved
						resolved = resolved.resolve()
						if(any(resolved.is_relative_to(root) for root in protected) and not any(resolved.is_relative_to(root) for root in writable)):
							raise RuntimeError(inferenceChunksErrorWrite + str(resolved))
			return
		sys.addaudithook(audit)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


if(__name__ == "__main__"):
	main()
