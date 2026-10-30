"""GIAANNnlp_parallelisation3Worker.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP parallelisation 3 Worker

Fresh-interpreter entry point for the optional CPU article workers.

"""



import contextlib
import sys

if(__name__ == "__main__"):
	with contextlib.redirect_stdout(sys.stderr):
		import GIAANNcmn_globalDefs as defs
		if(not defs.optimiseTrainParallelisation3b):
			raise RuntimeError("Article worker requires optimiseTrainParallelisation3b")
		if(defs.optimiseTrainParallelisation3b):
			import GIAANNnlp_parallelisation3
			if(len(sys.argv) != 3):
				raise RuntimeError("Article worker requires its private socket descriptor and CPU index")
			GIAANNnlp_parallelisation3.runWorker(int(sys.argv[1]), int(sys.argv[2]))
			if(defs.optimiseTrainWorkerExit):
				# runWorker returned successfully and its socket is closed. Only this isolated read-only worker exits.
				import os
				sys.stdout.flush()
				sys.stderr.flush()
				os._exit(defs.trainWorkerExitSuccessCode)
