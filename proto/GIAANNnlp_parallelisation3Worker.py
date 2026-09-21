"""Fresh-interpreter entry point for the optional CPU article workers."""

import contextlib
import sys

if(__name__ == "__main__"):
	with contextlib.redirect_stdout(sys.stderr):
		import GIAANNcmn_globalDefs as defs
		if(not defs.optimiseParallelisation3b):
			raise RuntimeError("Article worker requires optimiseParallelisation3b")
		if(defs.optimiseParallelisation3b):
			import GIAANNnlp_parallelisation3
			if(len(sys.argv) != 3):
				raise RuntimeError("Article worker requires its private socket descriptor and CPU index")
			GIAANNnlp_parallelisation3.runWorker(int(sys.argv[1]), int(sys.argv[2]))
