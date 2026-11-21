"""Build the PyTorch adapter beside the native library for training and worker snapshots."""
from pathlib import Path
import os
import shutil
import sysconfig
import tempfile

from torch.utils.cpp_extension import load


def main():
	accelerator = Path(__file__).resolve().parent
	proto = accelerator.parent / "proto"
	name = "giaann_gpu_train_commit"
	build = accelerator / "build"
	module = load(name=name, sources=[str(proto / "GIAANNcmn_gpuTrainCommit.cpp")], build_directory=str(build), extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp", "-L"+str(build), "-Wl,-rpath,'$$ORIGIN'", "-l:libgiaann_coo.so"], with_cuda=False, verbose=False)
	dest = proto / (name+sysconfig.get_config_var("EXT_SUFFIX"))
	with tempfile.NamedTemporaryFile(dir=proto, prefix=name, suffix=".tmp", delete=False) as handle:
		tmp = Path(handle.name)
	try:
		shutil.copy2(module.__file__, tmp)
		os.replace(tmp, dest)
	finally:
		tmp.unlink(missing_ok=True)
	return


if(__name__ == "__main__"):
	main()
