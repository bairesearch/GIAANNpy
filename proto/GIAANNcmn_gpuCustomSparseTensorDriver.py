"""CUDA sparse operations and shared native CUDA bindings for GIAANN."""

from GIAANNcmn_globalDefs import *

if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
	import atexit
	import ctypes as C
	import numpy as np
	from pathlib import Path
	from threading import Lock
	_workspace = C.c_void_p()
	_capacity = 0
	_lock = Lock()
	_lib = None
	_libraryLock = Lock()

if(useGPUcustomSparseTensorDriver):
	class COOView(C.Structure):
		_fields_ = [("indices", C.c_void_p), ("values", C.c_void_p), ("rowStride", C.c_int64), ("entryStride", C.c_int64), ("valueStride", C.c_int64), ("size", C.c_int)]

		def __init__(self, indices, values):
			super().__init__(indices.data_ptr(), values.data_ptr(), indices.stride(0), indices.stride(1), values.stride(0), values.numel())


def nativeLibrary():
	global _lib
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(_lib is None):
			with _libraryLock:
				if(_lib is None):
					_lib = loadNativeLibrary()
	return _lib


def loadNativeLibrary():
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		try:
			library = C.CDLL(str(Path(__file__).with_name(gpuCustomSparseTensorDriverLibrary)))
		except OSError as error:
			raise RuntimeError(gpuCustomSparseTensorDriverBuildError) from error
		library.sorter_create.argtypes = [C.POINTER(C.c_void_p), C.c_int]
		library.sorter_create.restype = C.c_char_p
		library.sorter_run.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_uint64, C.c_int, C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
		library.sorter_run.restype = C.c_char_p
		library.sorter_close.argtypes = [C.c_void_p]
		library.sorter_close.restype = None
		library.cpu_pack_checked.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
		library.cpu_pack_checked.restype = C.c_int
		library.cpu_unpack.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
		library.cpu_unpack.restype = None
		library.plan_route.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
		library.plan_route.restype = C.c_char_p
		_ptr, _int, _stride = C.c_void_p, C.c_int, C.c_int64
		library.coo_partition.argtypes = [_ptr, _int, _ptr, _int, _ptr, _int, _int]
		library.coo_partition.restype = _int
		library.coo_emit.argtypes = [_ptr, _ptr, _ptr, _stride, _ptr, _ptr, _ptr, _stride, _int, _ptr, _int, _int, _ptr, _ptr, _int]
		library.coo_emit.restype = None
		library.coo_delta_positions.argtypes = [_ptr, _stride, _int, _ptr, _stride, _int, _int, _ptr, _ptr]
		library.coo_delta_positions.restype = _int
		library.coo_delta_emit.argtypes = [_ptr, _stride, _ptr, _int, _ptr, _stride, _ptr, _int, _int, _ptr, _ptr, _int, _int, _ptr, _ptr, C.c_bool, _ptr]
		library.coo_delta_emit.restype = _int
		library.cpu_pack_rows.argtypes = [_ptr, _ptr, _ptr, _int, _int, _int, _stride]
		library.cpu_pack_rows.restype = _int
		library.coo_validate.argtypes = [_ptr, _stride, _ptr, _int, _int, _int]
		library.coo_validate.restype = _int
		library.coo_special_values.argtypes = [_ptr, _stride, _int, _int]
		library.coo_special_values.restype = _int
		library.coo_find_neuron.argtypes = [_ptr, _stride, _int, _int, _int, _stride, _stride, _ptr]
		library.coo_find_neuron.restype = _int
		library.coo_normalise.argtypes = [_ptr, _ptr, _ptr, _int, C.c_float, C.c_float, _int, C.c_float, _ptr]
		library.coo_normalise.restype = C.c_char_p
		library.coo_nonnegative.argtypes = [_ptr, _int, _int]
		library.coo_nonnegative.restype = _int
		library.coo_candidates.argtypes = [_ptr, _stride, _int, _ptr, _ptr, _int, _ptr, _int, _ptr]
		library.coo_candidates.restype = _int
		library.coo_decay_plan.argtypes = [_ptr, _ptr, _int, _stride, C.c_float, C.c_bool, _int, _ptr, _ptr, _int, _int]
		library.coo_decay_plan.restype = _int
		library.coo_decay_emit.argtypes = [_ptr, _stride, _int, _ptr, _int, _ptr, _int, _int, _int, _ptr, _ptr, _int]
		library.coo_decay_emit.restype = None
		if(useGPUcustomSparseTensorDriver):
			library.coo_inspect_strengths.argtypes = [_ptr, _stride, _stride, _ptr, _int, _ptr, _int, _ptr, _ptr]
			library.coo_inspect_strengths.restype = C.c_char_p
			library.coo_union_plan.argtypes = [COOView, COOView, _ptr, _int, _ptr, _ptr, _ptr, _ptr]
			library.coo_union_plan.restype = C.c_char_p
			library.coo_union_emit.argtypes = [COOView, COOView, _int, _ptr, _ptr, _int, _ptr, _ptr, _ptr]
			library.coo_union_emit.restype = C.c_char_p
	return library


def cudaDeviceIndex():
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(executionMode == "train" and (not pt.cuda.is_available() or gpuCustomSparseTensorDriverDeviceIndex >= pt.cuda.device_count())):
			raise RuntimeError(gpuCustomSparseTensorDriverUnavailable)
	return gpuCustomSparseTensorDriverDeviceIndex


@pt.inference_mode()
def propagateBlockSparse(weight, state, bias, steps=1):
	"""Compute repeated ReLU(weight @ state + bias) for callers with CPU BSR input."""
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(weight.layout != pt.sparse_bsr or weight.ndim != 2 or weight.size(0) != weight.size(1) or state.layout != pt.strided or bias.layout != pt.strided or state.ndim != 2 or state.size(0) != weight.size(0) or min(state.shape) < 1 or bias.shape != (weight.size(0),) or type(steps) is not int or steps < 1):
			raise ValueError(gpuCustomSparseTensorDriverInvalidBlocks)
		if(weight.dtype not in (pt.float32, pt.float16) or any(t.device.type != gpuCustomSparseTensorDriverCPU or t.dtype != weight.dtype or t.requires_grad for t in (weight, state, bias))):
			raise ValueError(gpuCustomSparseTensorDriverInvalidBlocks)
		blocks = weight.values()
		if(blocks.ndim != 3 or blocks.size(1) != blocks.size(2) or blocks.size(1) < 1 or weight.size(0) % blocks.size(1)):
			raise ValueError(gpuCustomSparseTensorDriverInvalidBlocks)
		width = blocks.size(1)
		groups = weight.size(0)//width
		ptr = weight.crow_indices().numpy()
		col = weight.col_indices().numpy()
		degree = np.diff(ptr)
		if(len(ptr) != groups+1 or ptr[0] != 0 or ptr[-1] != len(col) or len(col) != len(blocks) or np.any(degree < 0) or np.any(col < 0) or np.any(col >= groups)):
			raise ValueError(gpuCustomSparseTensorDriverInvalidBlocks)
		with pt.cuda.device(cudaDeviceIndex()):
			device = pt.device("cuda", gpuCustomSparseTensorDriverDeviceIndex)
			current = state.contiguous().reshape(groups, width, -1).to(device)
			base = bias.contiguous().reshape(groups, width, 1).to(device)
			if(np.all(degree == 1)):
				weights = blocks.to(device)
				columns = weight.col_indices().long().contiguous().to(device)
				ordered = current.index_select(0, columns)
				product = pt.empty_like(ordered)
				for step in range(steps):
					pt.bmm(weights, ordered, out=product)
					nextColumns = columns.data_ptr() if step+1 < steps else None
					error = nativeLibrary().plan_route(product.data_ptr(), base.data_ptr(), nextColumns, ordered.data_ptr(), groups, width, state.size(1), int(weight.dtype == pt.float16), pt.cuda.current_stream().cuda_stream)
					if(error):
						raise RuntimeError(error.decode())
				current = ordered
			else:
				current = propagateRaggedBlocks(blocks, ptr, col, degree, current, base, steps)
			result = current.reshape_as(state).cpu()
	else:
		raise RuntimeError(gpuCustomSparseTensorDriverDisabled)
	return result


def propagateRaggedBlocks(blocks, ptr, col, degree, current, base, steps):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		phases = []
		for slot in range(int(degree.max(initial=0))):
			rows = np.flatnonzero(degree > slot)
			entries = ptr[rows]+slot
			weights = blocks.index_select(0, pt.from_numpy(entries.astype(np.int64))).to(current.device)
			columns = pt.from_numpy(col[entries].astype(np.int64)).to(current.device)
			targets = pt.from_numpy(rows).to(current.device) if len(rows) < len(degree) else None
			phases.append((weights, columns, targets))
		for step in range(steps):
			nextState = pt.zeros_like(current)
			for slot, (weights, columns, targets) in enumerate(phases):
				ordered = current.index_select(0, columns)
				if(targets is None and slot == 0):
					pt.bmm(weights, ordered, out=nextState)
				elif(targets is None):
					nextState.baddbmm_(weights, ordered)
				else:
					nextState.index_add_(0, targets, pt.bmm(weights, ordered))
			current = nextState.add_(base).relu_()
	return current


def coalesceTensor(tensor):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(supportsTensor(tensor) and not tensor.is_coalesced() and gpuCustomSparseTensorDriverMinEntries <= tensor._nnz() <= gpuCustomSparseTensorDriverMaxEntries and (useHybridGPUCPUcustomSparseTensorDriver or math.prod(tensor.shape) <= np.iinfo(np.int64).max)):
			result = reduceCoordinates((tensor,))
			if(result is None):	#the order of duplicate FP32 additions can affect this result
				result = tensor.coalesce()
		else:
			result = tensor.coalesce()
	return result


def unitEntryStride(indices):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		result = indices if indices.stride(1) == 1 else indices.contiguous()
	return result


def gatherCanonical(tensor, queries, dtype):
	"""Read scalar values at query coordinates from canonical FP32 COO on its resident device."""
	result = None
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(isinstance(tensor, pt.Tensor) and (useHybridGPUCPUcustomSparseTensorDriver or tensor.is_cuda) and tensor.layout == pt.sparse_coo and isinstance(queries, pt.Tensor)):
			import GIAANNcmn_gpuTrainCommit as native
			stream = pt.cuda.current_stream(tensor.device).cuda_stream if tensor.is_cuda else 0
			result = native.getExtension().gather(useGPUcustomSparseTensorDriver, tensor, queries, gpuCustomSparseTensorDriverMinGatherEntries, gpuCustomSparseTensorDriverMaxEntries, gpuCustomSparseTensorDriverMaxRank, gpuCustomSparseTensorDriverPackThreads, stream)
			if(result is not None and result.dtype != dtype):
				result = result.to(dtype)
	return result


def sliceCanonical(tensor, axis, index, target):
	"""Select the contiguous range of a COO axis preceded only by singleton axes."""
	result = None
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(isinstance(tensor, pt.Tensor) and (useHybridGPUCPUcustomSparseTensorDriver or tensor.is_cuda) and tensor.layout == pt.sparse_coo and tensor.dtype == pt.float32 and tensor.is_coalesced() and not tensor.requires_grad and tensor.dense_dim() == 0 and tensor.dim() > 1 and type(axis) is int and type(index) is int and 0 <= axis < tensor.dim() and all(size == 1 for size in tensor.shape[:axis]) and tensor.device.type in (gpuCustomSparseTensorDriverCPU, gpuCustomSparseTensorDriverCUDA)):
			indices = tensor.indices()
			if(0 <= index < tensor.shape[axis] and tensor._nnz()):
				row = indices[axis]
				if(tensor.device.type == gpuCustomSparseTensorDriverCPU):
					first, end = np.searchsorted(row.detach().resolve_neg().numpy(), (index, index+1))
				else:
					bounds = pt.tensor((index, index+1), dtype=row.dtype, device=row.device)
					first, end = pt.searchsorted(row.contiguous(), bounds).tolist()
			else:
				first = end = 0
			if(axis == 0):
				selected = indices[1:, first:end].clone()
			elif(axis == tensor.dim()-1):
				selected = indices[:axis, first:end].clone()
			else:
				selected = pt.cat((indices[:axis, first:end], indices[axis+1:, first:end]), dim=0)
			values = tensor.values()[first:end].clone()
			shape = tensor.shape[:axis]+tensor.shape[axis+1:]
			if(pt.device(target).type == gpuCustomSparseTensorDriverCPU):
				result = pt.sparse_coo_tensor(selected, values, size=shape, device=target, is_coalesced=True)
			else:
				# CUDA coalesce canonicalises signed zero and NaN payloads even in unique runs.
				result = pt.sparse_coo_tensor(selected, values, size=shape, device=target).coalesce()
	return result


def canNormalise(tensor):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		result = (supportsTensor(tensor) or (useGPUcustomSparseTensorDriver and supportsDeviceTensor(tensor))) and (useGPUcustomSparseTensorDriver or tensor.is_coalesced()) and tensor.dim() == inferenceLeakyIntegrateAndFireNeuronTensorRank and tensor.size(0) == multipleDendriticBranchesNumber == 1 and tensor.size(1) == arrayNumberOfSegments and 0 < tensor.size(1) <= gpuCustomSparseTensorDriverMaxSegments and min(tensor.shape) > 0 and gpuCustomSparseTensorDriverMinEntries <= tensor._nnz() <= gpuCustomSparseTensorDriverMaxEntries and (not inferenceSynapticResponsePower or inferenceSynapticResponseExponent == inferenceSynapticResponseMaximumExponent)
	return result


def normalise(tensor, active=1.0):
	"""Transfer FP32 strengths and compute a segmented maximum and scaled output on CUDA."""
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		device = tensor.device if tensor.is_cuda else pt.device(gpuCustomSparseTensorDriverCUDA, cudaDeviceIndex())
		segments = tensor.size(inferenceLeakyIntegrateAndFireSegmentDimension)
		lastColumn = arrayIndexSegmentLastColumn if useSANIcolumns or useSANIfeaturesAndColumns else -1
		with pt.cuda.device(device):
			if(useHybridGPUCPUcustomSparseTensorDriver):
				indices = unitEntryStride(tensor.indices())
				dimensions = (C.c_uint64*tensor.dim())(*tensor.shape)
				threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
				if(nativeLibrary().coo_validate(indices.data_ptr(), indices.stride(0), dimensions, tensor.dim(), tensor._nnz(), threads)):
					raise RuntimeError(inferenceReviewPatch14InvalidConnections)
				ends = np.searchsorted(indices[inferenceLeakyIntegrateAndFireSegmentDimension].numpy(), np.arange(1, segments+1)).astype(np.int64, copy=False)
				values = tensor.values().contiguous().to(device)
			else:
				indices, values, ends = prepareStrengthsCUDA(tensor, device, segments)
			stats = pt.zeros(segments+1, dtype=pt.int32, device=device)
			error = nativeLibrary().coo_normalise(values.data_ptr(), stats.data_ptr(), ends.ctypes.data, segments, inferenceReviewPatch14ReferenceStrength, inferenceReviewPatch14ColumnInputGain, lastColumn, active, pt.cuda.current_stream().cuda_stream)
			if(error):
				raise RuntimeError(error.decode())
			resultValues = values.to(tensor.device)
			status = stats[-1].item()
			if(status & 1):
				raise RuntimeError(inferenceReviewPatch14InvalidConnections)
			if(status):
				raise RuntimeError(inferenceReviewPatch14InvalidNormalisedStrength)
		if(tensor.is_cuda):
			# Match the CUDA reference's handling of signed zeros and NaNs on reconstruction.
			result = pt.sparse_coo_tensor(indices, resultValues, size=tensor.shape, device=tensor.device).coalesce()
		else:
			result = pt.sparse_coo_tensor(indices.clone(), resultValues, size=tensor.shape, is_coalesced=True)
	return result


def prepareStrengthsCUDA(tensor, device, segments):
	if(useGPUcustomSparseTensorDriver):
		indices = tensor._indices()
		deviceIndices = indices.to(device)
		values = tensor._values().contiguous().to(device, copy=tensor.is_cuda)
		status = pt.zeros(1, dtype=pt.int32, device=device)
		shape = (C.c_int64*tensor.dim())(*tensor.shape)
		error = nativeLibrary().coo_inspect_strengths(deviceIndices.data_ptr(), deviceIndices.stride(0), deviceIndices.stride(1), values.data_ptr(), tensor._nnz(), shape, tensor.dim(), status.data_ptr(), pt.cuda.current_stream(device).cuda_stream)
		if(error):
			raise RuntimeError(error.decode())
		flags = status.item()
		if(flags & gpuCustomSparseTensorDriverInvalidCoordinates or (flags and tensor.is_coalesced())):
			raise RuntimeError(inferenceReviewPatch14InvalidConnections)
		if(flags):
			canonical = coalesceTensor(tensor)
			indices = canonical.indices()
			deviceIndices = indices.to(device)
			values = canonical.values().contiguous().to(device, copy=tensor.is_cuda)
		boundaries = pt.arange(1, segments+1, device=device)
		ends = pt.searchsorted(deviceIndices[inferenceLeakyIntegrateAndFireSegmentDimension].contiguous(), boundaries).cpu().numpy()
	return indices, values, ends


def supportsDeviceTensor(tensor):
	return tensor.is_cuda and tensor.layout == pt.sparse_coo and tensor.dtype == pt.float32 and not tensor.requires_grad and 0 < tensor.dim() <= gpuCustomSparseTensorDriverMaxRank and tensor.dense_dim() == 0


def canMerge(left, right):
	return useGPUcustomSparseTensorDriver and (supportsTensor(left) or supportsDeviceTensor(left)) and (supportsTensor(right) or supportsDeviceTensor(right)) and left.device == right.device and left.shape == right.shape and min(left._nnz(), right._nnz()) > 0 and gpuCustomSparseTensorDriverMinEntries <= left._nnz()+right._nnz() <= gpuCustomSparseTensorDriverMaxEntries


def addTensors(left, right, concatenateUpdates=False):
	result = None
	if(useGPUcustomSparseTensorDriver):
		left = coalesceTensor(left)
		right = coalesceTensor(right)
		if(canMerge(left, right)):
			result = mergeCUDA(left, right)
		if(result is None):
			if(concatenateUpdates):
				result = pt.sparse_coo_tensor(pt.cat((left.indices(), right.indices()), dim=1), pt.cat((left.values(), right.values())), size=left.shape, dtype=left.dtype, device=left.device).coalesce()
			else:
				result = (left+right).coalesce()
	return result


def mergeCUDA(left, right):
	"""Merge on the input CUDA device, or explicitly stage CPU operands to the configured GPU."""
	result = None
	if(useGPUcustomSparseTensorDriver):
		device = left.device if left.is_cuda else pt.device(gpuCustomSparseTensorDriverCUDA, cudaDeviceIndex())
		with pt.cuda.device(device):
			ai, av = left.indices().resolve_neg().to(device), left.values().resolve_neg().to(device)
			bi, bv = right.indices().resolve_neg().to(device), right.values().resolve_neg().to(device)
			a, b = COOView(ai, av), COOView(bi, bv)
			positions = pt.empty(b.size, dtype=pt.int32, device=device)
			duplicates = pt.empty_like(positions)
			status = pt.zeros(1, dtype=pt.int32, device=device)
			shape = (C.c_int64*left.dim())(*left.shape)
			stream = pt.cuda.current_stream(device).cuda_stream
			error = nativeLibrary().coo_union_plan(a, b, shape, left.dim(), positions.data_ptr(), duplicates.data_ptr(), status.data_ptr(), stream)
			if(error):
				raise RuntimeError(error.decode())
			prefix = pt.cumsum(duplicates, dim=0, dtype=pt.int32)
			flags, intersections = pt.cat((status, prefix[-1:])).tolist()
			if(flags & gpuCustomSparseTensorDriverInvalidCoordinates):
				raise ValueError(gpuCustomSparseTensorDriverInvalidKey)
			if(not flags):
				count = a.size+b.size-intersections
				indices = pt.empty((left.dim(), count), dtype=pt.long, device=device)
				values = pt.empty(count, dtype=left.dtype, device=device)
				error = nativeLibrary().coo_union_emit(a, b, left.dim(), positions.data_ptr(), prefix.data_ptr(), count, indices.data_ptr(), values.data_ptr(), stream)
				if(error):
					raise RuntimeError(error.decode())
				result = pt.sparse_coo_tensor(indices.to(left.device), values.to(left.device), size=left.shape, is_coalesced=True)
	return result


def supportsTensor(tensor):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		result = tensor.layout == pt.sparse_coo and tensor.device.type == gpuCustomSparseTensorDriverCPU and tensor.dtype == pt.float32 and not tensor.requires_grad and 0 < tensor.dim() <= gpuCustomSparseTensorDriverMaxRank and tensor.dense_dim() == 0
	return result


def reduceCoordinates(operands):
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		result = None
		shape = operands[0].shape
		rank = len(shape)
		extent = math.prod(shape)
		threads = min(pt.get_num_threads(), gpuCustomSparseTensorDriverPackThreads)
		dimensions = (C.c_uint64*rank)(*shape)
		count = sum(tensor._nnz() for tensor in operands)
		keys = pt.empty(count, dtype=pt.int64, device=gpuCustomSparseTensorDriverCPU)
		start = 0
		for tensor in operands:
			length = tensor._nnz()
			indices = tensor._indices().contiguous()
			if(nativeLibrary().cpu_pack_checked(indices.data_ptr(), keys[start:start+length].data_ptr(), dimensions, rank, length, threads)):
				raise ValueError(gpuCustomSparseTensorDriverInvalidKey)
			start += length
		values = operands[0]._values().contiguous() if len(operands) == 1 else pt.cat([tensor._values() for tensor in operands])
		outputKeys = pt.empty_like(keys)
		outputValues = pt.empty_like(values)
		outputCount = C.c_int()
		with _lock, pt.cuda.device(cudaDeviceIndex()):
			reserveWorkspace(count)
			error = nativeLibrary().sorter_run(_workspace, keys.data_ptr(), values.data_ptr(), count, extent, max(1, (extent-1).bit_length()), outputKeys.data_ptr(), outputValues.data_ptr(), C.byref(outputCount))
			if(error):
				raise RuntimeError(error.decode())
		# A negative count asks for PyTorch's own CPU reduction order. Invalid keys and CUDA failures raise.
		if(outputCount.value >= 0):
			outputIndices = pt.empty((rank, outputCount.value), dtype=pt.int64, device=gpuCustomSparseTensorDriverCPU)
			nativeLibrary().cpu_unpack(outputKeys.data_ptr(), outputIndices.data_ptr(), dimensions, rank, outputCount.value, threads)
			if(outputCount.value < count):
				outputValues = outputValues[:outputCount.value].clone()	#persistent tensors own only the surviving values
			result = pt.sparse_coo_tensor(outputIndices, outputValues, size=shape, dtype=pt.float32, device=gpuCustomSparseTensorDriverCPU, is_coalesced=True)
	return result


def reserveWorkspace(count):
	global _capacity
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		if(count > _capacity):
			if(_workspace.value is not None):
				nativeLibrary().sorter_close(_workspace)
				_workspace.value = None
				_capacity = 0
			error = nativeLibrary().sorter_create(C.byref(_workspace), count)
			if(error):
				raise RuntimeError(error.decode())
			_capacity = count


def releaseWorkspace():
	global _capacity
	if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
		with _lock:
			if(_workspace.value is not None):
				with pt.cuda.device(cudaDeviceIndex()):
					nativeLibrary().sorter_close(_workspace)
				_workspace.value = None
				_capacity = 0


if(useGPUcustomSparseTensorDriver or useHybridGPUCPUcustomSparseTensorDriver):
	atexit.register(releaseWorkspace)
