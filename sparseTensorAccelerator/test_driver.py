"""Opt-in hardware regression tests. Run from the repository with the driver enabled."""
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"proto"))
import GIAANNcmn_globalDefs as defs

if(defs.useGPUcustomSparseTensorDriver or defs.useGPUCPUhybridCustomSparseTensorDriver):
	import GIAANNcmn_gpuCustomSparseTensorDriver as cudaDriver
	if(defs.useGPUCPUhybridCustomSparseTensorDriver):
		import GIAANNcmn_hybridGPUCPUcustomSparseTensorDriver as driver
	else:
		driver = cudaDriver
	activeFlag = "useGPUCPUhybridCustomSparseTensorDriver" if defs.useGPUCPUhybridCustomSparseTensorDriver else "useGPUcustomSparseTensorDriver"
	import GIAANNcmn_sequenceObservedColumns as columns
	import GIAANNcmn_sparseTensors as sparse


@contextmanager
def smallDispatch():
	with ExitStack() as stack:
		for module in {driver, cudaDriver}:
			stack.enter_context(patch.object(module, "gpuCustomSparseTensorDriverMinEntries", 1))
		yield


@unittest.skipUnless(defs.useGPUcustomSparseTensorDriver or defs.useGPUCPUhybridCustomSparseTensorDriver, "Enable one custom driver to run hardware tests")
class GPUCustomSparseTensorDriverTests(unittest.TestCase):
	def assertCOOEqual(self, actual, expected):
		self.assertEqual((actual.shape, actual.dtype, actual.device, actual.is_coalesced()), (expected.shape, expected.dtype, expected.device, expected.is_coalesced()))
		pt.testing.assert_close(actual.indices(), expected.indices(), rtol=0, atol=0)
		pt.testing.assert_close(actual.values().contiguous().view(pt.int32), expected.values().contiguous().view(pt.int32), rtol=0, atol=0)

	def testTrainingAndBranchSum(self):
		generator = pt.Generator().manual_seed(193)
		shape = (2, 3, 7, 11, 23)
		idx = pt.stack([pt.randint(d, (317,), generator=generator) for d in shape])
		left = pt.sparse_coo_tensor(idx, pt.randint(-3, 9, (317,), generator=generator).float(), shape)
		right = pt.sparse_coo_tensor(idx.flip(1), pt.ones(317), shape)
		with smallDispatch():
			with patch.object(columns, activeFlag, False), patch.object(sparse, activeFlag, False):
				refTrain = columns.SequenceObservedColumns.addSparseUpdate(None, left, right)
				refBranch = sparse.collapseSparseBranchDimension(left)
			self.assertCOOEqual(columns.SequenceObservedColumns.addSparseUpdate(None, left, right), refTrain)
			self.assertCOOEqual(sparse.collapseSparseBranchDimension(left), refBranch)
			self.assertGreater(cudaDriver._capacity, 0)
			cudaDriver.releaseWorkspace()
			self.assertIsNone(cudaDriver._workspace.value)

	def testOrderSensitiveAndUnsupported(self):
		with smallDispatch():
			a = pt.sparse_coo_tensor([[0, 0, 0]], [2.0**30, 1.0, -(2.0**30)], (4,))
			self.assertIsNone(cudaDriver.reduceCoordinates((a,)))
			self.assertCOOEqual(driver.coalesceTensor(a), a.coalesce())
			self.assertEqual(driver.coalesceTensor(a.double()).dtype, pt.float64)
			before = a.requires_grad_()
			with pt.enable_grad():
				driver.coalesceTensor(before).values().sum().backward()
				self.assertIsNotNone(before.grad)
			bad = pt.sparse_coo_tensor([[1, 0], [4, 1]], [1., 2.], (3, 4), check_invariants=False)
			with self.assertRaises(ValueError):
				driver.coalesceTensor(bad)

	@unittest.skipUnless(defs.optimiseTrainSequenceBatch, "Run in batched training mode")
	def testChronologicalTrainingCommit(self):
		from types import SimpleNamespace
		import numpy as np
		import GIAANNcmn_gpuTrainCommit as commit
		import GIAANNcmn_trainSequenceBatch as batches
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections as Packed
		shape = (1, 1, 3, 2, 7)
		keys = pt.tensor([0, 1, 7])
		old = [pt.sparse_coo_tensor([[0], [0], [1], [0], [2]], [2.0**24], shape).coalesce() for _ in keys]
		packets = []
		for delta in (1.0, -(2.0**24), 1.0):
			indices = pt.tensor([[0,0,0], [0,0,0], [1,1,1], [0,1,2], [0,0,0], [2,2,2]])
			packet = pt.sparse_coo_tensor(indices, [delta, delta, delta], (1,1,3,3,2,7)).coalesce()
			packets.append((keys, packet))
		expected = batches.getExtension().mergeSequenceBatch(True, old, [p for k,p in packets], [pt.arange(len(keys)) for _ in packets], list(shape), 3, 1, defs.trainSequenceBatchInvalidInput)
		for packed in ([False, True] if defs.optimisationStoreDatabaseResidentCoordinatesAsInt32 else [False]):
			with self.subTest(packed=packed):
				cols = {i: SimpleNamespace(featureConnectionsBySourceFeature={}, loadedSourceFeatureIndices=set()) for i in range(2)}
				for k,t in zip(keys.tolist(), old):
					c,f = divmod(k,shape[-1])
					cols[c].featureConnectionsBySourceFeature[f] = Packed(t) if packed else t
					cols[c].loadedSourceFeatureIndices.add(f)
				dtype = commit._recordType if packed else np.dtype(np.int32)
				args = (True, cols, {}, packets, shape, 3, packed, False, Packed, dtype)
				before = cols[0].featureConnectionsBySourceFeature[0]
				with pt.cuda.device(cudaDriver.cudaDeviceIndex()):
					self.assertEqual(commit.getExtension().merge(*args, 3, defs.trainSequenceBatchInvalidInput, 0, None), (None,None))
					names, pending = commit.getExtension().merge(*args, 4, defs.trainSequenceBatchInvalidInput, 0, None)
				self.assertIs(before, cols[0].featureConnectionsBySourceFeature[0])
				self.assertEqual((names,pending.tiles,pending.inputEntries,pending.outputEntries), ([0,1,7],3,12,3))
				pending.publish()
				for k,reference in zip(names,expected):
					c,f = divmod(k,shape[-1])
					value = cols[c].featureConnectionsBySourceFeature[f]
					actual = value.materialise("cpu") if packed else value
					self.assertCOOEqual(actual,reference)
					self.assertEqual(actual.values().item(),1.0)

	@unittest.skipUnless(defs.optimiseTrainSequenceBatch, "Run in batched training mode")
	def testSparseHistoricalMerge(self):
		from types import SimpleNamespace
		import GIAANNcmn_gpuTrainCommit as commit
		shape = (1,1,1,1,64)
		ix = pt.zeros((5,48),dtype=pt.long)
		ix[-1] = pt.arange(48)
		old = pt.sparse_coo_tensor(ix,pt.arange(48,dtype=pt.float),shape,is_coalesced=True)
		col = SimpleNamespace(featureConnectionsBySourceFeature={0:old},loadedSourceFeatureIndices={0})
		q = pt.tensor([[0,0],[0,0],[0,0],[0,0],[0,0],[12,49]])
		update = pt.sparse_coo_tensor(q,[.5,-2.],(1,1,1,1,1,64),is_coalesced=True)
		state = SimpleNamespace(columns={0:col}, connections=[(pt.tensor([0]),update)], sourceShape=shape)
		with patch.object(commit, "optimisationStoreDatabaseResidentCoordinatesAsInt32", False):
			keys,pending = commit.merge(state,{})
		self.assertEqual((keys,pending.tiles),([0],int(defs.useGPUcustomSparseTensorDriver)))
		self.assertIs(col.featureConnectionsBySourceFeature[0],old)
		pending.publish()
		expected = (old+pt.sparse_coo_tensor(q[[0,1,2,4,5]],[.5,-2.],shape)).coalesce()
		self.assertCOOEqual(col.featureConnectionsBySourceFeature[0],expected)

	def testCanonicalGatherAndSegmentSlice(self):
		import GIAANNcmn_predictionActivate as activation
		shape = (1, 1, 9, 29)
		keys = pt.arange(0,9*29,2)
		indices = pt.stack(pt.unravel_index(keys,shape))
		values = pt.arange(keys.numel(),dtype=pt.float32)
		values[0] = -0.
		values[1] = float("nan")
		for device in ("cpu","cuda"):
			x = pt.sparse_coo_tensor(indices.to(device),values.to(device),shape,is_coalesced=True)
			q = pt.stack(pt.unravel_index(pt.tensor([0,1,2,27,0,106,259,260]),shape)).to(device)
			probes = q.T.contiguous().T
			for mod in (sparse,activation):
				for wanted in (pt.float32,pt.float64):
					expected = mod.gatherSparseTensorValuesAtIndicesReference(x,probes,wanted)
					actual = mod.gatherSparseTensorValuesAtIndices(x,probes,wanted)
					bits = pt.int64 if wanted==pt.float64 else pt.int32
					self.assertTrue(pt.equal(actual.view(bits),expected.view(bits)))
				self.assertEqual(mod.gatherSparseTensorValuesAtIndices(x,q[:,:0],pt.float32).numel(),0)
			bad = q.clone()
			bad[-1,0] = shape[-1]
			with self.assertRaises(RuntimeError):
				sparse.gatherSparseTensorValuesAtIndices(x,bad,pt.float32)
			expected = activation.gatherSparseTensorValuesAtIndicesReference(x,bad,pt.float32)
			self.assertTrue(pt.equal(activation.gatherSparseTensorValuesAtIndices(x,bad,pt.float32).view(pt.int32),expected.view(pt.int32)))
			with patch.object(sparse,"deviceSparse",device):
				for axis, coordinate in ((0,0),(1,0),(2,0),(2,6),(2,-1),(2,99),(3,7)):
					expected = sparse.sliceSparseTensorReference(x,axis,coordinate)
					actual = sparse.sliceSparseTensor(x,axis,coordinate)
					self.assertCOOEqual(actual,expected)
					if(actual._nnz()):
						actual.values().fill_(42)
				self.assertTrue(pt.equal(x.values().view(pt.int32).cpu(),values.view(pt.int32)))
			with pt.enable_grad():
				with_values = x.detach().requires_grad_()
				out = sparse.gatherSparseTensorValuesAtIndices(with_values,q[:,2:4],pt.float32)
				out.sum().backward()
				self.assertIsNotNone(with_values.grad)

	@unittest.skipUnless(defs.useGPUcustomSparseTensorDriver, "GPU driver")
	def testCUDAUnionDispatch(self):
		# Include insertions before, between and after rows, shared coordinates and a partial block.
		for device in ("cpu", "cuda"):
			with self.subTest(device=device), smallDispatch():
				leftKeys = pt.arange(3, 1500, 4, device=device)
				rightKeys = pt.arange(1502, device=device)
				def tensor(keys):
					return pt.sparse_coo_tensor(keys.unsqueeze(0), keys.float()-39, (1502,), is_coalesced=True)
				for left, right in ((tensor(leftKeys), tensor(rightKeys)), (tensor(rightKeys), tensor(leftKeys))):
					result = cudaDriver.mergeCUDA(left, right)
					self.assertCOOEqual(result, (left+right).coalesce())
				queries = leftKeys[:5].unsqueeze(0)
				gathered = cudaDriver.gatherCanonical(tensor(leftKeys), queries, pt.float32)
				if device == "cuda":
					self.assertTrue(pt.equal(gathered, leftKeys[:5].float()-39))
				else:
					self.assertIsNone(gathered)
					self.assertIsNone(cudaDriver.sliceCanonical(pt.sparse_coo_tensor(pt.stack((pt.zeros_like(leftKeys), leftKeys)), leftKeys.float(), (1,1502), is_coalesced=True), 0, 0, "cpu"))
		if defs.optimiseTrainSequenceBatch:
			import GIAANNcmn_gpuTrainCommit as commit
			self.assertEqual(commit.historyRatio(), 0)

	def testCanonicalMerges(self):
		# Cover a smaller operand whose prefix precedes the first partition pivot.
		generator = pt.Generator().manual_seed(724)
		for large, other in ((13400, 7900), (8000, 113), (8000, 1), (5, 2)):
			universe = 3*large
			shape = (1, 1, universe)
			leftKeys = pt.randperm(universe-5, generator=generator)[:large].sort().values+5
			rightKeys = pt.cat((pt.tensor([0]), pt.randperm(universe-1, generator=generator)[:other-1]+1)).sort().values
			operands = []
			for keys in (leftKeys, rightKeys):
				# Deliberately exercise non-contiguous rows and values.
				storage = pt.zeros((3, keys.numel()*2+5), dtype=pt.long)
				indices = storage[:, 1:2*keys.numel()+1:2]
				indices[-1] = keys
				values = pt.randn(2*keys.numel(), generator=generator)[::2]
				operands.append(pt.sparse_coo_tensor(indices, values, shape, is_coalesced=True))
			with smallDispatch():
				for left,right in (operands, operands[::-1]):
					expected = (left+right).coalesce()
					out = driver.addTensors(left,right)
					self.assertCOOEqual(out,expected)
					self.assertNotEqual(out.indices().data_ptr(), left.indices().data_ptr())
					self.assertNotEqual(out.values().data_ptr(), left.values().data_ptr())
		with smallDispatch():
			left = pt.sparse_coo_tensor([list(range(100))], pt.ones(100), (101,), is_coalesced=True)
			bad = pt.sparse_coo_tensor([[102]], [1.], (101,), is_coalesced=True, check_invariants=False)
			with self.assertRaises(ValueError):
				driver.addTensors(left, bad)

	def testSignedZeroAndNaNPayloads(self):
		with smallDispatch():
			coords = pt.arange(257).unsqueeze(0)
			values = pt.ones(257)
			values[:3] = pt.tensor([0x7fc00002, -2147483648, -2147483648],dtype=pt.int32).view(pt.float32)
			left = pt.sparse_coo_tensor(coords,values,(258,),is_coalesced=True)
			for rightCount in (2,257):
				rightValues = values[:rightCount].clone()
				rightValues[0] = pt.tensor(0x7fc00101,dtype=pt.int32).view(pt.float32)
				right = pt.sparse_coo_tensor(coords[:,:rightCount],rightValues,left.shape,is_coalesced=True)
				self.assertCOOEqual(driver.addTensors(left,right),(left+right).coalesce())
				expected = pt.sparse_coo_tensor(pt.cat((left.indices(),right.indices()),1),pt.cat((left.values(),right.values())),left.shape).coalesce()
				self.assertCOOEqual(driver.addTensors(left,right,True),expected)

			bad = pt.sparse_coo_tensor([[258]], [float("nan")], left.shape, is_coalesced=True, check_invariants=False)
			with self.assertRaises(ValueError):
				driver.addTensors(left, bad)

	@unittest.skipUnless(defs.useGPUCPUhybridCustomSparseTensorDriver, "Hybrid host policy")
	def testIndexedStateAndDecay(self):
		generator = pt.Generator().manual_seed(651)
		shape = (3, 9, 11, 97)
		keys = pt.randperm(3*9*11*97, generator=generator)[:13000].sort().values
		coordinates = pt.stack(pt.unravel_index(keys,shape))
		values = pt.randint(0,7,(len(keys),), generator=generator).float()/4
		x = pt.sparse_coo_tensor(coordinates,values,shape,is_coalesced=True)
		col, feat = 7, 19
		selected = (coordinates[2]==col) & (coordinates[3]==feat)
		self.assertTrue(pt.equal(driver.findNeuron(x,col,feat), selected.nonzero().flatten()))
		candidates = pt.tensor([0,0,1,2,456,901,2300,2900])
		mask = pt.isin((coordinates[0]*shape[2]+coordinates[2])*shape[3]+coordinates[3],candidates) & pt.isin(coordinates[1],pt.tensor([8,1,2]))
		self.assertTrue(pt.equal(driver.candidatePositions(x,candidates,[8,2,1]),mask.nonzero().flatten()))
		for multiply in (False,True):
			for rate in (0.,.25,2.):
				v = values.clone()
				soma = coordinates[1]==8
				v[soma] = v[soma]*rate if multiply else (v[soma]-rate).clamp(min=0)
				expected = pt.sparse_coo_tensor(coordinates[:,v>0],v[v>0],shape,is_coalesced=True)
				self.assertCOOEqual(driver.decaySoma(x,8,rate,multiply),expected)
		x.values()[0] = float("nan")
		with self.assertRaises(RuntimeError):
			driver.decaySoma(x,8,.25,False)
		with self.assertRaises(RuntimeError):
			driver.validateCandidateState(x)

	@unittest.skipUnless(defs.useGPUcustomSparseTensorDriver, "GPU driver")
	def testResidentSegmentedResponse(self):
		import GIAANNcmn_predictionActivate as activation
		if not activation.inferenceReviewPatch14NormaliseConnectionInputs:
			self.skipTest("requires normalized connection strengths")
		shape = (1, activation.arrayNumberOfSegments, 1, 5)
		rows = pt.tensor([[0,0,0,0],[0,0,2,2],[0,0,0,0],[0,1,2,3]],device="cuda")
		values = pt.tensor([-0., 1.5, .71, 3.], device="cuda")
		x = pt.sparse_coo_tensor(rows, values, shape, is_coalesced=True)
		before = values.clone()
		with smallDispatch(), pt.cuda.stream(pt.cuda.Stream()):
			pt.cuda.current_stream().wait_stream(pt.cuda.default_stream())
			active = pt.tensor(.375, device="cuda")
			with patch.object(activation, activeFlag, False), patch.object(sparse, activeFlag, False):
				expected = activation.calculateFeatureNeuronsTargetActivationPredict(x, active)
			result = activation.calculateFeatureNeuronsTargetActivationPredict(x, active)
			self.assertCOOEqual(result, expected)
			self.assertTrue(pt.equal(x.values().view(pt.int32), before.view(pt.int32)))
			pt.cuda.current_stream().synchronize()

	@unittest.skipUnless(defs.useGPUcustomSparseTensorDriver, "GPU driver")
	def testUncoalescedSegmentedResponse(self):
		import GIAANNcmn_predictionActivate as activation
		if not activation.inferenceReviewPatch14NormaliseConnectionInputs:
			self.skipTest("requires normalized connection strengths")
		shape = (1, activation.arrayNumberOfSegments, 1, 3)
		with smallDispatch():
			for device in ("cpu", "cuda"):
				for segments in ([0, 1, 2], [2, 1, 1]):
					idx = pt.tensor([[0,0,0], segments, [0,0,0], [0,0,0]], device=device)
					x = pt.sparse_coo_tensor(idx, pt.tensor([1.,2.,3.], device=device), shape)
					self.assertTrue(cudaDriver.canNormalise(x))
					with patch.object(activation, activeFlag, False):
						expected = activation.normaliseReviewSourceConnectionInputs(x)
					self.assertCOOEqual(activation.normaliseReviewSourceConnectionInputs(x), expected)
				x._values()[1] = -1
				with self.assertRaises(RuntimeError):
					activation.normaliseReviewSourceConnectionInputs(x)

	def testSegmentedResponse(self):
		import GIAANNcmn_predictionActivate as activation
		if not activation.inferenceReviewPatch14NormaliseConnectionInputs:
			self.skipTest("requires normalized connection strengths")
		shape = (1, activation.arrayNumberOfSegments, 57, 109)
		n = 1331
		generator = pt.Generator().manual_seed(349)
		keys = pt.randperm(shape[1]*shape[2]*shape[3], generator=generator)[:n].sort().values
		idx = pt.stack(pt.unravel_index(keys, shape))
		v = pt.rand(n, generator=generator)
		v[::11] = -0.
		# Include an empty segment and an explicitly zero-only segment.
		v[idx[1]==1] = 0
		mask = idx[1]!=2
		x = pt.sparse_coo_tensor(idx[:,mask],v[mask],shape,is_coalesced=True)
		before = x.values().clone()
		with smallDispatch():
			for gain in (0.,.371,1.):
				active = pt.tensor(gain)
				with patch.object(activation, activeFlag, False), patch.object(sparse, activeFlag, False):
					expected = activation.calculateFeatureNeuronsTargetActivationPredict(x,active)
				result = activation.calculateFeatureNeuronsTargetActivationPredict(x,active)
				self.assertCOOEqual(result,expected)
			self.assertTrue(pt.equal(x.values().view(pt.int32),before.view(pt.int32)))
			x.values()[0] = float("inf")
			with self.assertRaises(RuntimeError):
				activation.normaliseReviewSourceConnectionInputs(x)

	@unittest.skipUnless(getattr(defs, "optimiseTrainFusedSequenceUpdates", False), "requires GIAANN batched training mode")
	@unittest.skipUnless(defs.useGPUCPUhybridCustomSparseTensorDriver, "Hybrid host policy")
	def testBatchedTrainingPacketDispatch(self):
		from types import SimpleNamespace
		import GIAANNcmn_trainFusedSequence as fused
		db = SimpleNamespace(arrayNumberOfProperties=1, c=200019, f=200020)
		with patch.object(cudaDriver, "reduceCoordinates", wraps=cudaDriver.reduceCoordinates) as reductions:
			for length in (128, 640):
				for repeated in (False, True):
					features = pt.arange(length)
					if(repeated):
						features %= 37
					sequence = SimpleNamespace(fs=length, cs=1, conceptIndicesInSequenceObservedTensor=pt.tensor([12]), databaseNetworkObject=db, getObservedColumnFeatureIndices=lambda: (features, pt.arange(length)))
					inputs = (sequence, pt.tensor([0]), pt.tensor([0]), pt.tensor([length]))
					with patch.object(fused, "useGPUCPUhybridCustomSparseTensorDriver", False):
						expected = fused.prepare(*inputs)
					actual = fused.prepare(*inputs)
					for result, reference in zip(actual, expected):
						if(reference.is_sparse):
							self.assertCOOEqual(result, reference)
						else:
							self.assertTrue(pt.equal(result, reference))
					if(length == 128):
						self.assertEqual(reductions.call_count, 0)
			self.assertEqual(reductions.call_count, 2)

	def testBSR(self):
		with pt.random.fork_rng(devices=[]):
			pt.manual_seed(621)
			for ptr, col in (([0, 1, 2, 3], [2, 2, 0]), ([0, 2, 2, 3], [0, 2, 1])):
				for dtype in (pt.float32, pt.float16):
					weight = pt.sparse_bsr_tensor(pt.tensor(ptr), pt.tensor(col), pt.randn(len(col), 4, 4).to(dtype), (12, 12))
					x = pt.randn(12, 17).to(dtype)
					bias = pt.randn(12).to(dtype)
					reference = x.float()
					for step in range(3):
						reference = (weight.to_dense().float() @ reference + bias.float()[:, None]).relu()
					result = driver.propagateBlockSparse(weight, x, bias, 3)
					self.assertEqual(result.device, x.device)
					pt.testing.assert_close(result.float(), reference, rtol=0.005 if dtype == pt.float16 else 1e-5, atol=0.03 if dtype == pt.float16 else 1e-5)


@unittest.skipUnless(defs.useGPUstockSparseTensorDriver and defs.optimiseTrainSequenceBatch, "Enable the stock CUDA driver in training mode")
class StockCUDATrainingTests(unittest.TestCase):
	def testPackedMergeAcrossTilesAndEmptySources(self):
		from types import SimpleNamespace
		import GIAANNcmn_stockSparseTensorDriver as stock
		import GIAANNcmn_databasePacked as storage
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections as Packed
		shape = (1, 1, 3, 2, 7)
		history = pt.sparse_coo_tensor([[0], [0], [1], [0], [2]], [4.0], (1, 1, 3, 1, 4)).coalesce()
		empty = pt.sparse_coo_tensor(size=shape).coalesce()
		column = SimpleNamespace(featureConnectionsBySourceFeature={0: Packed(history), 1: Packed(empty), 2: Packed(empty)}, loadedSourceFeatureIndices={0, 1, 2})
		keys = pt.tensor([0, 1, 2])
		coords = pt.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 2], [0, 0, 2], [0, 0, 1], [2, 2, 6]])
		raw = pt.sparse_coo_tensor(coords, [-2.0, -2.0, 3.0], (1, 1, 3, 3, 2, 7))
		packet = stock.coalesceTensor(raw)
		self.assertTrue(packet.is_cpu and packet.is_coalesced())
		names, pending = stock.getExtension().merge({0: column}, {}, [(keys, packet)], shape, 3, False, Packed, storage.getEntryDtype(), 1, defs.trainSequenceBatchInvalidInput, stock.reduceTile)
		self.assertEqual((names, pending.inputEntries, pending.outputEntries, pending.tiles), ([0, 1, 2], 3, 2, 2))
		self.assertEqual(column.featureConnectionsBySourceFeature[0].entries["values"].tolist(), [4.0])
		pending.publish()
		for feature, coordinates, values in [(0, [[0, 0, 1, 0, 2]], [0.0]), (1, [], []), (2, [[0, 0, 2, 1, 6]], [3.0])]:
			self.assertEqual(column.featureConnectionsBySourceFeature[feature].shape, shape)
			records = column.featureConnectionsBySourceFeature[feature].entries
			self.assertEqual(records["indices"].tolist(), coordinates)
			self.assertEqual(records["values"].tolist(), values)


@unittest.skipUnless(defs.useGPUstockSparseTensorDriver and defs.executionMode == "inference" and defs.inferenceContextLikelihood, "Enable stock CUDA context-likelihood inference")
class StockCUDAInferenceTests(unittest.TestCase):
	def testDuplicateFeatureFrequenciesStayOnCUDA(self):
		from types import SimpleNamespace
		import GIAANNcmn_predictionContextLikelihood as scorer
		db = SimpleNamespace(c=1, f=5, arrayIndexPropertiesStrengthIndex=0)
		neurons = pt.sparse_coo_tensor([[0, 0, 0], [0, 0, 0], [0, 1, 1], [2, 2, 4]], [3.0, 7.0, 5.0], (1, 1, 2, 5), device="cuda")
		with patch.object(scorer.GIAANNcmn_databaseNetworkFiles, "loadObservedColumnFeatureNeuronsTensor", return_value=neurons) as loader:
			keys, frequencies = scorer.loadContextLikelihoodColumnFrequencies(db, 0)
		loader.assert_called_once_with(db, 0, targetDevice="cuda")
		self.assertEqual((keys.device, frequencies.device, frequencies.dtype), (neurons.device, neurons.device, pt.float64))
		self.assertEqual(keys.tolist(), [2, 4])
		self.assertEqual(frequencies.tolist(), [7.0, 5.0])


if(__name__ == "__main__"):
	unittest.main()
