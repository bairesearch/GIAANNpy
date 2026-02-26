#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {

constexpr int64_t kEmptyKey = static_cast<int64_t>(-1);

__device__ __forceinline__ uint64_t mix_hash64(uint64_t x) {
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;
	x ^= x >> 33;
	return x;
}

__global__ void build_sparse_hash_kernel(const int64_t* targetLinearKey, const int64_t nnz, int64_t* hashKeys, int64_t* hashSlots, const int64_t hashCapacity, const int64_t maxProbe) {
	int64_t updateIndex = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
	if(updateIndex < nnz) {
		int64_t key = targetLinearKey[updateIndex];
		uint64_t hashValue = mix_hash64(static_cast<uint64_t>(key));
		int64_t baseSlot = static_cast<int64_t>(hashValue % static_cast<uint64_t>(hashCapacity));
		int64_t probeLimit = maxProbe;
		if(probeLimit > hashCapacity) {
			probeLimit = hashCapacity;
		}
		for(int64_t probe = 0; probe < probeLimit; probe++) {
			int64_t slot = baseSlot + probe;
			if(slot >= hashCapacity) {
				slot -= hashCapacity;
			}
			unsigned long long* keyAddress = reinterpret_cast<unsigned long long*>(&hashKeys[slot]);
			unsigned long long observed = atomicCAS(keyAddress, static_cast<unsigned long long>(kEmptyKey), static_cast<unsigned long long>(key));
			if(static_cast<int64_t>(observed) == kEmptyKey) {
				unsigned long long* slotAddress = reinterpret_cast<unsigned long long*>(&hashSlots[slot]);
				atomicCAS(slotAddress, static_cast<unsigned long long>(kEmptyKey), static_cast<unsigned long long>(updateIndex));
				break;
			}
			if(static_cast<int64_t>(observed) == key) {
				break;
			}
		}
	}
	return;
}

__global__ void accumulate_sparse_updates_kernel(float* targetValues, const int64_t* updateLinearKey, const float* updateValues, const int64_t numUpdates, const int64_t* hashKeys, const int64_t* hashSlots, const int64_t hashCapacity, const int64_t maxProbe, int64_t* overflowKeys, float* overflowValues, int64_t overflowCapacity, int64_t* overflowCount, int64_t* hashHits) {
	int64_t updateIndex = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
	if(updateIndex < numUpdates) {
		int64_t key = updateLinearKey[updateIndex];
		float value = updateValues[updateIndex];
		uint64_t hashValue = mix_hash64(static_cast<uint64_t>(key));
		int64_t baseSlot = static_cast<int64_t>(hashValue % static_cast<uint64_t>(hashCapacity));
		int64_t probeLimit = maxProbe;
		if(probeLimit > hashCapacity) {
			probeLimit = hashCapacity;
		}
		bool matched = false;
		for(int64_t probe = 0; probe < probeLimit; probe++) {
			int64_t slot = baseSlot + probe;
			if(slot >= hashCapacity) {
				slot -= hashCapacity;
			}
			int64_t existingKey = hashKeys[slot];
			if(existingKey == key) {
				int64_t targetSlot = hashSlots[slot];
				if(targetSlot < 0) {
					break;
				}
				atomicAdd(&targetValues[targetSlot], value);
				atomicAdd(reinterpret_cast<unsigned long long*>(hashHits), static_cast<unsigned long long>(1));
				matched = true;
				break;
			}
			if(existingKey == kEmptyKey) {
				break;
			}
		}
		if(!matched) {
			int64_t overflowIndex = static_cast<int64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(overflowCount), static_cast<unsigned long long>(1)));
			if(overflowIndex < overflowCapacity) {
				overflowKeys[overflowIndex] = key;
				overflowValues[overflowIndex] = value;
			}
		}
	}
	return;
}

} // namespace

void build_sparse_hash_cuda(torch::Tensor targetLinearKey, torch::Tensor hashKeys, torch::Tensor hashSlots, int64_t maxProbe) {
	if(targetLinearKey.scalar_type() != torch::kInt64 || hashKeys.scalar_type() != torch::kInt64 || hashSlots.scalar_type() != torch::kInt64) {
		throw std::runtime_error("build_sparse_hash_cuda error: expected int64 tensors");
	}
	int64_t nnz = targetLinearKey.numel();
	if(nnz > 0) {
		const int threads = 256;
		const int blocks = static_cast<int>((nnz + threads - 1) / threads);
		build_sparse_hash_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(targetLinearKey.data_ptr<int64_t>(), nnz, hashKeys.data_ptr<int64_t>(), hashSlots.data_ptr<int64_t>(), hashKeys.numel(), maxProbe);
		C10_CUDA_KERNEL_LAUNCH_CHECK();
	}
	return;
}

void accumulate_sparse_updates_cuda(torch::Tensor targetValues, torch::Tensor updateLinearKey, torch::Tensor updateValues, torch::Tensor hashKeys, torch::Tensor hashSlots, torch::Tensor overflowKeys, torch::Tensor overflowValues, torch::Tensor overflowCount, torch::Tensor hashHits, int64_t maxProbe) {
	if(targetValues.scalar_type() != torch::kFloat32) {
		throw std::runtime_error("accumulate_sparse_updates_cuda error: targetValues must be float32");
	}
	if(updateValues.scalar_type() != torch::kFloat32) {
		throw std::runtime_error("accumulate_sparse_updates_cuda error: updateValues must be float32");
	}
	if(overflowValues.scalar_type() != torch::kFloat32) {
		throw std::runtime_error("accumulate_sparse_updates_cuda error: overflowValues must be float32");
	}
	if(updateLinearKey.scalar_type() != torch::kInt64 || hashKeys.scalar_type() != torch::kInt64 || hashSlots.scalar_type() != torch::kInt64 || overflowKeys.scalar_type() != torch::kInt64 || overflowCount.scalar_type() != torch::kInt64 || hashHits.scalar_type() != torch::kInt64) {
		throw std::runtime_error("accumulate_sparse_updates_cuda error: expected int64 index/counter tensors");
	}
	int64_t numUpdates = updateLinearKey.numel();
	if(numUpdates > 0) {
		const int threads = 256;
		const int blocks = static_cast<int>((numUpdates + threads - 1) / threads);
		accumulate_sparse_updates_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(targetValues.data_ptr<float>(), updateLinearKey.data_ptr<int64_t>(), updateValues.data_ptr<float>(), numUpdates, hashKeys.data_ptr<int64_t>(), hashSlots.data_ptr<int64_t>(), hashKeys.numel(), maxProbe, overflowKeys.data_ptr<int64_t>(), overflowValues.data_ptr<float>(), overflowKeys.numel(), overflowCount.data_ptr<int64_t>(), hashHits.data_ptr<int64_t>());
		C10_CUDA_KERNEL_LAUNCH_CHECK();
	}
	return;
}
