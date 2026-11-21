// Segment boundaries are computed from canonical COO on the host. Only FP32
// values travel over PCIe. A block covers one segment, including partial tiles.
#include "gather.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
namespace {
constexpr int limit = 64, lanes = 256, tile = 1024;
struct Plan {
	int ends[limit], tiles[limit], nsegments;
};
__device__ int segment_for(int block, const Plan &p) {
	int lo = 0, hi = p.nsegments;
	while (lo < hi) {
		int m = (lo + hi) / 2;
		if (p.tiles[m] <= block)
			lo = m + 1;
		else
			hi = m;
	}
	return lo;
}
template <bool scale>
__global__ void transform(Plan p, float *v, int *stats, float reference,
													float gain, int lastColumn, float active) {
	const int block = blockIdx.x, s = segment_for(block, p);
	int64_t begin = (s ? p.ends[s - 1] : 0) +
									int64_t(block - (s ? p.tiles[s - 1] : 0)) * tile;
	int localMax = 0, invalid = 0;
	float factor = 0;
	if (scale && stats[s])
		factor = __fmul_rn(reference, __frcp_rn(__int_as_float(stats[s])));
	for (int64_t i = begin + threadIdx.x; i < p.ends[s] && i < begin + tile;
			 i += lanes) {
		float x = v[i];
		if (scale) {
			float y = x > 0 ? __fmul_rn(x, factor) : x;
			if (s <= lastColumn)
				y = __fmul_rn(y, gain);
			invalid |= !isfinite(y) || (x > 0 && y <= 0);
			v[i] = __fmul_rn(y, active);
		} else {
			invalid |= !isfinite(x) || x < 0;
			localMax = max(localMax, __float_as_int(x));
		}
	}
	if (__any_sync(0xffffffff, invalid) && (threadIdx.x & 31) == 0)
		atomicOr(stats + p.nsegments, scale ? 2 : 1);
	if (!scale) {
		for (int d = 16; d; d >>= 1)
			localMax = max(localMax, __shfl_down_sync(0xffffffff, localMax, d));
		if ((threadIdx.x & 31) == 0)
			atomicMax(stats + s, localMax);
	}
}
} // namespace
extern "C" const char *coo_normalise(float *values, int *stats,
																		 const int64_t *ends, int count,
																		 float reference, float gain,
																		 int lastColumn, float active,
																		 cudaStream_t stream) {
	Plan p{};
	p.nsegments = count;
	if (count < 1 || count > limit)
		return "unsupported segment count";
	int blocks = 0, prev = 0;
	for (int s = 0; s < count; ++s) {
		if (ends[s] < prev || ends[s] > INT_MAX)
			return "invalid segment boundaries";
		int n = static_cast<int>(ends[s]) - prev;
		blocks += int((int64_t(n) + tile - 1) / tile);
		p.ends[s] = prev = ends[s];
		p.tiles[s] = blocks;
	}
	if (blocks) {
		transform<false><<<blocks, lanes, 0, stream>>>(p, values, stats, reference,
																									 gain, lastColumn, active);
		auto err = cudaGetLastError();
		if (err != cudaSuccess)
			return cudaGetErrorString(err);
		transform<true><<<blocks, lanes, 0, stream>>>(p, values, stats, reference,
																									gain, lastColumn, active);
	}
	auto err = cudaGetLastError();
	return err == cudaSuccess ? nullptr : cudaGetErrorString(err);
}

// Indexing a leading sparse axis can return sorted unique coordinates without
// PyTorch's coalesced flag. Check the actual stream before using segmented ranges.
namespace {
__global__ void inspectStrengths(GatherView indices, const float* values, int n, GatherShape shape, int* status) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		if (!gatherValid(indices, i, shape, false) || !isfinite(values[i]) || values[i] < 0)
			atomicOr(status, 1);
		if (i && gatherCompare(indices,i-1,indices,i,shape.rank) >= 0)
			atomicOr(status, 2);
	}
}
}
extern "C" const char* coo_inspect_strengths(const int64_t* indices, int64_t row, int64_t entry, const float* values, int n, const int64_t* sizes, int rank, int* status, cudaStream_t stream) {
	const char* result = "invalid strength tensor geometry";
	if (n > 0 && rank > 0 && rank <= gatherMaxRank) {
		GatherShape shape{};
		shape.rank = rank;
		std::copy_n(sizes,rank,shape.sizes);
		inspectStrengths<<<(n-1)/lanes+1,lanes,0,stream>>>({indices,row,entry},values,n,shape,status);
		auto error = cudaGetLastError();
		result = error == cudaSuccess ? nullptr : cudaGetErrorString(error);
	}
	return result;
}
