#include "gather.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace {
constexpr int blockSize = 256;
__global__ void gather(GatherView source, GatherView query, const float* values, int64_t valueStride, int64_t n, int64_t m, GatherShape shape, float* out, int* status) {
	const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	bool invalid = false;
	if (i < n)
		invalid = !gatherValid(source, i, shape, i != 0);
	if (i < m) {
		invalid |= !gatherValid(query, i, shape, false);
		out[i] = gatherAt(source, query, values, valueStride, n, i, shape.rank);
	}
	if (invalid)
		atomicOr(status, 1);
}
} // namespace
extern "C" const char* coo_gather_device(const int64_t* indices, int64_t row, int64_t entry, const float* values, int64_t valueStride, int64_t n, const int64_t* queries, int64_t queryRow, int64_t queryEntry, int64_t m, const int64_t* sizes, int rank, float* out, int* status, void* stream) {
	if (rank < 1 || rank > gatherMaxRank)
		return "Unsupported COO gather rank";
	GatherShape shape{};
	shape.rank = rank;
	std::copy_n(sizes, rank, shape.sizes);
	const auto count = std::max(n, m);
	auto queue = static_cast<cudaStream_t>(stream);
	int* deviceStatus = reinterpret_cast<int*>(out + m);
	cudaError_t error = cudaMemsetAsync(deviceStatus, 0, sizeof(int), queue);
	if (error == cudaSuccess && count) {
		gather<<<(count + blockSize - 1) / blockSize, blockSize, 0, queue>>>({indices, row, entry}, {queries, queryRow, queryEntry}, values, valueStride, n, m, shape, out, deviceStatus);
		error = cudaGetLastError();
	}
	if (error == cudaSuccess)
		error = cudaMemcpyAsync(status, deviceStatus, sizeof(int), cudaMemcpyDeviceToHost, queue);
	auto complete = cudaStreamSynchronize(queue);
	if (error == cudaSuccess)
		error = complete;
	return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}
