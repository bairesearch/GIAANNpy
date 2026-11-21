#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Apply bias/ReLU and gather next step's source blocks in the same pass.
template <class T>
__global__ void route(const T *product, const T *bias, const int64_t *cols,
											T *dst, int width, int batch, size_t count) {
	size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= count)
		return;
	size_t tile_size = size_t(width) * batch, tile = i / tile_size,
				 within = i % tile_size;
	size_t source = cols ? cols[tile] : tile;
	float v = float(product[source * tile_size + within]) +
						float(bias[source * width + within / batch]);
	dst[i] = T(v < 0 ? 0 : v);
}
extern "C" const char *plan_route(const void *product, const void *bias,
																	const int64_t *cols, void *dst, int groups,
																	int width, int batch, int half,
																	void *stream) {
	size_t count = size_t(groups) * width * batch;
	size_t grid = (count + 255) / 256;
	if (grid > 0x7fffffff)
		return "grid too large";
	cudaStream_t s = static_cast<cudaStream_t>(stream);
	if (half)
		route<<<grid, 256, 0, s>>>(static_cast<const __half *>(product),
															 static_cast<const __half *>(bias), cols,
															 static_cast<__half *>(dst), width, batch, count);
	else
		route<<<grid, 256, 0, s>>>(static_cast<const float *>(product),
															 static_cast<const float *>(bias), cols,
															 static_cast<float *>(dst), width, batch, count);
	cudaError_t err = cudaGetLastError();
	return err == cudaSuccess ? nullptr : cudaGetErrorString(err);
}
