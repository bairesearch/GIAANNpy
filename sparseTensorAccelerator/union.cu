// Union of two canonical COO streams. One thread owns each input entry.
// Insertion ranks and a prefix sum of intersections determine disjoint output slots.
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

struct UnionView { const int64_t* indices; const float* values; int64_t rowStride, entryStride, valueStride; int size; };

namespace {
constexpr int lanes = 256, maxRank = 64;
constexpr int invalidCoordinate = 1, referenceArithmetic = 2;
struct Shape { int64_t sizes[maxRank]; int rank; };

__device__ int compare(UnionView a, int i, UnionView b, int j, int rank) {
	int ordering = 0;
	for (int d = 0; d < rank && !ordering; ++d) {
		const int64_t x = a.indices[d*a.rowStride + i*a.entryStride];
		const int64_t y = b.indices[d*b.rowStride + j*b.entryStride];
		ordering = (x > y) - (x < y);
	}
	return ordering;
}
__device__ void validate(UnionView v, int i, const Shape& shape, int* status) {
	for (int d = 0; d < shape.rank; ++d) {
		const int64_t x = v.indices[d*v.rowStride + i*v.entryStride];
		if (x < 0 || x >= shape.sizes[d]) atomicOr(status, invalidCoordinate);
	}
	if (i && compare(v,i-1,v,i,shape.rank) >= 0) atomicOr(status, invalidCoordinate);
	const unsigned bits = __float_as_uint(v.values[i*v.valueStride]);
	const unsigned mag = bits & 0x7fffffffu;
	if (bits == 0x80000000u || (mag && mag < 0x00800000u) || mag >= 0x7f800000u)
		atomicOr(status, referenceArithmetic);
}
__global__ void planUnion(UnionView a, UnionView b, Shape shape, int* positions, int* duplicates, int* status) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < a.size) validate(a, i, shape, status);
	if (i < b.size) {
		validate(b, i, shape, status);
		int lo = 0, hi = a.size;
		while (lo < hi) {
			const int mid = lo + (hi-lo)/2;
			if (compare(a,mid,b,i,shape.rank) < 0) lo = mid+1;
			else hi = mid;
		}
		positions[i] = lo;
		duplicates[i] = lo < a.size && compare(a,lo,b,i,shape.rank) == 0;
	}
}
__device__ void emit(UnionView source, int i, int out, float value, int rank, int size, int64_t* indices, float* values) {
	for (int d = 0; d < rank; ++d)
		indices[int64_t(d)*size + out] = source.indices[d*source.rowStride + i*source.entryStride];
	values[out] = value;
}
__global__ void emitUnion(UnionView a, UnionView b, int rank, const int* positions, const int* prefix, int size, int64_t* indices, float* values) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < a.size) {
		int lo = 0, hi = b.size;
		while (lo < hi) {
			const int mid = lo + (hi-lo)/2;
			if (positions[mid] <= i) lo = mid+1;
			else hi = mid;
		}
		const int intersections = lo ? prefix[lo-1] : 0;
		const bool ownedByB = lo && positions[lo-1] == i && intersections != (lo > 1 ? prefix[lo-2] : 0);
		if (!ownedByB)
			emit(a,i,i+lo-intersections,a.values[i*a.valueStride],rank,size,indices,values);
	}
	if (i < b.size) {
		const int before = i ? prefix[i-1] : 0;
		float value = b.values[i*b.valueStride];
		if (prefix[i] != before) value = __fadd_rn(a.values[positions[i]*a.valueStride],value);
		emit(b,i,positions[i]+i-before,value,rank,size,indices,values);
	}
}
}
extern "C" const char* coo_union_plan(UnionView a, UnionView b, const int64_t* sizes, int rank, int* positions, int* duplicates, int* status, cudaStream_t stream) {
	const char* result = "invalid CUDA union geometry";
	if (rank >= 1 && rank <= maxRank && a.size > 0 && b.size > 0) {
		Shape shape{};
		shape.rank = rank;
		std::copy_n(sizes,rank,shape.sizes);
		planUnion<<<(std::max(a.size,b.size)-1)/lanes+1,lanes,0,stream>>>(a,b,shape,positions,duplicates,status);
		auto error = cudaGetLastError();
		result = error == cudaSuccess ? nullptr : cudaGetErrorString(error);
	}
	return result;
}
extern "C" const char* coo_union_emit(UnionView a, UnionView b, int rank, const int* positions, const int* prefix, int count, int64_t* indices, float* values, cudaStream_t stream) {
	emitUnion<<<(std::max(a.size,b.size)-1)/lanes+1,lanes,0,stream>>>(a,b,rank,positions,prefix,count,indices,values);
	auto error = cudaGetLastError();
	return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}
