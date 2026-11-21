// Dynamic support path. Sort full (key, value) pairs stably, compact run heads,
// then give each run a single owner to preserve the original FP32 sum order.
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <new>
#include <thrust/iterator/counting_iterator.h>

struct Sorter {
	cudaStream_t stream = nullptr;
	void *arena = nullptr, *scratch = nullptr;
	size_t bytes = 0;
	uint64_t *keys = nullptr, *sorted = nullptr;
	float *values = nullptr, *ordered = nullptr;
	int *heads = nullptr, *info = nullptr;
	int32_t *records = nullptr;
	int capacity;
	~Sorter() {
		if (stream)
			cudaStreamSynchronize(stream);
		if (records)
			cudaFree(records);
		if (scratch)
			cudaFree(scratch);
		if (arena)
			cudaFree(arena);
		if (stream)
			cudaStreamDestroy(stream);
	}
};
struct Head {
	const uint64_t *keys;
	__device__ bool operator()(int i) const {
		return i == 0 || keys[i] != keys[i - 1];
	}
};
// PyTorch CPU may order equal keys differently. For three or more terms,
// permit the GPU result only if every possible partial sum is exact in FP32.
// A 24-bit integer times a shared power of two has this property.
enum { invalid_key = 1, native_reduction = 2 };
template<bool chronological>
__global__ void fold_runs(const uint64_t *keys, const float *v, const int *heads, int *info,
													int n, uint64_t extent, uint64_t *out_k, float *out_v) {
	int group = int(blockIdx.x * blockDim.x + threadIdx.x);
	if (group >= info[0]) return;
	int first = heads[group], end = group+1 == info[0] ? n : heads[group+1];
	bool ambiguous = false;
	int least = 128, highest = -150, nonzero = 0;
	float sum = v[first];
	for (int i = first; i < end; ++i) {
		const float term = v[i];
		if (i != first) sum = __fadd_rn(sum, term);
		if constexpr (chronological) continue;
		const unsigned bits = __float_as_uint(term) & 0x7fffffff;
		const int exponent = bits >> 23;
		if (bits && (exponent == 0 || exponent == 255)) ambiguous = true;
		if (end-first > 2 && bits && exponent != 255 && exponent != 0) {
			const unsigned significand = (bits & 0x7fffff) | 0x800000;
			least = min(least, exponent-150 + __ffs(significand)-1);
			highest = max(highest, exponent-127);
			++nonzero;
		}
	}
	if (nonzero > 2) {
		const int carry = 32 - __clz(nonzero-1);
		ambiguous |= highest-least+1+carry > 24 || highest+carry > 127;
	}
	const unsigned magnitude = __float_as_uint(sum) & 0x7fffffff;
	ambiguous |= magnitude && magnitude < 0x800000;
	if constexpr (!chronological) { if (ambiguous) atomicOr(info+1, native_reduction); }
	if (keys[first] >= extent) atomicOr(info+1, invalid_key);
	out_k[group] = keys[first];
	out_v[group] = sum;
}
extern "C" const char *sorter_create(void **ctx, int capacity) {
	*ctx = nullptr;
	auto *s = new (std::nothrow) Sorter{};
	if (!s)
		return "sorter allocation failed";
#define INIT(expr)                                                             \
	do {                                                                         \
		cudaError_t e = (expr);                                                    \
		if (e != cudaSuccess) {                                                    \
			delete s;                                                                \
			return cudaGetErrorString(e);                                            \
		}                                                                          \
	} while (0)
	s->capacity = capacity;
	INIT(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
	INIT(cudaMalloc(&s->arena, size_t(capacity) * 28 + 8));
	s->keys = static_cast<uint64_t *>(s->arena);
	s->sorted = s->keys + capacity;
	s->values = reinterpret_cast<float *>(s->sorted + capacity);
	s->ordered = s->values + capacity;
	s->heads = reinterpret_cast<int *>(s->ordered + capacity);
	s->info = s->heads + capacity;
	size_t sort_bytes = 0, select_bytes = 0;
	INIT(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, s->keys, s->sorted,
																			 s->values, s->ordered, capacity, 0, 64,
																			 s->stream));
	INIT(cub::DeviceSelect::If(nullptr, select_bytes,
														 thrust::make_counting_iterator(0), s->heads,
														 s->info, capacity, Head{s->sorted}, s->stream));
	s->bytes = std::max(sort_bytes, select_bytes);
	INIT(cudaMalloc(&s->scratch, std::max(s->bytes, size_t(1))));
	*ctx = s;
	return nullptr;
#undef INIT
}
extern "C" void sorter_close(void* ctx) {
	delete static_cast<Sorter*>(ctx);
}
static const char* order_runs(Sorter& s, const uint64_t* keys, const float* values, int n, int bits) {
#define CHECK(expr)                          \
	do {                                     \
		cudaError_t e = (expr);              \
		if (e != cudaSuccess) {              \
			cudaStreamSynchronize(s.stream); \
			return cudaGetErrorString(e);    \
		}                                    \
	} while (0)
	CHECK(cudaMemsetAsync(s.info + 1, 0, 4, s.stream));
	CHECK(cudaMemcpyAsync(s.keys, keys, size_t(n) * 8, cudaMemcpyHostToDevice, s.stream));
	CHECK(cudaMemcpyAsync(s.values, values, size_t(n) * 4, cudaMemcpyHostToDevice,
						  s.stream));
	size_t bytes = s.bytes;
	CHECK(cub::DeviceRadixSort::SortPairs(s.scratch, bytes, s.keys, s.sorted,
										  s.values, s.ordered, n, 0, bits,
										  s.stream));
	bytes = s.bytes;
	CHECK(cub::DeviceSelect::If(s.scratch, bytes,
								thrust::make_counting_iterator(0), s.heads,
								s.info, n, Head{s.sorted}, s.stream));
	return nullptr;
#undef CHECK
}

template <bool chronological>
static const char* run_sort(void* ctx, const uint64_t* keys,
							const float* values, int n, uint64_t size,
							int bits, uint64_t* out_k, float* out_v, int* count) {
	Sorter& s = *static_cast<Sorter*>(ctx);
	if (n > s.capacity || n < 0)
		return "invalid count";
	*count = 0;
	if (!n)
		return nullptr;
#define CHECK(expr)                          \
	do {                                     \
		cudaError_t e = (expr);              \
		if (e != cudaSuccess) {              \
			cudaStreamSynchronize(s.stream); \
			return cudaGetErrorString(e);    \
		}                                    \
	} while (0)
	if (auto error = order_runs(s, keys, values, n, bits))
		return error;
	fold_runs<chronological><<<(n - 1) / 256 + 1, 256, 0, s.stream>>>(
		s.sorted, s.ordered, s.heads, s.info, n, size, s.keys, s.values);
	CHECK(cudaGetLastError());
	int info[2];
	CHECK(cudaMemcpyAsync(info, s.info, 8, cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaStreamSynchronize(s.stream));
	if (info[1] & invalid_key)
		return "invalid COO coordinate";
	if (info[1] & native_reduction) {
		*count = -1;
		return nullptr;
	}
	CHECK(cudaMemcpyAsync(out_k, s.keys, size_t(info[0]) * 8, cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaMemcpyAsync(out_v, s.values, size_t(info[0]) * 4,
						  cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaStreamSynchronize(s.stream));
	*count = info[0];
	return nullptr;
}

// Batch commits receive canonical sequence packets, in training order, after
// the historical value. CUB preserves that order for every equal key.
extern "C" const char* sorter_run(void* ctx, const uint64_t* keys, const float* values, int n, uint64_t size, int bits, uint64_t* out_keys, float* out_values, int* count) {
	return run_sort<false>(ctx, keys, values, n, size, bits, out_keys, out_values, count);
}
extern "C" const char* sorter_run_chronological(void* ctx, const uint64_t* keys, const float* values, int n, uint64_t size, int bits, uint64_t* out_keys, float* out_values, int* count) {
	return run_sort<true>(ctx, keys, values, n, size, bits, out_keys, out_values, count);
}

// The fixed-vocabulary trainer stores (property, branch, segment, concept,
// feature, value) records. Property and branch are singleton axes on this path.
enum { record_width = 6,
	   segment_axis = 2,
	   concept_axis = 3,
	   feature_axis = 4,
	   value_axis = 5 };
__global__ void decode_training_runs(const uint64_t* sorted, const float* ordered, const int* heads, int* info, int n, int source_bits, int concept_bits, int feature_bits, int32_t* records, int* ends, int sources) {
	const int group = int(blockIdx.x * blockDim.x + threadIdx.x);
	if (group < info[0]) {
		const int first = heads[group], stop = group + 1 < info[0] ? heads[group + 1] : n;
		const uint64_t key = sorted[first];
		const uint64_t source = key >> source_bits;
		if (source >= uint64_t(sources)) {
			atomicOr(info + 1, invalid_key);
		} else {
			float value = ordered[first];
			for (int i = first + 1; i < stop; ++i)
				value = __fadd_rn(value, ordered[i]);
			if (!isfinite(value))
				atomicOr(info + 1, native_reduction);
			const uint64_t local = key & ((uint64_t(1) << source_bits) - 1);
			int32_t* row = records + size_t(group) * record_width;
			row[0] = 0;
			row[1] = 0;
			row[segment_axis] = local >> (concept_bits + feature_bits);
			row[concept_axis] = (local >> feature_bits) & ((uint64_t(1) << concept_bits) - 1);
			row[feature_axis] = local & ((uint64_t(1) << feature_bits) - 1);
			row[value_axis] = __float_as_int(value);
			if (stop == n || (sorted[stop] >> source_bits) != source)
				ends[source] = group + 1;
		}
	}
}
extern "C" const char* sorter_run_training_records(void* ctx, const uint64_t* keys, const float* values, int n, int bits, int source_bits, int concept_bits, int feature_bits, int sources, int32_t* rows, int* ends, int* count) {
	Sorter& s = *static_cast<Sorter*>(ctx);
	if (n <= 0 || n > s.capacity || sources <= 0 || sources > s.capacity || source_bits >= 63 || source_bits < concept_bits + feature_bits || concept_bits < 0 || feature_bits < 0)
		return "invalid training tile";
#define CHECK(expr)                          \
	do {                                     \
		cudaError_t e = (expr);              \
		if (e != cudaSuccess) {              \
			cudaStreamSynchronize(s.stream); \
			return cudaGetErrorString(e);    \
		}                                    \
	} while (0)
	if (!s.records)
		CHECK(cudaMalloc(&s.records, size_t(s.capacity) * (record_width + 1) * sizeof(int32_t)));
	int* device_ends = s.records + size_t(s.capacity) * record_width;
	if (auto error = order_runs(s, keys, values, n, bits))
		return error;
	CHECK(cudaMemsetAsync(device_ends, 0, sources * sizeof(int), s.stream));
	decode_training_runs<<<(n - 1) / 256 + 1, 256, 0, s.stream>>>(s.sorted, s.ordered, s.heads, s.info, n, source_bits, concept_bits, feature_bits, s.records, device_ends, sources);
	CHECK(cudaGetLastError());
	int info[2];
	CHECK(cudaMemcpyAsync(info, s.info, sizeof(info), cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaStreamSynchronize(s.stream));
	if (info[1])
		return "invalid coordinate or nonfinite chronological training sum";
	*count = info[0];
	CHECK(cudaMemcpyAsync(rows, s.records, size_t(*count) * record_width * sizeof(int32_t), cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaMemcpyAsync(ends, device_ends, sources * sizeof(int), cudaMemcpyDeviceToHost, s.stream));
	CHECK(cudaStreamSynchronize(s.stream));
	return nullptr;
#undef CHECK
}
