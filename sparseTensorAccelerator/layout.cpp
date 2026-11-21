#include <algorithm>
#include <cstdint>

// Flat keys are scratch. Peel off one axis at a time in cache-sized tiles.
// For a non-power-of-two divisor d, floor(2^64/d)*x >> 64 misses x/d by
// at most one. The remainder comparison supplies the exact integer correction.
extern "C" void cpu_unpack(uint64_t *keys, uint64_t *indices, const uint64_t *shape, int rank, int n, int threads) {
	if (!n) return;
	constexpr int grain = 512;
	// PyTorch supports at most 64 dimensions.
	uint64_t reciprocal[64] = {};
	for (int d=0; d<rank; ++d) {
		if (shape[d] & (shape[d]-1)) reciprocal[d] = (static_cast<__uint128_t>(1) << 64) / shape[d];
	}
	#pragma omp parallel for num_threads(threads) if(threads>1 && n>=16384)
	for (int tile=0; tile<(n-1)/grain+1; ++tile) {
		int start=tile*grain, length=std::min(grain,n-start);
		uint64_t *local = keys+start;
		for (int d=rank-1; d>=0; --d) {
			uint64_t *dst=indices+size_t(d)*n+start;
			uint64_t size=shape[d];
			if (d==0) std::copy_n(local,length,dst);
			else if (size==1) std::fill_n(dst,length,uint64_t(0));
			else if (!(size & (size-1))) {
				int shift=__builtin_ctzll(size);
				for (int i=0; i<length; ++i) { dst[i]=local[i] & (size-1); local[i] >>= shift; }
			} else {
				for (int i=0; i<length; ++i) {
					uint64_t q=(static_cast<__uint128_t>(local[i])*reciprocal[d])>>64;
					uint64_t r=local[i]-q*size;
					bool carry=r>=size;
					dst[i]=carry ? r-size : r;
					local[i]=q+carry;
				}
			}
		}
	}
}

// Fuse validation with packing in cache-sized chunks. This allows the driver
// to transfer one key per entry even when the host tensor has many index rows.
extern "C" int cpu_pack_rows(const uint64_t *src, uint64_t *dst,
																const uint64_t *shape, int rank, int n,
																int threads, int64_t row_stride) {
	if (!n)
		return 0;
	int invalid = 0;
	constexpr int grain = 512;
#pragma omp parallel for num_threads(threads) if (threads > 1 && n >= 65536)   \
		reduction(| : invalid)
	for (int tile = 0; tile < (n - 1) / grain + 1; ++tile) {
		int start = tile * grain, len = std::min(grain, n - start);
		std::fill_n(dst + start, len, uint64_t(0));
		for (int d = 0; d < rank; ++d) {
			uint64_t size = shape[d];
			const uint64_t *row = src + d * row_stride + start;
			if (size && !(size & (size - 1))) {
				int shift = __builtin_ctzll(size);
#pragma omp simd reduction(| : invalid)
				for (int j = 0; j < len; ++j) {
					invalid |= row[j] >= size;
					dst[start + j] = (dst[start + j] << shift) | row[j];
				}
			} else {
#pragma omp simd reduction(| : invalid)
				for (int j = 0; j < len; ++j) {
					invalid |= row[j] >= size;
					dst[start + j] = dst[start + j] * size + row[j];
				}
			}
		}
	}
	return invalid;
}

extern "C" int cpu_pack_checked(const uint64_t *src, uint64_t *dst, const uint64_t *shape, int rank, int n, int threads) {
	return cpu_pack_rows(src, dst, shape, rank, n, threads, n);
}
