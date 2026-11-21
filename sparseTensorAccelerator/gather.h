// Coordinates may have arbitrary strides. Compare in COO order without flattening.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define GIAANN_COORD_INLINE __host__ __device__ inline
#else
#define GIAANN_COORD_INLINE inline
#endif
struct GatherView {
	const int64_t* data;
	int64_t row, entry;
	GIAANN_COORD_INLINE int64_t at(int d, int64_t i) const {
		return data[d * row + i * entry];
	}
};
constexpr int gatherMaxRank = 64;
struct GatherShape {
	int64_t sizes[gatherMaxRank];
	int rank;
};
GIAANN_COORD_INLINE bool gatherValid(const GatherView& src, int64_t i, const GatherShape& shape, bool checkOrder) {
	int compare = 0;
	bool valid = true;
	for (int d = 0; d < shape.rank; ++d) {
		const auto cur = src.at(d, i);
		valid &= uint64_t(cur) < uint64_t(shape.sizes[d]);
		if (checkOrder && !compare) {
			const auto prev = src.at(d, i - 1);
			compare = (cur > prev) - (cur < prev);
		}
	}
	return valid && (!checkOrder || compare > 0);
}
GIAANN_COORD_INLINE int gatherCompare(const GatherView& src, int64_t i, const GatherView& query, int64_t j, int rank) {
	int result = 0;
	for (int d = 0; d < rank && !result; ++d) {
		const auto left = src.at(d, i), right = query.at(d, j);
		result = (left > right) - (left < right);
	}
	return result;
}
GIAANN_COORD_INLINE float gatherAt(const GatherView& src, const GatherView& query, const float* values, int64_t stride, int64_t n, int64_t j, int rank) {
	int64_t lo = 0, hi = n;
	while (lo < hi) {
		const auto mid = lo + (hi - lo) / 2;
		if (gatherCompare(src, mid, query, j, rank) < 0)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo < n && gatherCompare(src, lo, query, j, rank) == 0 ? values[lo * stride] : 0.0f;
}
#undef GIAANN_COORD_INLINE
