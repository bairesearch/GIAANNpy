#include "gather.h"
#include <algorithm>
#include <memory>

namespace {
constexpr int validationTile = 512, parallelThreshold = 16384, parallelQueries = 1024;
// Build linear keys while checking all coordinate bounds. The input is already
// canonical, so its keys can be searched directly when the order check passes.
template <int Rank>
int gatherHost(const GatherView& source, const GatherView& query, const float* values, int64_t valueStride, int64_t n, int64_t m, const int64_t* shape, int dims, float* output, int threads) {
	const int rank = Rank ? Rank : dims;
	std::unique_ptr<uint64_t[]> storage(new uint64_t[n]);
	auto keys = storage.get();
	int invalid = 0;
#pragma omp parallel for num_threads(threads) if (threads > 1 && n >= parallelThreshold) reduction(| : invalid)
	for (int64_t start = 0; start < n; start += validationTile) {
		const int end = std::min<int64_t>(n - start, validationTile);
		auto dst = keys + start;
		if constexpr (Rank == 4 || Rank == 5) {
			if (source.entry == 1 && shape[0] == 1 && shape[1] == 1) {
				const auto src = source.data + start;
				const auto row = source.row;
				const uint64_t s2 = shape[2], s3 = shape[3];
				const uint64_t s4 = Rank == 5 ? shape[4] : 1;
#pragma omp simd reduction(| : invalid)
				for (int k = 0; k < end; ++k) {
					const uint64_t a = src[2 * row + k], b = src[3 * row + k];
					invalid |= (src[k] | src[row + k]) != 0 || a >= s2 || b >= s3;
					auto key = a * s3 + b;
					if constexpr (Rank == 5) {
						const uint64_t c = src[4 * row + k];
						invalid |= c >= s4;
						key = key * s4 + c;
					}
					dst[k] = key;
				}
				continue;
			}
		}
		for (int d = 0; d < rank; ++d) {
			const uint64_t size = shape[d];
			const int64_t* row = source.data + d * source.row + start * source.entry;
			if (source.entry == 1) {
#pragma omp simd reduction(| : invalid)
				for (int k = 0; k < end; ++k) {
					uint64_t x = row[k];
					invalid |= x >= size;
					dst[k] = d ? dst[k] * size + x : x;
				}
			} else {
				for (int k = 0; k < end; ++k) {
					uint64_t x = row[k * source.entry];
					invalid |= x >= size;
					dst[k] = d ? dst[k] * size + x : x;
				}
			}
		}
	}
#pragma omp parallel for num_threads(threads) if (threads > 1 && n >= parallelThreshold) reduction(| : invalid)
	for (int64_t i = 1; i < n; ++i)
		invalid |= keys[i] <= keys[i - 1];
#pragma omp parallel for num_threads(threads) if (threads > 1 && m >= parallelQueries) reduction(| : invalid)
	for (int64_t j = 0; j < m; ++j) {
		uint64_t key = 0;
		for (int d = 0; d < rank; ++d) {
			const uint64_t item = query.at(d, j), size = shape[d];
			invalid |= item >= size;
			key = key * size + item;
		}
		auto hit = std::lower_bound(keys, (keys + n), key);
		output[j] = hit != (keys + n) && *hit == key ? values[(hit - keys) * valueStride] : 0.f;
	}
	return invalid;
}
} // namespace
extern "C" int coo_gather_host(const int64_t* indices, int64_t row, int64_t entry, const float* values, int64_t valueStride, int64_t n, const int64_t* queries, int64_t queryRow, int64_t queryEntry, int64_t m, const int64_t* sizes, int rank, float* out, int threads) {
	const GatherView source{indices, row, entry}, query{queries, queryRow, queryEntry};
	int invalid = 0;
	try {
		if (rank == 4)
			invalid = gatherHost<4>(source, query, values, valueStride, n, m, sizes, rank, out, threads);
		else if (rank == 5)
			invalid = gatherHost<5>(source, query, values, valueStride, n, m, sizes, rank, out, threads);
		else
			invalid = gatherHost<0>(source, query, values, valueStride, n, m, sizes, rank, out, threads);
	} catch (const std::bad_alloc&) {
		invalid = 1;
	}
	return invalid;
}
