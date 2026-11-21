#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// Canonical PyTorch COO stores one row per dimension. Tile the sorted union on
// unique pivots, count overlaps, then emit the exact-sized tensor in parallel.
// Slices may have a parent row stride larger than their number of entries.
struct COO {
	const int64_t *idx;
	const float *val;
	const uint64_t *key;
	int64_t row;
};
struct Tile {
	int a, b, count, out;
};
static constexpr int tile_size = 4096;

extern "C" int coo_validate(const int64_t *idx, int64_t row,
														const int64_t *sizes, int rank, int n,
														int threads) {
	int invalid = 0;
#pragma omp parallel for num_threads(threads)                                  \
		reduction(| : invalid) if (threads > 1 && n > tile_size)
	for (int startTile = 0; startTile < (n + int64_t(tile_size) - 1) / tile_size;
			 ++startTile) {
		int start = startTile * tile_size,
				end = start + std::min(tile_size, n - start);
		for (int d = 0; d < rank; ++d) {
			const int64_t *p = idx + d * row;
			const uint64_t size = sizes[d];
#pragma omp simd reduction(| : invalid)
			for (int i = start; i < end; ++i)
				invalid |= uint64_t(p[i]) >= size;
		}
	}
	return invalid;
}

static int countMatches(const uint64_t *a, int na, const uint64_t *b, int nb) {
	int i = 0, j = 0, matches = 0;
	while (i < na && j < nb) {
		if (a[i] < b[j]) {
			++i;
			if (i + int64_t(16) < na && a[i + 16] < b[j])
				i = std::lower_bound(a + i + 16, a + na, b[j]) - a;
		} else if (b[j] < a[i]) {
			++j;
			if (j + int64_t(16) < nb && b[j + 16] < a[i])
				j = std::lower_bound(b + j + 16, b + nb, a[i]) - b;
		} else {
			++i;
			++j;
			++matches;
		}
	}
	return matches;
}

// The caller owns each buffer. This pass computes partitions and the exact
// output size. Saving split points lets the emit pass avoid a second
// intersection search.
extern "C" int coo_partition(const uint64_t *a, int na, const uint64_t *b,
														 int nb, Tile *tiles, int nt, int threads) {
#pragma omp parallel for num_threads(threads) if (threads > 1)
	for (int t = 0; t <= nt; ++t) {
		int i, j;
		if (t == 0) {
			i = 0;
			j = 0;
		} else if (t == nt) {
			i = na;
			j = nb;
		} else if (na >= nb) {
			i = int(int64_t(na) * t / nt);
			j = std::lower_bound(b, b + nb, a[i]) - b;
		} else {
			j = int(int64_t(nb) * t / nt);
			i = std::lower_bound(a, a + na, b[j]) - a;
		}
		tiles[t] = {i, j, 0, 0};
	}
#pragma omp parallel for num_threads(threads) if (threads > 1)
	for (int t = 0; t < nt; ++t) {
		auto &p = tiles[t];
		const auto &q = tiles[t + 1];
		int x = q.a - p.a, y = q.b - p.b;
		p.count = x + y - countMatches(a + p.a, x, b + p.b, y);
	}
	int count = 0;
	for (int t = 0; t < nt; ++t) {
		tiles[t].out = count;
		count += tiles[t].count;
	}
	return count;
}

template <int FixedRank>
static void emitTile(const COO &a, const COO &b, int rank, const Tile &p,
										 const Tile &q, int64_t *idx, float *val, int total) {
	const int dims = FixedRank ? FixedRank : rank;
	int i = p.a, j = p.b, k = p.out;
	auto copyRun = [&](const COO &s, int start, int length) {
		if (length == 1) {
			for (int d = 0; d < dims; ++d)
				idx[int64_t(d) * total + k] = s.idx[d * s.row + start];
			val[k] = s.val[start];
		} else if (length > 1) {
			for (int d = 0; d < dims; ++d)
				std::memcpy(idx + int64_t(d) * total + k, s.idx + d * s.row + start,
										size_t(length) * sizeof(*idx));
			std::memcpy(val + k, s.val + start, size_t(length) * sizeof(*val));
		}
		k += length;
	};
	while (i < q.a && j < q.b) {
		if (a.key[i] < b.key[j]) {
			int start = i++;
			if (i < q.a && a.key[i] < b.key[j])
				i = std::lower_bound(a.key + i, a.key + q.a, b.key[j]) - a.key;
			copyRun(a, start, i - start);
		} else if (b.key[j] < a.key[i]) {
			int start = j++;
			if (j < q.b && b.key[j] < a.key[i])
				j = std::lower_bound(b.key + j, b.key + q.b, a.key[i]) - b.key;
			copyRun(b, start, j - start);
		} else {
			for (int d = 0; d < dims; ++d)
				idx[int64_t(d) * total + k] = a.idx[d * a.row + i];
			val[k++] = a.val[i++] + b.val[j++];
		}
	}
	copyRun(a, i, q.a - i);
	copyRun(b, j, q.b - j);
}
extern "C" void coo_emit(const int64_t *ai, const float *av, const uint64_t *ak,
												 int64_t as, const int64_t *bi, const float *bv,
												 const uint64_t *bk, int64_t bs, int rank,
												 const Tile *tiles, int nt, int threads, int64_t *idx,
												 float *val, int total) {
	const COO a{ai, av, ak, as}, b{bi, bv, bk, bs};
#pragma omp parallel for num_threads(threads) if (threads > 1)
	for (int t = 0; t < nt; ++t) {
		if (rank == 4)
			emitTile<4>(a, b, rank, tiles[t], tiles[t + 1], idx, val, total);
		else if (rank == 5)
			emitTile<5>(a, b, rank, tiles[t], tiles[t + 1], idx, val, total);
		else
			emitTile<0>(a, b, rank, tiles[t], tiles[t + 1], idx, val, total);
	}
}

// For each (branch,segment) fiber, locate one (concept,feature) by binary
// search.
extern "C" int coo_find_neuron(const int64_t *idx, int64_t row, int n,
															 int branches, int segments, int64_t column,
															 int64_t feature, int64_t *positions) {
	int hits = 0, begin = 0;
	for (int b = 0; b < branches && begin < n; ++b) {
		int branchEnd = std::upper_bound(idx + begin, idx + n, b) - idx;
		for (int s = 0; s < segments && begin < branchEnd; ++s) {
			int end = std::upper_bound(idx + row + begin, idx + row + branchEnd, s) -
								(idx + row);
			auto columns = idx + 2 * row, features = idx + 3 * row;
			auto c = std::equal_range(columns + begin, columns + end, column);
			int first = c.first - columns, last = c.second - columns;
			auto pos = std::lower_bound(features + first, features + last, feature);
			if (pos != features + last && *pos == feature)
				positions[hits++] = pos - features;
			begin = end;
		}
	}
	return hits;
}

extern "C" int coo_nonnegative(const float *v, int n, int threads) {
	int invalid = 0;
#pragma omp parallel for simd num_threads(                                     \
				threads) if (threads > 1 && n > tile_size) reduction(| : invalid)
	for (int i = 0; i < n; ++i)
		invalid |= !(v[i] >= 0.0f && std::isfinite(v[i]));
	return invalid;
}

// Both state fibers and eligible keys are sorted. Scan only requested fibers,
// jumping ahead for widely separated keys. Emit canonical positions directly.
extern "C" int coo_candidates(const int64_t *idx, int64_t row, int n,
															const int64_t *dims, const int64_t *keys, int nq,
															const int64_t *segments, int ns, int64_t *out) {
	int hits = 0, start = 0, begin = 0;
	int64_t neurons = dims[2] * dims[3];
	const auto col = idx + 2 * row, feature = idx + 3 * row;
	for (int b = 0; b < dims[0] && start < n && begin < nq; ++b) {
		int end = std::upper_bound(idx + start, idx + n, b) - idx;
		int stop =
				std::lower_bound(keys + begin, keys + nq, (b + 1) * neurons) - keys;
		for (int s = 0; s < ns && begin < stop; ++s) {
			auto fiber =
					std::equal_range(idx + row + start, idx + row + end, segments[s]);
			int i = fiber.first - (idx + row), last = fiber.second - (idx + row),
					j = begin;
			auto key = [&](int k) {
				return b * neurons + col[k] * dims[3] + feature[k];
			};
			while (i < last && j < stop) {
				int64_t a = key(i), q = keys[j];
				if (a < q) {
					++i;
					if (i + int64_t(16) < last && key(i + 16) < q) {
						int lo = i + 16, hi = last;
						while (lo < hi) {
							int m = lo + (hi - lo) / 2;
							if (key(m) < q)
								lo = m + 1;
							else
								hi = m;
						}
						i = lo;
					}
				} else if (a > q) {
					++j;
					if (j + int64_t(16) < stop && keys[j + 16] < a)
						j = std::lower_bound(keys + j + 16, keys + stop, a) - keys;
				} else {
					out[hits++] = i++;
					++j;
				}
			}
		}
		start = end;
		begin = stop;
	}
	return hits;
}

// A small training update needs only one insertion point per update. Read
// coordinates in place and copy the untouched parts of every index row in bulk.
extern "C" int coo_delta_positions(const int64_t *base, int64_t step, int n,
																	 const int64_t *updates, int64_t us, int m,
																	 int rank, int32_t *pos, int32_t *out) {
	int overlap = 0, first = 0;
	for (int j = 0; j < m; ++j) {
		int lo = first, hi = n;
		for (int d = 0; d < rank && lo < hi; ++d) {
			const int64_t *axis = base + d * step;
			auto bounds = std::equal_range(axis + lo, axis + hi, updates[d * us + j]);
			lo = bounds.first - axis;
			hi = bounds.second - axis;
		}
		pos[j] = lo;
		out[j] = lo + j - overlap;
		first = lo + (hi > lo);
		overlap += hi > lo;
	}
	return n + m - overlap;
}
extern "C" int coo_delta_emit(const int64_t *base, int64_t step,
															const float *bv, int n, const int64_t *update,
															int64_t us, const float *uv, int m, int rank,
															const int32_t *pos, const int32_t *out, int total,
															int threads, int64_t *idx, float *values,
															bool baseFirst, const uint64_t *shape) {
	int invalid = 0;
	// Split every coordinate row into independent output windows so a few
	// large rows can still use all workers. The output offset of the next
	// delta determines which unchanged source element starts a window.
	constexpr int window = 32768;
	int64_t windows = (int64_t(total) + window - 1) / window;
#pragma omp parallel for num_threads(threads) if (threads > 1)                 \
		reduction(| : invalid)
	for (int64_t task = 0; task < windows * (rank + 1); ++task) {
		int d = task / windows, to = (task % windows) * window,
				end = to + std::min(window, total - to);
		int j = std::lower_bound(out, out + m, to) - out;
		int from = j == m ? n - (total - to) : pos[j] - (out[j] - to);
		bool floats = d == rank;
		// Fuse the source bounds check with the compulsory output copy.
		auto copy = [&](int begin, int target, int length) {
			if (floats) {
				std::memcpy(values + target, bv + begin,
										size_t(length) * sizeof(float));
			} else {
				auto *dst = idx + int64_t(d) * total + target;
				const auto *src = base + d * step + begin;
				const auto size = shape[d];
				int bad = 0;
#pragma omp simd reduction(| : bad)
				for (int k = 0; k < length; ++k) {
					dst[k] = src[k];
					bad |= uint64_t(src[k]) >= size;
				}
				invalid |= bad;
			}
		};
		for (; j < m && out[j] < end; ++j) {
			const int p = pos[j], target = out[j], span = target - to;
			copy(from, to, span);
			bool match = j + 1 < m ? (pos[j + 1] - p > out[j + 1] - target - 1)
														 : (n - p > total - target - 1);
			if (floats)
				values[target] =
						match ? (baseFirst ? bv[p] + uv[j] : uv[j] + bv[p]) : uv[j];
			else {
				idx[int64_t(d) * total + target] = update[d * us + j];
				invalid |= uint64_t(update[d * us + j]) >= shape[d];
				if (match)
					invalid |= uint64_t(base[d * step + p]) >= shape[d];
			}
			from = p + match;
			to = target + 1;
		}
		copy(from, to, end - to);
	}
	return invalid;
}

extern "C" int coo_decay_plan(const int64_t *segments, const float *old, int n,
															int64_t soma, float rate, bool multiply,
															int threads, float *scaled, int32_t *offset,
															int nt, int tile) {
	int invalid = 0;
#pragma omp parallel for num_threads(threads) if (threads > 1)                 \
		reduction(| : invalid)
	for (int t = 0; t < nt; ++t) {
		int start = t * tile, end = start + std::min(tile, n - start), positive = 0;
#pragma omp simd reduction(| : invalid) reduction(+ : positive)
		for (int i = start; i < end; ++i) {
			float x = old[i];
			invalid |= !(std::isfinite(x) && x >= 0.f);
			float y = segments[i] == soma
										? (multiply ? x * rate : std::max(0.f, x - rate))
										: x;
			scaled[i] = y;
			positive += y > 0;
		}
		offset[t] = positive;
	}
	int total = 0;
	for (int t = 0; t < nt; ++t) {
		int count = offset[t];
		offset[t] = total;
		total += count;
	}
	offset[nt] = total;
	return invalid ? -1 : total;
}
extern "C" void coo_decay_emit(const int64_t *idx, int64_t stride, int rank,
															 const float *scaled, int n,
															 const int32_t *offset, int nt, int tile,
															 int threads, int64_t *out, float *v, int total) {
#pragma omp parallel for num_threads(threads) if (threads > 1)
	for (int t = 0; t < nt; ++t) {
		int start = t * tile, end = start + std::min(tile, n - start),
				dst = offset[t], count = offset[t + 1] - dst;
		if (count == end - start) {
			for (int d = 0; d < rank; ++d)
				std::copy_n(idx + d * stride + start, count,
										out + int64_t(d) * total + dst);
			std::copy_n(scaled + start, count, v + dst);
		} else if (count) {
			for (int d = 0; d < rank; ++d) {
				int k = dst;
				const int64_t *src = idx + d * stride;
				int64_t *target = out + int64_t(d) * total;
				for (int i = start; i < end; ++i)
					if (scaled[i] > 0)
						target[k++] = src[i];
			}
			for (int i = start; i < end; ++i)
				if (scaled[i] > 0)
					v[dst++] = scaled[i];
		}
	}
}

// PyTorch sparse add initializes unmatched destinations with +0, and its
// duplicate reduction can choose a different NaN payload. Preserve the stock
// operation for any operand whose bit pattern makes that distinction visible.
static bool specialFloat(float value) {
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	const uint32_t magnitude = bits & 0x7fffffffu;
	return bits == 0x80000000u || (magnitude != 0 && magnitude < 0x00800000u) || magnitude >= 0x7f800000u;
}

extern "C" int coo_special_values(const float *values, int64_t stride, int n, int threads) {
	int special = 0;
	if (stride == 1) {
#pragma omp parallel for simd num_threads(threads) if(threads > 1 && n > tile_size) reduction(|:special)
		for (int i = 0; i < n; ++i)
			special |= specialFloat(values[i]);
	} else {
		for (int i = 0; i < n && !special; ++i)
			special = specialFloat(values[int64_t(i) * stride]);
	}
	return special;
}
