// Ordered float32 updates over packed int32 resident connection records.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

std::vector<at::Tensor> mergePackedSequenceBatch(bool optimisationStoreDatabaseResidentCoordinatesAsInt32, const std::vector<at::Tensor>& sources, const std::vector<at::Tensor>& packets, const std::vector<at::Tensor>& buckets, const std::vector<int64_t>& shape, int64_t bucketDimension, int64_t grain, int64_t zero, int64_t one, int64_t recordRank, int64_t sourceRank, int64_t coordinateMaximum, const std::string& invalid)
{
	std::vector<at::Tensor> result(sources.size());
	TORCH_CHECK(optimisationStoreDatabaseResidentCoordinatesAsInt32, invalid);
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32)
	{
		TORCH_CHECK(zero == int64_t{} && one == sizeof(char) && recordRank == one + one && sourceRank > zero && coordinateMaximum == std::numeric_limits<int32_t>::max(), invalid);
		const int64_t rank = shape.size();
		const int64_t width = rank + one;
		TORCH_CHECK(rank == sourceRank && bucketDimension >= zero && bucketDimension <= rank && grain > zero && packets.size() == buckets.size(), invalid);
		int64_t capacity = one;
		for(const auto size : shape)
		{
			TORCH_CHECK(size > zero && size <= coordinateMaximum + one && capacity <= std::numeric_limits<int64_t>::max() / size, invalid);
			capacity *= size;
		}
		struct Entry { int64_t key; float value; };
		std::vector<std::vector<Entry>> entries(sources.size());
		for(size_t packet = zero; packet < packets.size(); ++packet)
		{
			const auto& update = packets[packet];
			const auto& mapping = buckets[packet];
			TORCH_CHECK(update.device().is_cpu() && update.layout() == at::kSparse && update.scalar_type() == at::kFloat && update.sparse_dim() == width && update.dense_dim() == zero && update.is_coalesced() && !update.requires_grad(), invalid);
			TORCH_CHECK(mapping.device().is_cpu() && mapping.layout() == at::kStrided && mapping.scalar_type() == at::kLong && mapping.dim() == one && mapping.numel() == update.size(bucketDimension), invalid);
			for(int64_t dimension = zero; dimension < rank; ++dimension) TORCH_CHECK(update.size(dimension + (dimension >= bucketDimension)) == shape[dimension], invalid);
			const auto indexTensor = update.indices();
			const auto valueTensor = update.values();
			auto indices = indexTensor.accessor<int64_t, 2>();
			auto values = valueTensor.accessor<float, 1>();
			auto map = mapping.accessor<int64_t, 1>();
			for(int64_t bucket = zero; bucket < mapping.numel(); ++bucket) TORCH_CHECK(map[bucket] >= zero && map[bucket] < int64_t(sources.size()) && (bucket == zero || map[bucket] > map[bucket - one]), invalid);
			for(int64_t position = zero; position < update._nnz(); ++position)
			{
				const int64_t local = indices[bucketDimension][position];
				TORCH_CHECK(local >= zero && local < mapping.numel() && std::isfinite(values[position]), invalid);
				int64_t key = zero;
				for(int64_t dimension = zero; dimension < rank; ++dimension)
				{
					const int64_t coordinate = indices[dimension + (dimension >= bucketDimension)][position];
					TORCH_CHECK(coordinate >= zero && coordinate < shape[dimension], invalid);
					key = key * shape[dimension] + coordinate;
				}
				if(position > zero)
				{
					int64_t firstDifferent = zero;
					while(firstDifferent < width && indices[firstDifferent][position - one] == indices[firstDifferent][position]) ++firstDifferent;
					TORCH_CHECK(firstDifferent < width && indices[firstDifferent][position - one] < indices[firstDifferent][position], invalid);
				}
				entries[map[local]].push_back({key, values[position]});
			}
		}
		for(const auto& source : sources)
		{
			TORCH_CHECK(source.device().is_cpu() && source.layout() == at::kStrided && source.scalar_type() == at::kInt && source.dim() == recordRank && source.size(one) == width && source.is_contiguous(), invalid);
		}
		std::atomic<int64_t> next{zero};
		at::parallel_for(zero, sources.size(), grain, [&](int64_t, int64_t)
		{
			int64_t bucket = next.fetch_add(one, std::memory_order_relaxed);
			while(bucket < int64_t(sources.size()))
			{
				auto& updates = entries[bucket];
				std::stable_sort(updates.begin(), updates.end(), [](const Entry& left, const Entry& right) { return left.key < right.key; });
				const auto& source = sources[bucket];
				const auto* oldRecords = source.const_data_ptr<int32_t>();
				const int64_t oldCount = source.size(zero);
				int64_t previous = -one;
				for(int64_t position = zero; position < oldCount; ++position)
				{
					int64_t key = zero;
					for(int64_t dimension = zero; dimension < rank; ++dimension)
					{
						const int64_t coordinate = oldRecords[position * width + dimension];
						TORCH_CHECK(coordinate >= zero && coordinate < shape[dimension], invalid);
						key = key * shape[dimension] + coordinate;
					}
					float value;
					std::memcpy(&value, oldRecords + position * width + rank, sizeof(value));
					TORCH_CHECK(key > previous && std::isfinite(value), invalid);
					previous = key;
				}
				auto oldKey = [&](int64_t position)
				{
					int64_t key = zero;
					for(int64_t dimension = zero; dimension < rank; ++dimension) key = key * shape[dimension] + oldRecords[position * width + dimension];
					return key;
				};
				std::vector<int64_t> starts;
				std::vector<int64_t> insertions;
				std::vector<bool> overlaps;
				int64_t extra = zero;
				int64_t lower = zero;
				for(int64_t position = zero; position < int64_t(updates.size()); ++position)
				{
					if(position == zero || updates[position].key != updates[position - one].key)
					{
						int64_t upper = oldCount;
						while(lower < upper)
						{
							const int64_t middle = lower + (upper - lower) / recordRank;
							if(oldKey(middle) < updates[position].key) lower = middle + one;
							else upper = middle;
						}
						const bool overlap = lower < oldCount && oldKey(lower) == updates[position].key;
						starts.push_back(position);
						insertions.push_back(lower);
						overlaps.push_back(overlap);
						extra += !overlap;
					}
				}
				starts.push_back(updates.size());
				TORCH_CHECK(extra <= std::numeric_limits<int64_t>::max() - oldCount, invalid);
				const int64_t count = oldCount + extra;
				TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() / width / sizeof(int32_t), invalid);
				auto outputTensor = at::empty({count, width}, source.options());
				auto* output = outputTensor.mutable_data_ptr<int32_t>();
				auto copy = [&](int64_t begin, int64_t end, int64_t destination)
				{
					if(end > begin) std::memcpy(output + destination * width, oldRecords + begin * width, (end - begin) * width * sizeof(int32_t));
				};
				int64_t oldPosition = zero;
				int64_t position = zero;
				for(int64_t group = zero; group < int64_t(insertions.size()); ++group)
				{
					const int64_t insertion = insertions[group];
					copy(oldPosition, insertion, position);
					position += insertion - oldPosition;
					int64_t key = updates[starts[group]].key;
					for(int64_t dimension = rank - one; dimension >= zero; --dimension)
					{
						output[position * width + dimension] = int32_t(key % shape[dimension]);
						key /= shape[dimension];
					}
					int64_t updatePosition = starts[group];
					float value;
					if(overlaps[group]) std::memcpy(&value, oldRecords + insertion * width + rank, sizeof(value));
					else value = updates[updatePosition++].value;
					for(; updatePosition < starts[group + one]; ++updatePosition) value += updates[updatePosition].value;
					TORCH_CHECK(std::isfinite(value), invalid);
					std::memcpy(output + position * width + rank, &value, sizeof(value));
					++position;
					oldPosition = insertion + overlaps[group];
				}
				copy(oldPosition, oldCount, position);
				position += oldCount - oldPosition;
				TORCH_CHECK(position == count, invalid);
				result[bucket] = outputTensor;
				bucket = next.fetch_add(one, std::memory_order_relaxed);
			}
		});
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	module.def("mergePackedSequenceBatch", &mergePackedSequenceBatch, pybind11::call_guard<pybind11::gil_scoped_release>());
}
