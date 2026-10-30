// Ordered sequence-batch training kernels. Python supplies feature switches, limits and errors.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>
#include <optional>
#include <string>

std::vector<at::Tensor> mergeSequenceBatch(bool optimiseTrainSequenceBatch, const std::vector<at::Tensor>& sources, const std::vector<at::Tensor>& packets, const std::vector<at::Tensor>& buckets, const std::vector<int64_t>& shape, int64_t bucketDimension, int64_t grain, const std::string& invalid)
{
	std::vector<at::Tensor> result(sources.size());
	TORCH_CHECK(optimiseTrainSequenceBatch, invalid);
	if(optimiseTrainSequenceBatch)
	{
		const int64_t rank = shape.size();
		TORCH_CHECK(rank > 0 && bucketDimension >= 0 && bucketDimension <= rank && grain > 0 && packets.size() == buckets.size(), invalid);
		int64_t capacity = 1;
		for(const auto size : shape)
		{
			TORCH_CHECK(size > 0 && capacity <= std::numeric_limits<int64_t>::max() / size, invalid);
			capacity *= size;
		}
		auto validate = [&](const at::Tensor& tensor, int64_t dimensions)
		{
			TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat && tensor.sparse_dim() == dimensions && tensor.dense_dim() == 0 && tensor.is_coalesced() && !tensor.requires_grad(), invalid);
		};
		struct Entry { int64_t key; float value; };
		std::vector<std::vector<Entry>> entries(sources.size());
		for(size_t packet = 0; packet < packets.size(); ++packet)
		{
			const auto& update = packets[packet];
			const auto& mapping = buckets[packet];
			validate(update, rank + 1);
			TORCH_CHECK(mapping.device().is_cpu() && mapping.scalar_type() == at::kLong && mapping.dim() == 1 && mapping.numel() == update.size(bucketDimension), invalid);
			for(int64_t dimension = 0; dimension < rank; ++dimension) TORCH_CHECK(update.size(dimension + (dimension >= bucketDimension)) == shape[dimension], invalid);
			const auto indexTensor = update.indices();
			const auto valueTensor = update.values();
			auto indices = indexTensor.accessor<int64_t, 2>();
			auto values = valueTensor.accessor<float, 1>();
			auto map = mapping.accessor<int64_t, 1>();
			for(int64_t bucket = 0; bucket < mapping.numel(); ++bucket) TORCH_CHECK(map[bucket] >= 0 && map[bucket] < int64_t(sources.size()) && (bucket == 0 || map[bucket] > map[bucket - 1]), invalid);
			for(int64_t position = 0; position < update._nnz(); ++position)
			{
				const int64_t local = indices[bucketDimension][position];
				TORCH_CHECK(local >= 0 && local < mapping.numel() && std::isfinite(values[position]), invalid);
				int64_t key = 0;
				for(int64_t dimension = 0; dimension < rank; ++dimension)
				{
					const int64_t coordinate = indices[dimension + (dimension >= bucketDimension)][position];
					TORCH_CHECK(coordinate >= 0 && coordinate < shape[dimension], invalid);
					key = key * shape[dimension] + coordinate;
				}
				entries[map[local]].push_back({key, values[position]});
			}
		}
		for(const auto& source : sources)
		{
			validate(source, rank);
			for(int64_t dimension = 0; dimension < rank; ++dimension) TORCH_CHECK(source.size(dimension) == shape[dimension], invalid);
		}
		std::atomic<int64_t> next{0};
		at::parallel_for(0, sources.size(), grain, [&](int64_t, int64_t)
		{
			int64_t bucket = next.fetch_add(1, std::memory_order_relaxed);
			while(bucket < int64_t(sources.size()))
			{
				auto& updates = entries[bucket];
				// Stable order keeps each float32 addition in its original sequence order.
				std::stable_sort(updates.begin(), updates.end(), [](const Entry& left, const Entry& right) { return left.key < right.key; });
				const auto& source = sources[bucket];
				const auto oldIndexTensor = source.indices();
				const auto oldValueTensor = source.values();
				auto oldIndices = oldIndexTensor.accessor<int64_t, 2>();
				auto oldValues = oldValueTensor.accessor<float, 1>();
				const int64_t oldCount = source._nnz();
				// Validate complete historical coordinate rows once, including coordinates updated in place in the output order.
				// Keep failure checks outside the reduction loops so the compiler can vectorise the scan.
				for(int64_t dimension = 0; dimension < rank; ++dimension)
				{
					int64_t minimum = std::numeric_limits<int64_t>::max();
					int64_t maximum = std::numeric_limits<int64_t>::min();
					for(int64_t position = 0; position < oldCount; ++position)
					{
						minimum = std::min(minimum, oldIndices[dimension][position]);
						maximum = std::max(maximum, oldIndices[dimension][position]);
					}
					TORCH_CHECK(minimum >= 0 && maximum < shape[dimension], invalid);
				}
				auto oldKey = [&](int64_t position)
				{
					int64_t key = 0;
					for(int64_t dimension = 0; dimension < rank; ++dimension) key = key * shape[dimension] + oldIndices[dimension][position];
					return key;
				};
				std::vector<int64_t> starts;
				std::vector<int64_t> insertions;
				std::vector<bool> overlaps;
				int64_t extra = 0;
				int64_t lower = 0;
				for(int64_t position = 0; position < int64_t(updates.size()); ++position)
				{
					if(position == 0 || updates[position].key != updates[position - 1].key)
					{
						int64_t upper = source._nnz();
						while(lower < upper)
						{
							const int64_t middle = lower + (upper - lower) / 2;
							if(oldKey(middle) < updates[position].key) lower = middle + 1;
							else upper = middle;
						}
						const bool overlap = lower < source._nnz() && oldKey(lower) == updates[position].key;
						starts.push_back(position);
						insertions.push_back(lower);
						overlaps.push_back(overlap);
						extra += !overlap;
					}
				}
				starts.push_back(updates.size());
				TORCH_CHECK(extra <= std::numeric_limits<int64_t>::max() - source._nnz(), invalid);
				const int64_t count = source._nnz() + extra;
				const bool sharedIndices = extra == 0;
				auto indexTensor = sharedIndices ? oldIndexTensor : at::empty({rank, count}, oldIndexTensor.options());
				auto valueTensor = at::empty({count}, oldValueTensor.options());
				auto indices = indexTensor.accessor<int64_t, 2>();
				auto values = valueTensor.accessor<float, 1>();
				auto copy = [&](int64_t begin, int64_t end, int64_t output)
				{
					for(int64_t dimension = 0; dimension < rank; ++dimension)
					{
						if(!sharedIndices)
						{
							if(oldIndices.stride(1) == 1) std::copy_n(oldIndices[dimension].data() + begin, end - begin, indices[dimension].data() + output);
							else for(int64_t position = begin; position < end; ++position) indices[dimension][output + position - begin] = oldIndices[dimension][position];
						}
					}
					if(oldValues.stride(0) == 1) std::copy_n(oldValues.data() + begin, end - begin, values.data() + output);
					else for(int64_t position = begin; position < end; ++position) values[output + position - begin] = oldValues[position];
				};
				int64_t oldPosition = 0;
				int64_t output = 0;
				for(size_t group = 0; group < insertions.size(); ++group)
				{
					const int64_t insertion = insertions[group];
					copy(oldPosition, insertion, output);
					output += insertion - oldPosition;
					int64_t key = updates[starts[group]].key;
					if(!sharedIndices)
					{
						for(int64_t dimension = rank; dimension-- > 0;)
						{
							indices[dimension][output] = key % shape[dimension];
							key /= shape[dimension];
						}
					}
					float value = overlaps[group] ? oldValues[insertion] : updates[starts[group]].value;
					for(int64_t position = starts[group] + !overlaps[group]; position < starts[group + 1]; ++position) value += updates[position].value;
					values[output++] = value;
					oldPosition = insertion + overlaps[group];
				}
				copy(oldPosition, source._nnz(), output);
				output += source._nnz() - oldPosition;
				TORCH_CHECK(output == count, invalid);
				result[bucket] = at::sparse_coo_tensor(indexTensor, valueTensor, shape, source.options(), true);
				bucket = next.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	return result;
}

at::Tensor createTemporalConnections(bool optimiseTrainSequenceBatch, const at::Tensor& active, const at::Tensor& columns, const at::Tensor& words, int64_t cs, int64_t fs, bool includeSameTime, bool allowSelf, bool nextWordOnly, bool nextColumnOnly, bool useSANI, int64_t lastInputSegment, int64_t mode, int64_t columnsMode, int64_t featuresMode, int64_t combinedMode, int64_t columnSegments, int64_t featureSegments, bool internalColumns, bool linkFirstColumn, bool linkFirstFeature, int64_t lastColumnSegment, int64_t maximumSegments, int64_t grainSize, bool patchTrainColumnStepsRepeatedColumns, const std::optional<at::Tensor>& columnSteps, const std::string& invalid)
{
	at::Tensor result;
	TORCH_CHECK(optimiseTrainSequenceBatch, invalid);
	if(optimiseTrainSequenceBatch)
	{
		TORCH_CHECK(active.device().is_cpu() && active.layout() == at::kStrided && active.scalar_type() == at::kFloat && active.dim() == 4 && !active.requires_grad(), invalid);
		TORCH_CHECK(columns.device().is_cpu() && words.device().is_cpu() && columns.scalar_type() == at::kLong && words.scalar_type() == at::kLong && columns.dim() == 1 && words.dim() == 2, invalid);
		TORCH_CHECK(cs > 0 && fs > 0 && active.size(2) == cs && active.size(3) == fs && columns.size(0) == cs && words.size(0) == cs && words.size(1) == fs, invalid);
		const int64_t branches = active.size(0);
		const int64_t segments = active.size(1);
		TORCH_CHECK(branches > 0 && segments > 0 && segments <= maximumSegments && maximumSegments < std::numeric_limits<uint64_t>::digits && grainSize > 0, invalid);
		TORCH_CHECK(lastInputSegment >= 0 && lastInputSegment < segments && lastColumnSegment >= 0 && lastColumnSegment < segments, invalid);
		TORCH_CHECK(!useSANI || mode == columnsMode || mode == featuresMode || mode == combinedMode, invalid);
		TORCH_CHECK(columnSegments >= 0 && columnSegments <= segments && featureSegments > 0 && featureSegments <= segments, invalid);
		TORCH_CHECK(mode != combinedMode || columnSegments + featureSegments == segments, invalid);
		TORCH_CHECK(cs <= std::numeric_limits<int64_t>::max() / fs, invalid);
		const int64_t neurons = cs * fs;
		TORCH_CHECK(branches <= std::numeric_limits<int64_t>::max() / neurons, invalid);
		const int64_t* columnStepData = nullptr;
		if(patchTrainColumnStepsRepeatedColumns)
		{
			TORCH_CHECK(columnSteps.has_value() && columnSteps->defined(), invalid);
			TORCH_CHECK(columnSteps->device().is_cpu() && columnSteps->layout() == at::kStrided && columnSteps->scalar_type() == at::kLong && columnSteps->dim() == 1 && columnSteps->numel() == cs && columnSteps->is_contiguous(), invalid);
			columnStepData = columnSteps->const_data_ptr<int64_t>();
			TORCH_CHECK(columnStepData[0] == 0, invalid);
			for(int64_t concept = 1; concept < cs; ++concept) TORCH_CHECK(columnStepData[concept] >= columnStepData[concept - 1] && columnStepData[concept] <= columnStepData[concept - 1] + 1, invalid);
		}
		auto activation = active.accessor<float, 4>();
		auto columnOrder = columns.accessor<int64_t, 1>();
		auto wordOrder = words.accessor<int64_t, 2>();
		std::vector<uint64_t> sourceMasks(neurons);
		std::vector<uint64_t> repeatedMasks(neurons);
		std::vector<uint64_t> targetMasks(branches * neurons);
		// A bit records an input segment, so repeated input segments do not generate duplicate COO entries.
		at::parallel_for(0, neurons, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t neuron = begin; neuron < end; ++neuron)
			{
				const int64_t concept = neuron / fs;
				const int64_t feature = neuron % fs;
				TORCH_CHECK(wordOrder[concept][feature] >= 0 && wordOrder[concept][feature] < std::numeric_limits<int64_t>::max() && columnOrder[concept] >= 0 && columnOrder[concept] < std::numeric_limits<int64_t>::max(), invalid);
				for(int64_t segment = 0; segment < segments; ++segment)
				{
					if(useSANI || segment == lastInputSegment)
					{
						const uint64_t bit = uint64_t(1) << segment;
						int64_t activeBranches = 0;
						for(int64_t branch = 0; branch < branches; ++branch)
						{
							const float value = activation[branch][segment][concept][feature];
							TORCH_CHECK(value == 0 || value == 1, invalid);
							if(value > 0)
							{
								targetMasks[branch * neurons + neuron] |= bit;
								++activeBranches;
							}
						}
						if(activeBranches > 0) sourceMasks[neuron] |= bit;
						if(activeBranches > 1) repeatedMasks[neuron] |= bit;
					}
				}
			}
		});
		std::vector<int64_t> sources;
		std::vector<std::vector<int64_t>> targets(branches);
		for(int64_t neuron = 0; neuron < neurons; ++neuron)
		{
			if(sourceMasks[neuron] != 0) sources.push_back(neuron);
			for(int64_t branch = 0; branch < branches; ++branch)
			{
				if(targetMasks[branch * neurons + neuron] != 0) targets[branch].push_back(neuron);
			}
		}
		const int64_t sourceCount = sources.size();
		TORCH_CHECK(sourceCount == 0 || branches <= std::numeric_limits<int64_t>::max() / segments / sourceCount, invalid);
		const int64_t taskCount = branches * segments * sourceCount;
		std::vector<int64_t> counts(taskCount);
		std::vector<int64_t> starts(taskCount);
		auto selectedSegments = [&](int64_t branch, int64_t source, int64_t target)
		{
			const int64_t sourceConcept = source / fs;
			const int64_t targetConcept = target / fs;
			const int64_t sourceWord = wordOrder[sourceConcept][source % fs];
			const int64_t targetWord = wordOrder[targetConcept][target % fs];
			const uint64_t commonSegments = sourceMasks[source] & targetMasks[branch * neurons + target];
			bool permitted = commonSegments != 0;
			if(source == target)
			{
				permitted = permitted && (allowSelf || (commonSegments & repeatedMasks[source]) != 0);
			}
			else
			{
				permitted = permitted && (includeSameTime ? targetWord >= sourceWord : targetWord > sourceWord);
				permitted = permitted && (!nextWordOnly || targetWord <= sourceWord + 1);
				permitted = permitted && columnOrder[targetConcept] >= columnOrder[sourceConcept];
				permitted = permitted && (!nextColumnOnly || columnOrder[targetConcept] <= columnOrder[sourceConcept] + 1);
			}
			uint64_t selected = 0;
			if(permitted)
			{
				const int64_t distance = std::max<int64_t>(1, targetWord - sourceWord);
				int64_t conceptDistance = std::abs(targetConcept - sourceConcept);
				if(patchTrainColumnStepsRepeatedColumns)
				{
					conceptDistance = std::abs(columnStepData[targetConcept] - columnStepData[sourceConcept]);
				}
				if(!useSANI)
				{
					selected = uint64_t(1);
				}
				else if(mode == columnsMode)
				{
					const int64_t assigned = std::max<int64_t>(0, lastColumnSegment - conceptDistance);
					if(linkFirstColumn || conceptDistance <= lastColumnSegment) selected |= uint64_t(1) << assigned;
				}
				else if(mode == featuresMode)
				{
					const int64_t assigned = std::max<int64_t>(0, segments - distance);
					if(linkFirstFeature || distance <= segments) selected |= uint64_t(1) << assigned;
				}
				else if(mode == combinedMode)
				{
					const int64_t assignedFeature = columnSegments + featureSegments - std::min(distance, featureSegments);
					if(linkFirstFeature || distance <= featureSegments) selected |= uint64_t(1) << assignedFeature;
					if(columnSegments > 0)
					{
						const int64_t assignedColumn = std::clamp<int64_t>(columnSegments - conceptDistance - int64_t(internalColumns), 0, columnSegments - 1);
						const bool validColumn = internalColumns ? (linkFirstColumn || conceptDistance < columnSegments) : (conceptDistance > 0 && (linkFirstColumn || conceptDistance <= columnSegments));
						if(validColumn) selected |= uint64_t(1) << assignedColumn;
					}
				}
			}
			return selected;
		};
		std::vector<std::vector<int64_t>> selectedTargets(taskCount);
		// Compute each pair once; task-local targets remain sorted in canonical COO order.
		at::parallel_for(0, branches * sourceCount, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t pairTask = begin; pairTask < end; ++pairTask)
			{
				const int64_t branch = pairTask / sourceCount;
				const int64_t sourceIndex = pairTask % sourceCount;
				const int64_t source = sources[sourceIndex];
				for(const auto target : targets[branch])
				{
					uint64_t selected = selectedSegments(branch, source, target);
					while(selected != 0)
					{
						const int64_t segment = __builtin_ctzll(selected);
						const int64_t task = (branch * segments + segment) * sourceCount + sourceIndex;
						selectedTargets[task].push_back(target);
						++counts[task];
						selected &= selected - 1;
					}
				}
			}
		});
		int64_t total = 0;
		for(int64_t task = 0; task < taskCount; ++task)
		{
			starts[task] = total;
			TORCH_CHECK(counts[task] <= std::numeric_limits<int64_t>::max() - total, invalid);
			total += counts[task];
		}
		const std::vector<int64_t> shape = {branches, segments, cs, fs, cs, fs};
		auto indexTensor = at::empty({int64_t(shape.size()), total}, columns.options());
		auto valueTensor = at::ones({total}, active.options());
		auto indices = indexTensor.accessor<int64_t, 2>();
		at::parallel_for(0, taskCount, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t task = begin; task < end; ++task)
			{
				const int64_t branch = task / sourceCount / segments;
				const int64_t segment = task / sourceCount % segments;
				const int64_t source = sources[task % sourceCount];
				int64_t position = starts[task];
				for(const auto target : selectedTargets[task])
				{
					if(optimiseTrainSequenceBatch)
					{
						indices[0][position] = branch;
						indices[1][position] = segment;
						indices[2][position] = source / fs;
						indices[3][position] = source % fs;
						indices[4][position] = target / fs;
						indices[5][position] = target % fs;
						++position;
					}
				}
				TORCH_CHECK(position == starts[task] + counts[task], invalid);
			}
		});
		result = at::sparse_coo_tensor(indexTensor, valueTensor, shape, active.options().layout(at::kSparse), true);
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	module.def("mergeSequenceBatch", &mergeSequenceBatch, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("createTemporalConnections", &createTemporalConnections, pybind11::call_guard<pybind11::gil_scoped_release>());
}
