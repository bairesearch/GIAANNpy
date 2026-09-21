// Independent CPU training kernels; Python supplies feature switches and tuning constants.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <cstdint>
#include <limits>
#include <vector>

#ifndef INTRA_OP_PARALLEL
#error "CPU training kernels require the PyTorch intra-op parallel backend"
#endif

at::Tensor createTemporalConnections(bool optimiseParallelisation2a, const at::Tensor& active, const at::Tensor& columns, const at::Tensor& words, int64_t cs, int64_t fs, bool includeSameTime, bool allowSelf, bool nextWordOnly, bool nextColumnOnly, bool useSANI, int64_t lastInputSegment, int64_t mode, int64_t columnsMode, int64_t featuresMode, int64_t combinedMode, int64_t columnSegments, int64_t featureSegments, bool internalColumns, bool linkFirstColumn, bool linkFirstFeature, int64_t lastColumnSegment, int64_t maximumSegments, int64_t grainSize)
{
	at::Tensor result;
	TORCH_CHECK(optimiseParallelisation2a, "Temporal construction requires optimiseParallelisation2a");
	if(optimiseParallelisation2a)
	{
		TORCH_CHECK(active.device().is_cpu() && active.layout() == at::kStrided && active.scalar_type() == at::kFloat && active.dim() == 4 && !active.requires_grad(), "Expected float32 CPU activation tensor of rank four without autograd");
		TORCH_CHECK(columns.device().is_cpu() && words.device().is_cpu() && columns.scalar_type() == at::kLong && words.scalar_type() == at::kLong && columns.dim() == 1 && words.dim() == 2, "Expected CPU int64 column and word order tensors");
		TORCH_CHECK(cs > 0 && fs > 0 && active.size(2) == cs && active.size(3) == fs && columns.size(0) == cs && words.size(0) == cs && words.size(1) == fs, "Temporal construction dimensions do not match");
		const int64_t branches = active.size(0);
		const int64_t segments = active.size(1);
		TORCH_CHECK(branches > 0 && segments > 0 && segments <= maximumSegments && maximumSegments < std::numeric_limits<uint64_t>::digits && grainSize > 0, "Invalid activation-mask dimensions or grain size");
		TORCH_CHECK(lastInputSegment >= 0 && lastInputSegment < segments && lastColumnSegment >= 0 && lastColumnSegment < segments, "Invalid input or column segment");
		TORCH_CHECK(!useSANI || mode == columnsMode || mode == featuresMode || mode == combinedMode, "Unknown temporal segment mode");
		TORCH_CHECK(columnSegments >= 0 && columnSegments <= segments && featureSegments > 0 && featureSegments <= segments, "Invalid temporal segment allocation");
		TORCH_CHECK(mode != combinedMode || columnSegments + featureSegments == segments, "Combined temporal segments do not cover the output");
		TORCH_CHECK(cs <= std::numeric_limits<int64_t>::max() / fs, "Temporal neuron count overflows int64");
		const int64_t neurons = cs * fs;
		TORCH_CHECK(branches <= std::numeric_limits<int64_t>::max() / neurons, "Temporal branch allocation overflows int64");
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
				TORCH_CHECK(wordOrder[concept][feature] >= 0 && wordOrder[concept][feature] < std::numeric_limits<int64_t>::max() && columnOrder[concept] >= 0 && columnOrder[concept] < std::numeric_limits<int64_t>::max(), "Temporal order index is out of range");
				for(int64_t segment = 0; segment < segments; ++segment)
				{
					if(useSANI || segment == lastInputSegment)
					{
						const uint64_t bit = uint64_t(1) << segment;
						int64_t activeBranches = 0;
						for(int64_t branch = 0; branch < branches; ++branch)
						{
							const float value = activation[branch][segment][concept][feature];
							TORCH_CHECK(value == 0 || value == 1, "Temporal construction requires binary neuron activations");
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
		TORCH_CHECK(sourceCount == 0 || branches <= std::numeric_limits<int64_t>::max() / segments / sourceCount, "Temporal task count overflows int64");
		const int64_t taskCount = branches * segments * sourceCount;
		std::vector<int64_t> counts(taskCount);
		std::vector<int64_t> starts(taskCount);
		auto eligible = [&](int64_t branch, int64_t segment, int64_t source, int64_t target)
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
			bool selected = false;
			if(permitted)
			{
				const int64_t distance = std::max<int64_t>(1, targetWord - sourceWord);
				const int64_t conceptDistance = std::abs(targetConcept - sourceConcept);
				if(!useSANI)
				{
					selected = segment == 0;
				}
				else if(mode == columnsMode)
				{
					const int64_t assigned = std::max<int64_t>(0, lastColumnSegment - conceptDistance);
					selected = (linkFirstColumn || conceptDistance <= lastColumnSegment) && segment == assigned;
				}
				else if(mode == featuresMode)
				{
					const int64_t assigned = std::max<int64_t>(0, segments - distance);
					selected = (linkFirstFeature || distance <= segments) && segment == assigned;
				}
				else if(mode == combinedMode)
				{
					const int64_t assignedFeature = columnSegments + featureSegments - std::min(distance, featureSegments);
					selected = (linkFirstFeature || distance <= featureSegments) && segment == assignedFeature;
					if(columnSegments > 0)
					{
						const int64_t assignedColumn = std::clamp<int64_t>(columnSegments - conceptDistance - int64_t(internalColumns), 0, columnSegments - 1);
						const bool validColumn = internalColumns ? (linkFirstColumn || conceptDistance < columnSegments) : (conceptDistance > 0 && (linkFirstColumn || conceptDistance <= columnSegments));
						selected = selected || (validColumn && segment == assignedColumn);
					}
				}
			}
			return selected;
		};
		// Tasks are ordered exactly like canonical COO coordinates: branch, segment, source, target.
		at::parallel_for(0, taskCount, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t task = begin; task < end; ++task)
			{
				const int64_t branch = task / sourceCount / segments;
				const int64_t segment = task / sourceCount % segments;
				const int64_t source = sources[task % sourceCount];
				for(const auto target : targets[branch]) counts[task] += eligible(branch, segment, source, target);
			}
		});
		int64_t total = 0;
		for(int64_t task = 0; task < taskCount; ++task)
		{
			starts[task] = total;
			TORCH_CHECK(counts[task] <= std::numeric_limits<int64_t>::max() - total, "Temporal output count overflows int64");
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
				for(const auto target : targets[branch])
				{
					if(eligible(branch, segment, source, target))
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
				TORCH_CHECK(position == starts[task] + counts[task], "Temporal output count mismatch");
			}
		});
		result = at::sparse_coo_tensor(indexTensor, valueTensor, shape, active.options().layout(at::kSparse), true);
	}
	return result;
}

std::tuple<at::Tensor, at::Tensor> prepareConnectionUpdates(bool optimiseParallelisation2b, const at::Tensor& indices, const at::Tensor& values, const at::Tensor& featureMap, const at::Tensor& conceptMap, const std::vector<int64_t>& sourceSize, int64_t propertyIndex, bool mapFeatures, int64_t grainSize)
{
	std::tuple<at::Tensor, at::Tensor> result;
	TORCH_CHECK(optimiseParallelisation2b, "Delta mapping requires optimiseParallelisation2b");
	if(optimiseParallelisation2b)
	{
		TORCH_CHECK(sourceSize.size() == 5 && indices.device().is_cpu() && indices.scalar_type() == at::kLong && indices.dim() == 2 && indices.size(0) == int64_t(sourceSize.size()) + 1, "Expected rank-six CPU connection coordinates");
		TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kFloat && values.dim() == 1 && values.size(0) == indices.size(1) && !values.requires_grad(), "Expected float32 CPU connection values without autograd");
		TORCH_CHECK(featureMap.device().is_cpu() && conceptMap.device().is_cpu() && featureMap.scalar_type() == at::kLong && conceptMap.scalar_type() == at::kLong && featureMap.dim() == 1 && conceptMap.dim() == 1, "Expected CPU int64 feature and concept maps");
		for(const auto size : sourceSize) TORCH_CHECK(size > 0, "Connection dimensions must be positive");
		TORCH_CHECK(propertyIndex >= 0 && propertyIndex < sourceSize[0] && grainSize > 0, "Invalid strength property or mapping grain size");
		TORCH_CHECK(sourceSize[3] <= std::numeric_limits<int64_t>::max() / sourceSize[4], "Combined source key overflows int64");
		const int64_t count = values.numel();
		const int64_t conceptCount = conceptMap.numel();
		const int64_t featureCount = featureMap.numel();
		auto keys = at::empty({count}, indices.options());
		auto mapped = at::empty_like(indices);
		auto input = indices.accessor<int64_t, 2>();
		auto output = mapped.accessor<int64_t, 2>();
		auto sourceKeys = keys.accessor<int64_t, 1>();
		auto features = featureMap.accessor<int64_t, 1>();
		auto concepts = conceptMap.accessor<int64_t, 1>();
		at::parallel_for(0, count, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t position = begin; position < end; ++position)
			{
				TORCH_CHECK(input[0][position] >= 0 && input[0][position] < sourceSize[1] && input[1][position] >= 0 && input[1][position] < sourceSize[2], "Connection branch or segment is out of range");
				TORCH_CHECK(input[2][position] >= 0 && input[2][position] < conceptCount && input[4][position] >= 0 && input[4][position] < conceptCount, "Connection concept lookup is out of range");
				const int64_t sourceConcept = concepts[input[2][position]];
				const int64_t targetConcept = concepts[input[4][position]];
				int64_t sourceFeature = input[3][position];
				int64_t targetFeature = input[5][position];
				TORCH_CHECK(sourceFeature >= 0 && targetFeature >= 0, "Negative connection feature index");
				if(mapFeatures)
				{
					TORCH_CHECK(sourceFeature < featureCount && targetFeature < featureCount, "Connection feature lookup is out of range");
					sourceFeature = features[sourceFeature];
					targetFeature = features[targetFeature];
				}
				TORCH_CHECK(sourceConcept >= 0 && sourceConcept < sourceSize[3] && targetConcept >= 0 && targetConcept < sourceSize[3] && sourceFeature >= 0 && sourceFeature < sourceSize[4] && targetFeature >= 0 && targetFeature < sourceSize[4], "Mapped connection coordinate is out of range");
				sourceKeys[position] = sourceConcept * sourceSize[4] + sourceFeature;
				output[0][position] = propertyIndex;
				output[1][position] = input[0][position];
				output[2][position] = input[1][position];
				output[4][position] = targetConcept;
				output[5][position] = targetFeature;
			}
		});
		const auto uniqueResult = at::_unique2(keys, true, true, false);
		const auto uniqueKeys = std::get<0>(uniqueResult);
		const auto inverseTensor = std::get<1>(uniqueResult);
		auto inverse = inverseTensor.accessor<int64_t, 1>();
		at::parallel_for(0, count, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t position = begin; position < end; ++position) output[3][position] = inverse[position];
		});
		const std::vector<int64_t> updateSize = {sourceSize[0], sourceSize[1], sourceSize[2], uniqueKeys.numel(), sourceSize[3], sourceSize[4]};
		// Keep PyTorch's original coalescing order for exact duplicate floating-point reductions.
		result = std::make_tuple(uniqueKeys, at::sparse_coo_tensor(mapped, values, updateSize, values.options().layout(at::kSparse)).coalesce());
	}
	return result;
}

void parallelMergeTasks(bool optimiseParallelisation2c, int64_t taskCount, int64_t grainSize, const std::function<void(int64_t, int64_t)>& operation);

std::vector<at::Tensor> mergeSparseSources(bool optimiseParallelisation2c, bool optimiseParallelisation2d, const std::vector<at::Tensor>& sources, const at::Tensor& updates, const std::vector<int64_t>& sourceSize, int64_t bucketDimension, int64_t chunkEntries, int64_t grainSize)
{
	std::vector<at::Tensor> result;
	TORCH_CHECK(optimiseParallelisation2c || optimiseParallelisation2d, "Partitioned merging requires optimiseParallelisation2c or optimiseParallelisation2d");
	if(optimiseParallelisation2c || optimiseParallelisation2d)
	{
		const int64_t rank = sourceSize.size();
		const int64_t bucketCount = sources.size();
		TORCH_CHECK(rank > 0 && bucketDimension >= 0 && bucketDimension <= rank && chunkEntries > 0 && grainSize > 0, "Invalid merge layout or task size");
		for(const auto size : sourceSize) TORCH_CHECK(size > 0, "Connection dimensions must be positive");
		auto validate = [](const at::Tensor& tensor, int64_t expectedRank)
		{
			TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat && tensor.sparse_dim() == expectedRank && tensor.dense_dim() == 0 && tensor.is_coalesced() && !tensor.requires_grad(), "Expected coalesced scalar float32 CPU sparse tensor without autograd");
		};
		validate(updates, rank + 1);
		TORCH_CHECK(updates.size(bucketDimension) == bucketCount, "Update bucket count mismatch");
		for(int64_t dimension = 0; dimension < rank; ++dimension) TORCH_CHECK(updates.size(dimension + (dimension >= bucketDimension)) == sourceSize[dimension], "Update dimensions do not match source shape");
		const auto updateIndexTensor = updates.indices();
		const auto updateValueTensor = updates.values();
		auto updateIndices = updateIndexTensor.accessor<int64_t, 2>();
		auto updateValues = updateValueTensor.accessor<float, 1>();
		std::vector<std::vector<int64_t>> positions(bucketCount);
		for(int64_t position = 0; position < updates._nnz(); ++position)
		{
			for(int64_t dimension = 0; dimension <= rank; ++dimension) TORCH_CHECK(updateIndices[dimension][position] >= 0 && updateIndices[dimension][position] < updates.size(dimension), "Update coordinate is out of range");
			positions[updateIndices[bucketDimension][position]].push_back(position);
		}
		std::vector<at::Tensor> sourceIndexTensors;
		std::vector<at::Tensor> sourceValueTensors;
		std::vector<at::TensorAccessor<int64_t, 2>> sourceIndices;
		std::vector<at::TensorAccessor<float, 1>> sourceValues;
		std::vector<int64_t> sourceCounts;
		for(const auto& source : sources)
		{
			validate(source, rank);
			for(int64_t dimension = 0; dimension < rank; ++dimension) TORCH_CHECK(source.size(dimension) > 0 && source.size(dimension) <= sourceSize[dimension], "Existing tensor exceeds target shape");
			sourceIndexTensors.push_back(source.indices());
			sourceValueTensors.push_back(source.values());
			sourceIndices.push_back(sourceIndexTensors.back().accessor<int64_t, 2>());
			sourceValues.push_back(sourceValueTensors.back().accessor<float, 1>());
			sourceCounts.push_back(source._nnz());
		}
		auto compare = [&](int64_t bucket, int64_t oldPosition, int64_t newPosition)
		{
			int comparison = 0;
			for(int64_t dimension = 0; dimension < rank && comparison == 0; ++dimension)
			{
				const auto oldCoordinate = sourceIndices[bucket][dimension][oldPosition];
				const auto newCoordinate = updateIndices[dimension + (dimension >= bucketDimension)][positions[bucket][newPosition]];
				comparison = (oldCoordinate > newCoordinate) - (oldCoordinate < newCoordinate);
			}
			return comparison;
		};
		struct Task { int64_t bucket, oldBegin, oldEnd, newBegin, newEnd, count, outputBegin; std::vector<int64_t> insertions; };
		std::vector<Task> tasks;
		std::vector<int64_t> firstTasks(bucketCount + 1);
		for(int64_t bucket = 0; bucket < bucketCount; ++bucket)
		{
			firstTasks[bucket] = tasks.size();
			const int64_t oldCount = sourceCounts[bucket];
			const int64_t newCount = positions[bucket].size();
			TORCH_CHECK(newCount <= std::numeric_limits<int64_t>::max() - oldCount, "Merge input count overflows int64");
			const int64_t totalCount = oldCount + newCount;
			if(optimiseParallelisation2c && totalCount > 0)
			{
				int64_t oldBegin = 0;
				int64_t newBegin = 0;
				// Partition the merged input order, including insertion-heavy and empty-source buckets.
				for(int64_t diagonal = 0; diagonal < totalCount; diagonal += std::min(chunkEntries, totalCount - diagonal))
				{
					const int64_t endDiagonal = diagonal + std::min(chunkEntries, totalCount - diagonal);
					int64_t low = std::max<int64_t>(0, endDiagonal - newCount);
					int64_t high = std::min(endDiagonal, oldCount);
					while(low < high)
					{
						const int64_t middle = low + (high - low) / 2;
						const int64_t newMiddle = endDiagonal - middle;
						if(newMiddle > 0 && middle < oldCount && compare(bucket, middle, newMiddle - 1) <= 0) low = middle + 1;
						else high = middle;
					}
					const int64_t oldEnd = low;
					int64_t newEnd = endDiagonal - oldEnd;
					// Keep an equal old/new pair together so no task emits a duplicate boundary key.
					if(oldEnd > 0 && newEnd < newCount && compare(bucket, oldEnd - 1, newEnd) == 0) ++newEnd;
					if(oldEnd != oldBegin || newEnd != newBegin) tasks.push_back({bucket, oldBegin, oldEnd, newBegin, newEnd, 0, 0, {}});
					oldBegin = oldEnd;
					newBegin = newEnd;
				}
			}
			else
			{
				tasks.push_back({bucket, 0, oldCount, 0, newCount, 0, 0, {}});
			}
		}
		firstTasks[bucketCount] = tasks.size();
		auto countTasks = [&](int64_t begin, int64_t end)
		{
			for(int64_t taskIndex = begin; taskIndex < end; ++taskIndex)
			{
				auto& task = tasks[taskIndex];
				task.count = task.oldEnd - task.oldBegin + task.newEnd - task.newBegin;
				if(optimiseParallelisation2d)
				{
					task.insertions.resize(task.newEnd - task.newBegin);
					int64_t previous = task.oldBegin;
					for(int64_t next = task.newBegin; next < task.newEnd; ++next)
					{
						int64_t low = previous;
						int64_t high = task.oldEnd;
						while(low < high)
						{
							const int64_t middle = low + (high - low) / 2;
							if(compare(task.bucket, middle, next) < 0) low = middle + 1;
							else high = middle;
						}
						task.insertions[next - task.newBegin] = low;
						previous = low;
						task.count -= low < task.oldEnd && compare(task.bucket, low, next) == 0;
					}
				}
				else
				{
					int64_t oldPosition = task.oldBegin;
					int64_t newPosition = task.newBegin;
					while(oldPosition < task.oldEnd && newPosition < task.newEnd)
					{
						const int comparison = compare(task.bucket, oldPosition, newPosition);
						task.count -= comparison == 0;
						oldPosition += comparison <= 0;
						newPosition += comparison >= 0;
					}
				}
			}
		};
		if(optimiseParallelisation2c)
		{
			parallelMergeTasks(optimiseParallelisation2c, tasks.size(), grainSize, countTasks);
		}
		else
		{
			at::parallel_for(0, tasks.size(), grainSize, countTasks);
		}
		std::vector<at::Tensor> outputIndexTensors;
		std::vector<at::Tensor> outputValueTensors;
		std::vector<at::TensorAccessor<int64_t, 2>> outputIndices;
		std::vector<at::TensorAccessor<float, 1>> outputValues;
		std::vector<bool> sharedIndices(bucketCount);
		for(int64_t bucket = 0; bucket < bucketCount; ++bucket)
		{
			int64_t count = 0;
			for(int64_t task = firstTasks[bucket]; task < firstTasks[bucket + 1]; ++task)
			{
				tasks[task].outputBegin = count;
				TORCH_CHECK(tasks[task].count <= std::numeric_limits<int64_t>::max() - count, "Merge output count overflows int64");
				count += tasks[task].count;
			}
			sharedIndices[bucket] = optimiseParallelisation2d && count == sourceCounts[bucket];
			outputIndexTensors.push_back(sharedIndices[bucket] ? sourceIndexTensors[bucket] : at::empty({rank, count}, updateIndexTensor.options()));
			outputValueTensors.push_back(at::empty({count}, updateValueTensor.options()));
			outputIndices.push_back(outputIndexTensors.back().accessor<int64_t, 2>());
			outputValues.push_back(outputValueTensors.back().accessor<float, 1>());
		}
		auto copyOld = [&](int64_t bucket, int64_t begin, int64_t end, int64_t destination)
		{
			const int64_t count = end - begin;
			if(count > 0)
			{
				for(int64_t dimension = 0; dimension < rank; ++dimension)
				{
					int64_t minimum = std::numeric_limits<int64_t>::max();
					int64_t maximum = std::numeric_limits<int64_t>::min();
					for(int64_t position = begin; position < end; ++position)
					{
						minimum = std::min(minimum, sourceIndices[bucket][dimension][position]);
						maximum = std::max(maximum, sourceIndices[bucket][dimension][position]);
					}
					TORCH_CHECK(minimum >= 0 && maximum < sourceSize[dimension], "Existing merge coordinate is out of range");
					if(!sharedIndices[bucket])
					{
						if(sourceIndices[bucket].stride(1) == 1) std::copy_n(sourceIndices[bucket][dimension].data() + begin, count, outputIndices[bucket][dimension].data() + destination);
						else for(int64_t offset = 0; offset < count; ++offset) outputIndices[bucket][dimension][destination + offset] = sourceIndices[bucket][dimension][begin + offset];
					}
				}
				if(sourceValues[bucket].stride(0) == 1) std::copy_n(sourceValues[bucket].data() + begin, count, outputValues[bucket].data() + destination);
				else for(int64_t offset = 0; offset < count; ++offset) outputValues[bucket][destination + offset] = sourceValues[bucket][begin + offset];
			}
		};
		auto fillTasks = [&](int64_t begin, int64_t end)
		{
			for(int64_t taskIndex = begin; taskIndex < end; ++taskIndex)
			{
				const auto& task = tasks[taskIndex];
				const int64_t bucket = task.bucket;
				int64_t oldPosition = task.oldBegin;
				int64_t newPosition = task.newBegin;
				int64_t outputPosition = task.outputBegin;
				if(optimiseParallelisation2d)
				{
					for(; newPosition < task.newEnd; ++newPosition)
					{
						const int64_t insertion = task.insertions[newPosition - task.newBegin];
						copyOld(bucket, oldPosition, insertion, outputPosition);
						outputPosition += insertion - oldPosition;
						const bool overlap = insertion < task.oldEnd && compare(bucket, insertion, newPosition) == 0;
						for(int64_t dimension = 0; dimension < rank; ++dimension)
						{
							if(!sharedIndices[bucket]) outputIndices[bucket][dimension][outputPosition] = updateIndices[dimension + (dimension >= bucketDimension)][positions[bucket][newPosition]];
						}
						outputValues[bucket][outputPosition] = overlap ? sourceValues[bucket][insertion] + updateValues[positions[bucket][newPosition]] : updateValues[positions[bucket][newPosition]];
						oldPosition = insertion + int64_t(overlap);
						++outputPosition;
					}
					copyOld(bucket, oldPosition, task.oldEnd, outputPosition);
					outputPosition += task.oldEnd - oldPosition;
				}
				else
				{
					while(oldPosition < task.oldEnd || newPosition < task.newEnd)
					{
						const int comparison = oldPosition == task.oldEnd ? 1 : (newPosition == task.newEnd ? -1 : compare(bucket, oldPosition, newPosition));
						for(int64_t dimension = 0; dimension < rank; ++dimension)
						{
							const int64_t coordinate = comparison <= 0 ? sourceIndices[bucket][dimension][oldPosition] : updateIndices[dimension + (dimension >= bucketDimension)][positions[bucket][newPosition]];
							TORCH_CHECK(coordinate >= 0 && coordinate < sourceSize[dimension], "Existing merge coordinate is out of range");
							outputIndices[bucket][dimension][outputPosition] = coordinate;
						}
						outputValues[bucket][outputPosition] = comparison < 0 ? sourceValues[bucket][oldPosition] : (comparison > 0 ? updateValues[positions[bucket][newPosition]] : sourceValues[bucket][oldPosition] + updateValues[positions[bucket][newPosition]]);
						oldPosition += comparison <= 0;
						newPosition += comparison >= 0;
						++outputPosition;
					}
				}
				TORCH_CHECK(outputPosition == task.outputBegin + task.count, "Merge task output count mismatch");
			}
		};
		if(optimiseParallelisation2c)
		{
			parallelMergeTasks(optimiseParallelisation2c, tasks.size(), grainSize, fillTasks);
		}
		else
		{
			at::parallel_for(0, tasks.size(), grainSize, fillTasks);
		}
		for(int64_t bucket = 0; bucket < bucketCount; ++bucket) result.push_back(at::sparse_coo_tensor(outputIndexTensors[bucket], outputValueTensors[bucket], sourceSize, sources[bucket].options(), true));
	}
	return result;
}

void parallelMergeTasks(bool optimiseParallelisation2c, int64_t taskCount, int64_t grainSize, const std::function<void(int64_t, int64_t)>& operation)
{
	TORCH_CHECK(optimiseParallelisation2c, "Dynamic merge scheduling requires optimiseParallelisation2c");
	if(optimiseParallelisation2c)
	{
		TORCH_CHECK(taskCount >= 0 && grainSize > 0 && taskCount <= std::numeric_limits<int64_t>::max() - at::get_num_threads(), "Invalid dynamic merge task count or grain size");
		std::atomic<int64_t> nextTask{0};
		// Workers take another bounded task when ready; tiny buckets cannot strand large-source chunks on one worker.
		at::parallel_for(0, taskCount, grainSize, [&](int64_t, int64_t)
		{
			int64_t task = nextTask.fetch_add(1, std::memory_order_relaxed);
			while(task < taskCount)
			{
				operation(task, task + 1);
				task = nextTask.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	module.def("createTemporalConnections", &createTemporalConnections, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("prepareConnectionUpdates", &prepareConnectionUpdates, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("mergeSparseSources", &mergeSparseSources, pybind11::call_guard<pybind11::gil_scoped_release>());
}
