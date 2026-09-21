// CPU training connection merges. The Python wrapper supplies layout constants from globalDefs.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <vector>

#ifndef INTRA_OP_PARALLEL
#error "CPU connection merges require the PyTorch intra-op parallel backend"
#endif

std::vector<at::Tensor> mergeConnectionSources(bool optimiseParallelisation1, const std::vector<at::Tensor>& sources, const at::Tensor& updates, const std::vector<int64_t>& sourceSize, int64_t bucketDimension, int64_t grainSize)
{
	std::vector<at::Tensor> result;
	TORCH_CHECK(optimiseParallelisation1, "CPU connection merges require optimiseParallelisation1");
	if(optimiseParallelisation1)
	{
		const int64_t rank = sourceSize.size();
		const int64_t bucketCount = sources.size();
		TORCH_CHECK(rank > 0 && bucketDimension >= 0 && bucketDimension <= rank && grainSize > 0, "Invalid connection layout or merge grain size");
		for(const auto size : sourceSize)
		{
			TORCH_CHECK(size > 0, "Connection dimensions must be positive");
		}
		// Check metadata before accessing COO indices; reject unsupported inputs without a fallback.
		auto validateTensor = [](const at::Tensor& tensor, int64_t expectedRank)
		{
			TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat, "Expected float32 CPU sparse COO connections");
			TORCH_CHECK(tensor.sparse_dim() == expectedRank && tensor.dense_dim() == 0 && tensor.is_coalesced() && !tensor.requires_grad(), "Expected coalesced scalar connections without autograd");
		};
		validateTensor(updates, rank + 1);
		TORCH_CHECK(updates.size(bucketDimension) == bucketCount, "Connection source bucket count mismatch");
		for(int64_t dimension = 0; dimension < rank; ++dimension)
		{
			TORCH_CHECK(updates.size(dimension + (dimension >= bucketDimension)) == sourceSize[dimension], "Connection update shape mismatch");
		}
		const auto updateIndexTensor = updates.indices();
		const auto updateValueTensor = updates.values();
		auto updateIndices = updateIndexTensor.accessor<int64_t, 2>();
		auto updateValues = updateValueTensor.accessor<float, 1>();
		std::vector<std::vector<int64_t>> updatePositions(bucketCount);
		// Stable grouping preserves lexicographic COO order within every source bucket.
		for(int64_t position = 0; position < updates._nnz(); ++position)
		{
			for(int64_t dimension = 0; dimension <= rank; ++dimension)
			{
				TORCH_CHECK(updateIndices[dimension][position] >= 0 && updateIndices[dimension][position] < updates.size(dimension), "Connection update coordinate out of range");
			}
			updatePositions[updateIndices[bucketDimension][position]].push_back(position);
		}
		for(const auto& source : sources)
		{
			validateTensor(source, rank);
			for(int64_t dimension = 0; dimension < rank; ++dimension)
			{
				TORCH_CHECK(source.size(dimension) > 0 && source.size(dimension) <= sourceSize[dimension], "Existing connection shape exceeds target shape");
			}
		}
		// Accessors borrow sizes/strides as well as storage; retain the indices()/values() Tensor views.
		std::vector<at::Tensor> sourceIndexTensors;
		std::vector<at::Tensor> sourceValueTensors;
		std::vector<at::TensorAccessor<int64_t, 2>> sourceIndices;
		std::vector<at::TensorAccessor<float, 1>> sourceValues;
		std::vector<int64_t> sourceCounts;
		std::vector<int64_t> outputCounts(bucketCount);
		for(const auto& source : sources)
		{
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
				const auto newCoordinate = updateIndices[dimension + (dimension >= bucketDimension)][updatePositions[bucket][newPosition]];
				comparison = (oldCoordinate > newCoordinate) - (oldCoordinate < newCoordinate);
			}
			return comparison;
		};
		// Count overlaps before allocation. Worker loops use raw accessors only, never Tensor operations.
		at::parallel_for(0, bucketCount, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t bucket = begin; bucket < end; ++bucket)
			{
				int64_t oldPosition = 0;
				int64_t newPosition = 0;
				const int64_t oldCount = sourceCounts[bucket];
				const int64_t newCount = updatePositions[bucket].size();
				int64_t outputCount = oldCount + newCount;
				while(oldPosition < oldCount && newPosition < newCount)
				{
					const int comparison = compare(bucket, oldPosition, newPosition);
					outputCount -= (comparison == 0);
					oldPosition += (comparison <= 0);
					newPosition += (comparison >= 0);
				}
				outputCounts[bucket] = outputCount;
			}
		});
		std::vector<at::Tensor> outputIndexTensors;
		std::vector<at::Tensor> outputValueTensors;
		std::vector<at::TensorAccessor<int64_t, 2>> outputIndices;
		std::vector<at::TensorAccessor<float, 1>> outputValues;
		for(const auto outputCount : outputCounts)
		{
			outputIndexTensors.push_back(at::empty({rank, outputCount}, at::TensorOptions().dtype(at::kLong).device(at::kCPU)));
			outputValueTensors.push_back(at::empty({outputCount}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU)));
			outputIndices.push_back(outputIndexTensors.back().accessor<int64_t, 2>());
			outputValues.push_back(outputValueTensors.back().accessor<float, 1>());
		}
		// Only independent source buckets run concurrently. Sequence update order is unchanged.
		at::parallel_for(0, bucketCount, grainSize, [&](int64_t begin, int64_t end)
		{
			for(int64_t bucket = begin; bucket < end; ++bucket)
			{
				const auto& positions = updatePositions[bucket];
				const int64_t oldCount = sourceCounts[bucket];
				const int64_t newCount = positions.size();
				int64_t oldPosition = 0;
				int64_t newPosition = 0;
				int64_t outputPosition = 0;
				while(oldPosition < oldCount || newPosition < newCount)
				{
					const int comparison = oldPosition == oldCount ? 1 : (newPosition == newCount ? -1 : compare(bucket, oldPosition, newPosition));
					for(int64_t dimension = 0; dimension < rank; ++dimension)
					{
						const int64_t coordinate = comparison <= 0 ? sourceIndices[bucket][dimension][oldPosition] : updateIndices[dimension + (dimension >= bucketDimension)][positions[newPosition]];
						TORCH_CHECK(coordinate >= 0 && coordinate < sourceSize[dimension], "Existing connection coordinate out of range");
						outputIndices[bucket][dimension][outputPosition] = coordinate;
					}
					// Retain explicit zeros, negative values, and untouched properties, like COO coalesce.
					outputValues[bucket][outputPosition] = comparison < 0 ? sourceValues[bucket][oldPosition] : (comparison > 0 ? updateValues[positions[newPosition]] : sourceValues[bucket][oldPosition] + updateValues[positions[newPosition]]);
					oldPosition += (comparison <= 0);
					newPosition += (comparison >= 0);
					++outputPosition;
				}
				TORCH_CHECK(outputPosition == outputCounts[bucket], "Connection merge output size mismatch");
			}
		});
		for(int64_t bucket = 0; bucket < bucketCount; ++bucket)
		{
			result.push_back(at::sparse_coo_tensor(outputIndexTensors[bucket], outputValueTensors[bucket], sourceSize, sources[bucket].options(), true));
		}
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	module.def("mergeConnectionSources", &mergeConnectionSources, pybind11::call_guard<pybind11::gil_scoped_release>());
}
