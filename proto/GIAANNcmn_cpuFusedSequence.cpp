#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> preparePackets(bool enabled, const at::Tensor& concepts, const at::Tensor& starts, const at::Tensor& ends, const at::Tensor& conceptMap, const at::Tensor& featureMap, const at::Tensor& steps, const at::Tensor& proximity, const std::vector<int64_t>& shape, int64_t columnSegments, int64_t activationFeatureSegments, int64_t connectionFeatureSegments, bool allowSelf, bool nextWordOnly, bool nextColumnOnly, int64_t zero, int64_t one, int64_t sourceRank, int64_t connectionRank, int64_t propertyDimension, int64_t branchDimension, int64_t segmentDimension, int64_t bucketDimension, int64_t targetConceptDimension, int64_t targetFeatureDimension, const std::string& invalid)
{
	std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> result;
	TORCH_CHECK(enabled, invalid);
	if(enabled)
	{
		// Validate the axis constants before using any of them for unchecked native indexing.
		TORCH_CHECK(zero == 0 && one == 1 && propertyDimension == zero && branchDimension == one && segmentDimension == branchDimension + one && bucketDimension == segmentDimension + one && targetConceptDimension == bucketDimension + one && targetFeatureDimension == targetConceptDimension + one && sourceRank == targetFeatureDimension && connectionRank == sourceRank + one, invalid);
		TORCH_CHECK(connectionFeatureSegments >= zero && columnSegments <= std::numeric_limits<int64_t>::max() - connectionFeatureSegments, invalid);
		TORCH_CHECK(shape.size() == sourceRank && shape[propertyDimension] == one && shape[branchDimension] == one && columnSegments > zero && activationFeatureSegments > zero && connectionFeatureSegments >= activationFeatureSegments && columnSegments + connectionFeatureSegments == shape[segmentDimension], invalid);
		TORCH_CHECK(shape[bucketDimension] > zero && shape[targetConceptDimension] > zero && shape[bucketDimension] <= std::numeric_limits<int64_t>::max() / shape[targetConceptDimension], invalid);
		const int64_t cs = concepts.numel();
		const int64_t fs = featureMap.numel();
		TORCH_CHECK(cs > zero && fs > zero, invalid);
		for(const auto& tensor : {concepts, starts, ends, conceptMap, featureMap, steps}) TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kStrided && tensor.scalar_type() == at::kLong && tensor.dim() == one && tensor.is_contiguous(), invalid);
		TORCH_CHECK(starts.numel() == cs && ends.numel() == cs && conceptMap.numel() == cs && steps.numel() == cs, invalid);
		TORCH_CHECK(proximity.device().is_cpu() && proximity.layout() == at::kStrided && proximity.scalar_type() == at::kFloat && proximity.dim() == one && proximity.numel() >= fs && proximity.is_contiguous() && !proximity.requires_grad(), invalid);
		auto cp = concepts.accessor<int64_t, 1>();
		auto begin = starts.accessor<int64_t, 1>();
		auto end = ends.accessor<int64_t, 1>();
		auto cm = conceptMap.accessor<int64_t, 1>();
		auto fm = featureMap.accessor<int64_t, 1>();
		auto step = steps.accessor<int64_t, 1>();
		auto weights = proximity.accessor<float, 1>();
		std::vector<int64_t> owner(fs, -one);
		std::vector<int64_t> activeWords;
		for(int64_t column = zero; column < cs; ++column)
		{
			TORCH_CHECK(begin[column] >= zero && begin[column] <= cp[column] && cp[column] < end[column] && end[column] <= fs && cm[column] >= zero && cm[column] < shape[bucketDimension], invalid);
			TORCH_CHECK(column == zero ? step[column] == zero : (cp[column] > cp[column-one] && begin[column] >= end[column-one] && step[column] >= step[column-one] && step[column] <= step[column-one] + one), invalid);
			for(int64_t word = begin[column]; word < end[column]; ++word)
			{
				owner[word] = column;
				activeWords.push_back(word);
			}
		}
		for(int64_t word = zero; word < fs; ++word) TORCH_CHECK(fm[word] >= zero && fm[word] < shape[targetConceptDimension] && std::isfinite(weights[word]), invalid);
		// Every active column includes input segment zero; eligibility has a shared active segment.
		// Generate sequence coordinates in exactly the original segment/source/target order.
		struct Pair { int64_t source; int64_t target; };
		std::vector<std::vector<Pair>> pairs(shape[segmentDimension]);
		int64_t finalColumn = zero;
		for(const auto source : activeWords)
		{
			const int64_t sourceColumn = owner[source];
			finalColumn = std::max(finalColumn, sourceColumn);
			while(finalColumn + one < cs && step[finalColumn + one] - step[sourceColumn] < columnSegments) ++finalColumn;
			const int64_t featureEnd = source > std::numeric_limits<int64_t>::max() - connectionFeatureSegments ? std::numeric_limits<int64_t>::max() : source + connectionFeatureSegments;
			const int64_t lastWord = std::max(end[finalColumn]-one, std::min(fs-one, featureEnd));
			auto target = std::lower_bound(activeWords.begin(), activeWords.end(), source + int64_t(!allowSelf));
			while(target != activeWords.end() && *target <= lastWord)
			{
				const int64_t targetColumn = owner[*target];
				const int64_t wordDistance = *target - source;
				const int64_t columnDistance = step[targetColumn] - step[sourceColumn];
				if((!nextWordOnly || wordDistance <= one) && (!nextColumnOnly || targetColumn <= sourceColumn + one))
				{
					if(columnDistance < columnSegments) pairs[columnSegments - columnDistance - one].push_back({source, *target});
					if(wordDistance <= connectionFeatureSegments) pairs[columnSegments + connectionFeatureSegments - std::max<int64_t>(one, wordDistance)].push_back({source, *target});
				}
				++target;
			}
		}
		int64_t count = zero;
		for(const auto& segment : pairs)
		{
			TORCH_CHECK(int64_t(segment.size()) <= std::numeric_limits<int64_t>::max() - count, invalid);
			count += segment.size();
		}
		auto connectionIndices = at::empty({connectionRank, count}, concepts.options());
		auto connectionValues = at::empty({count}, proximity.options());
		auto sourceKeys = at::empty({count}, concepts.options());
		auto ci = connectionIndices.accessor<int64_t, 2>();
		auto cv = connectionValues.accessor<float, 1>();
		auto sk = sourceKeys.accessor<int64_t, 1>();
		int64_t output = zero;
		for(int64_t segment = zero; segment < shape[segmentDimension]; ++segment)
		{
			for(const auto& pair : pairs[segment])
			{
				const int64_t sourceColumn = owner[pair.source];
				const int64_t targetColumn = owner[pair.target];
				ci[propertyDimension][output] = zero;
				ci[branchDimension][output] = zero;
				ci[segmentDimension][output] = segment;
				ci[targetConceptDimension][output] = cm[targetColumn];
				ci[targetFeatureDimension][output] = fm[pair.target];
				sk[output] = cm[sourceColumn] * shape[targetConceptDimension] + fm[pair.source];
				cv[output] = weights[pair.target-pair.source];
				++output;
			}
		}
		const auto connectionUnique = at::_unique2(sourceKeys, true, true, false);
		const auto connectionKeys = std::get<0>(connectionUnique);
		connectionIndices.select(zero, bucketDimension).copy_(std::get<1>(connectionUnique));
		const std::vector<int64_t> connectionShape = {one, one, shape[segmentDimension], connectionKeys.numel(), shape[bucketDimension], shape[targetConceptDimension]};
		auto connectionPacket = at::sparse_coo_tensor(connectionIndices, connectionValues, connectionShape, proximity.options().layout(at::kSparse)).coalesce();
		int64_t neuronCount = zero;
		for(int64_t column = zero; column < cs; ++column)
		{
			const int64_t activeSegments = std::min(column + one, columnSegments) + std::min(cp[column] + one, activationFeatureSegments);
			TORCH_CHECK(end[column] - begin[column] <= (std::numeric_limits<int64_t>::max() - neuronCount) / activeSegments, invalid);
			neuronCount += (end[column] - begin[column]) * activeSegments;
		}
		auto neuronIndices = at::empty({sourceRank, neuronCount}, concepts.options());
		auto neuronValues = at::ones({neuronCount}, proximity.options());
		const auto neuronUnique = at::_unique2(conceptMap, true, true, false);
		const auto neuronKeys = std::get<0>(neuronUnique);
		const auto neuronInverse = std::get<1>(neuronUnique);
		auto ni = neuronIndices.accessor<int64_t, 2>();
		auto inverse = neuronInverse.accessor<int64_t, 1>();
		output = zero;
		for(int64_t segment = zero; segment < shape[segmentDimension]; ++segment)
		{
			for(int64_t column = zero; column < cs; ++column)
			{
				const bool active = segment < std::min(column + one, columnSegments) || (segment >= columnSegments && segment < columnSegments + std::min(cp[column] + one, activationFeatureSegments));
				if(active)
				{
					for(int64_t word = begin[column]; word < end[column]; ++word)
					{
						ni[propertyDimension][output] = zero;
						ni[branchDimension][output] = zero;
						ni[segmentDimension][output] = segment;
						ni[bucketDimension][output] = inverse[column];
						ni[targetConceptDimension][output] = fm[word];
						++output;
					}
				}
			}
		}
		TORCH_CHECK(output == neuronCount, invalid);
		const std::vector<int64_t> neuronShape = {one, one, shape[segmentDimension], neuronKeys.numel(), shape[targetConceptDimension]};
		auto neuronPacket = at::sparse_coo_tensor(neuronIndices, neuronValues, neuronShape, proximity.options().layout(at::kSparse)).coalesce();
		result = std::make_tuple(neuronKeys, neuronPacket, connectionKeys, connectionPacket);
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	module.def("preparePackets", &preparePackets, pybind11::call_guard<pybind11::gil_scoped_release>());
}
