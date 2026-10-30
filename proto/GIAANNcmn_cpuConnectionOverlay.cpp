// Sparse overlays keep sequence additions in float32 order and materialise ordinary COO at read/save boundaries.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class ConnectionOverlay
{
public:
	using Key = std::array<int64_t, 5>;
	struct KeyHash
	{
		bool enabled;
		size_t operator()(const Key& key) const
		{
			size_t result = 0;
			if(enabled)
			{
				result = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(key.data()), sizeof(Key)));
			}
			return result;
		}
	};
	ConnectionOverlay(bool enabled, const at::Tensor& tensor, int64_t minimumEntries, double compactionFraction, const std::string& invalidInput);
	void add(const at::TensorAccessor<int64_t, 2>& indices, const at::TensorAccessor<float, 1>& values, const std::vector<int64_t>& positions, const std::vector<int64_t>& shape);
	at::Tensor materialise(const std::vector<int64_t>& shape);
	at::Tensor baseTensor() const;
	void validateShape(const std::vector<int64_t>& shape) const;

private:
	bool enabled;
	at::Tensor base;
	std::vector<int64_t> currentShape;
	std::unordered_map<Key, float, KeyHash> entries;
	int64_t minimumEntries;
	double compactionFraction;
	std::string invalidInput;
	void copyOld(const at::TensorAccessor<int64_t, 2>& indices, const at::TensorAccessor<float, 1>& values, at::TensorAccessor<int64_t, 2>& outputIndices, at::TensorAccessor<float, 1>& outputValues, int64_t begin, int64_t end, int64_t destination) const;
	int64_t lowerBound(const Key& key, const at::TensorAccessor<int64_t, 2>& indices) const;
	Key keyAt(const at::TensorAccessor<int64_t, 2>& indices, int64_t position) const;
};

std::vector<at::Tensor> updateConnectionOverlays(bool enabled, const std::vector<std::shared_ptr<ConnectionOverlay>>& overlays, const at::Tensor& updates, const std::vector<int64_t>& shape, int64_t grainSize, const std::string& invalidInput);
std::vector<std::shared_ptr<ConnectionOverlay>> createConnectionOverlays(bool enabled, const std::vector<at::Tensor>& sources, int64_t minimumEntries, double compactionFraction, int64_t grainSize, const std::string& invalidInput);
std::vector<at::Tensor> materialiseConnectionOverlays(bool enabled, const std::vector<std::shared_ptr<ConnectionOverlay>>& overlays, const std::vector<int64_t>& shape, int64_t grainSize, const std::string& invalidInput);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
	pybind11::class_<ConnectionOverlay, std::shared_ptr<ConnectionOverlay>>(module, "ConnectionOverlay").def(pybind11::init<bool, const at::Tensor&, int64_t, double, const std::string&>()).def("materialise", &ConnectionOverlay::materialise, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("updateConnectionOverlays", &updateConnectionOverlays, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("createConnectionOverlays", &createConnectionOverlays, pybind11::call_guard<pybind11::gil_scoped_release>());
	module.def("materialiseConnectionOverlays", &materialiseConnectionOverlays, pybind11::call_guard<pybind11::gil_scoped_release>());
}

std::vector<at::Tensor> updateConnectionOverlays(bool enabled, const std::vector<std::shared_ptr<ConnectionOverlay>>& overlays, const at::Tensor& updates, const std::vector<int64_t>& shape, int64_t grainSize, const std::string& invalidInput)
{
	std::vector<at::Tensor> result;
	TORCH_CHECK(enabled, invalidInput);
	if(enabled)
	{
		TORCH_CHECK(grainSize > 0 && shape.size() == ConnectionOverlay::Key{}.size(), invalidInput);
		TORCH_CHECK(updates.device().is_cpu() && updates.layout() == at::kSparse && updates.scalar_type() == at::kFloat && updates.sparse_dim() == int64_t(shape.size()) + 1 && updates.dense_dim() == 0 && updates.is_coalesced() && !updates.requires_grad(), invalidInput);
		TORCH_CHECK(updates.size(3) == int64_t(overlays.size()), invalidInput);
		for(size_t dimension = 0; dimension < shape.size(); ++dimension) TORCH_CHECK(shape[dimension] > 0 && updates.size(dimension + (dimension >= 3)) == shape[dimension], invalidInput);
		std::set<const ConnectionOverlay*> unique;
		for(const auto& overlay : overlays)
		{
			TORCH_CHECK(overlay && unique.insert(overlay.get()).second, invalidInput);
			overlay->validateShape(shape);
		}
		const auto indexTensor = updates.indices();
		const auto valueTensor = updates.values();
		const auto indices = indexTensor.accessor<int64_t, 2>();
		const auto values = valueTensor.accessor<float, 1>();
		std::vector<std::vector<int64_t>> positions(overlays.size());
		std::array<int64_t, ConnectionOverlay::Key{}.size() + 1> previous{};
		for(int64_t position = 0; position < updates._nnz(); ++position)
		{
			std::array<int64_t, ConnectionOverlay::Key{}.size() + 1> key{};
			for(int64_t dimension = 0; dimension < updates.dim(); ++dimension)
			{
				key[dimension] = indices[dimension][position];
				TORCH_CHECK(key[dimension] >= 0 && key[dimension] < updates.size(dimension), invalidInput);
			}
			TORCH_CHECK((position == 0 || previous < key) && std::isfinite(values[position]), invalidInput);
			previous = key;
			positions[key[3]].push_back(position);
		}
		at::parallel_for(0, overlays.size(), grainSize, [&](int64_t begin, int64_t end) { for(int64_t bucket = begin; bucket < end; ++bucket) overlays[bucket]->add(indices, values, positions[bucket], shape); });
		for(const auto& overlay : overlays) result.push_back(overlay->baseTensor());
	}
	return result;
}

std::vector<std::shared_ptr<ConnectionOverlay>> createConnectionOverlays(bool enabled, const std::vector<at::Tensor>& sources, int64_t minimumEntries, double compactionFraction, int64_t grainSize, const std::string& invalidInput)
{
	std::vector<std::shared_ptr<ConnectionOverlay>> result;
	TORCH_CHECK(enabled && grainSize > 0, invalidInput);
	if(enabled)
	{
		result.resize(sources.size());
		at::parallel_for(0, sources.size(), grainSize, [&](int64_t begin, int64_t end) { for(int64_t source = begin; source < end; ++source) result[source] = std::make_shared<ConnectionOverlay>(enabled, sources[source], minimumEntries, compactionFraction, invalidInput); });
	}
	return result;
}

std::vector<at::Tensor> materialiseConnectionOverlays(bool enabled, const std::vector<std::shared_ptr<ConnectionOverlay>>& overlays, const std::vector<int64_t>& shape, int64_t grainSize, const std::string& invalidInput)
{
	std::vector<at::Tensor> result;
	TORCH_CHECK(enabled && grainSize > 0, invalidInput);
	if(enabled)
	{
		for(const auto& overlay : overlays) TORCH_CHECK(overlay, invalidInput);
		result.resize(overlays.size());
		at::parallel_for(0, overlays.size(), grainSize, [&](int64_t begin, int64_t end) { for(int64_t source = begin; source < end; ++source) result[source] = overlays[source]->materialise(shape); });
	}
	return result;
}

ConnectionOverlay::ConnectionOverlay(bool enabled, const at::Tensor& tensor, int64_t minimumEntries, double compactionFraction, const std::string& invalidInput): enabled(enabled), entries(0, KeyHash{enabled}), minimumEntries(minimumEntries), compactionFraction(compactionFraction), invalidInput(invalidInput)
{
	TORCH_CHECK(enabled, invalidInput);
	if(enabled)
	{
		TORCH_CHECK(minimumEntries > 0 && std::isfinite(compactionFraction) && compactionFraction > 0, invalidInput);
		TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat && tensor.sparse_dim() == int64_t(Key{}.size()) && tensor.dense_dim() == 0 && tensor.is_coalesced() && !tensor.requires_grad(), invalidInput);
		base = tensor;
		currentShape = tensor.sizes().vec();
		validateShape(tensor.sizes().vec());
		const auto indexTensor = tensor.indices();
		const auto valueTensor = tensor.values();
		const auto indices = indexTensor.accessor<int64_t, 2>();
		const auto values = valueTensor.accessor<float, 1>();
		Key previous{};
		for(int64_t position = 0; position < tensor._nnz(); ++position)
		{
			const auto key = keyAt(indices, position);
			for(size_t dimension = 0; dimension < key.size(); ++dimension) TORCH_CHECK(key[dimension] >= 0 && key[dimension] < tensor.size(dimension), invalidInput);
			TORCH_CHECK((position == 0 || previous < key) && std::isfinite(values[position]), invalidInput);
			previous = key;
		}
	}
}

void ConnectionOverlay::add(const at::TensorAccessor<int64_t, 2>& indices, const at::TensorAccessor<float, 1>& values, const std::vector<int64_t>& positions, const std::vector<int64_t>& shape)
{
	if(enabled)
	{
		currentShape = shape;
		const auto baseIndexTensor = base.indices();
		const auto baseValueTensor = base.values();
		const auto baseIndices = baseIndexTensor.accessor<int64_t, 2>();
		const auto baseValues = baseValueTensor.accessor<float, 1>();
		for(const auto position : positions)
		{
			Key key{indices[0][position], indices[1][position], indices[2][position], indices[4][position], indices[5][position]};
			auto found = entries.find(key);
			float value = values[position];
			if(found != entries.end())
			{
				value = found->second + value;
				TORCH_CHECK(std::isfinite(value), invalidInput);
				found->second = value;
			}
			else
			{
				const int64_t insertion = lowerBound(key, baseIndices);
				if(insertion < base._nnz() && keyAt(baseIndices, insertion) == key) value = baseValues[insertion] + value;
				TORCH_CHECK(std::isfinite(value), invalidInput);
				entries.emplace(key, value);
			}
		}
		if(double(entries.size()) >= std::max(double(minimumEntries), double(base._nnz()) * compactionFraction))
		{
			base = materialise(shape);
			entries.clear();
		}
	}
}

at::Tensor ConnectionOverlay::materialise(const std::vector<int64_t>& shape)
{
	at::Tensor result;
	if(enabled)
	{
		validateShape(shape);
		if(entries.empty())
		{
			result = at::sparse_coo_tensor(base.indices(), base.values(), shape, base.options(), true);
		}
		else
		{
			TORCH_CHECK(entries.size() <= size_t(std::numeric_limits<int64_t>::max() - base._nnz()), invalidInput);
			const auto baseIndexTensor = base.indices();
			const auto baseValueTensor = base.values();
			const auto indices = baseIndexTensor.accessor<int64_t, 2>();
			const auto values = baseValueTensor.accessor<float, 1>();
			std::vector<std::pair<Key, float>> ordered(entries.begin(), entries.end());
			std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
			std::vector<int64_t> insertions;
			int64_t count = base._nnz();
			for(const auto& entry : ordered)
			{
				const int64_t insertion = lowerBound(entry.first, indices);
				insertions.push_back(insertion);
				count += insertion == base._nnz() || keyAt(indices, insertion) != entry.first;
			}
			auto indexTensor = at::empty({base.dim(), count}, base.indices().options());
			auto valueTensor = at::empty({count}, base.values().options());
			auto outputIndices = indexTensor.accessor<int64_t, 2>();
			auto outputValues = valueTensor.accessor<float, 1>();
			int64_t oldPosition = 0;
			int64_t outputPosition = 0;
			int64_t entryPosition = 0;
			for(const auto& entry : ordered)
			{
				const int64_t insertion = insertions[entryPosition];
				copyOld(indices, values, outputIndices, outputValues, oldPosition, insertion, outputPosition);
				outputPosition += insertion - oldPosition;
				for(size_t dimension = 0; dimension < entry.first.size(); ++dimension) outputIndices[dimension][outputPosition] = entry.first[dimension];
				outputValues[outputPosition] = entry.second;
				oldPosition = insertion + int64_t(insertion < base._nnz() && keyAt(indices, insertion) == entry.first);
				++outputPosition;
				++entryPosition;
			}
			copyOld(indices, values, outputIndices, outputValues, oldPosition, base._nnz(), outputPosition);
			TORCH_CHECK(outputPosition + base._nnz() - oldPosition == count, invalidInput);
			result = at::sparse_coo_tensor(indexTensor, valueTensor, shape, base.options(), true);
		}
	}
	return result;
}

at::Tensor ConnectionOverlay::baseTensor() const
{
	at::Tensor result;
	if(enabled)
	{
		result = base;
	}
	return result;
}

void ConnectionOverlay::validateShape(const std::vector<int64_t>& shape) const
{
	if(enabled)
	{
		TORCH_CHECK(shape.size() == Key{}.size(), invalidInput);
		for(size_t dimension = 0; dimension < shape.size(); ++dimension) TORCH_CHECK(shape[dimension] > 0 && shape[dimension] >= currentShape[dimension], invalidInput);
	}
}

void ConnectionOverlay::copyOld(const at::TensorAccessor<int64_t, 2>& indices, const at::TensorAccessor<float, 1>& values, at::TensorAccessor<int64_t, 2>& outputIndices, at::TensorAccessor<float, 1>& outputValues, int64_t begin, int64_t end, int64_t destination) const
{
	if(enabled)
	{
		const int64_t length = end - begin;
		if(length > 0)
		{
			for(int64_t dimension = 0; dimension < base.dim(); ++dimension)
			{
				if(indices.stride(1) == 1) std::copy_n(indices[dimension].data() + begin, length, outputIndices[dimension].data() + destination);
				else for(int64_t offset = 0; offset < length; ++offset) outputIndices[dimension][destination + offset] = indices[dimension][begin + offset];
			}
			if(values.stride(0) == 1) std::copy_n(values.data() + begin, length, outputValues.data() + destination);
			else for(int64_t offset = 0; offset < length; ++offset) outputValues[destination + offset] = values[begin + offset];
		}
	}
}

int64_t ConnectionOverlay::lowerBound(const Key& key, const at::TensorAccessor<int64_t, 2>& indices) const
{
	int64_t low = 0;
	if(enabled)
	{
		int64_t high = base._nnz();
		while(low < high)
		{
			const int64_t middle = low + (high - low) / 2;
			if(keyAt(indices, middle) < key) low = middle + 1;
			else high = middle;
		}
	}
	return low;
}

ConnectionOverlay::Key ConnectionOverlay::keyAt(const at::TensorAccessor<int64_t, 2>& indices, int64_t position) const
{
	Key result{};
	if(enabled)
	{
		for(size_t dimension = 0; dimension < result.size(); ++dimension) result[dimension] = indices[dimension][position];
	}
	return result;
}
