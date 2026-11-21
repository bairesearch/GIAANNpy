#pragma once
// CPU staging and publication for GIAANN sparse training batches.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <array>
#include <cfenv>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace {
using Key = uint64_t;
using Size = int64_t;
constexpr Size sourceRank = 5, propertyAxis = 0, branchAxis = 1, segmentAxis = 2, conceptAxis = 3, featureAxis = 4, recordWidth = sourceRank + 1;
struct Source {
	py::object destination;
	py::object feature, previous;
	at::Tensor indices, weights;
	const char* records = nullptr;
	Size count = 0;
	bool present = false;
};
struct Pending {
	std::vector<Source> sources;
	py::list outputs;
	py::dict updatedFeatures;
	Size inputEntries{}, outputEntries{}, tiles{};
	void prepareOutstanding(py::dict groups, py::tuple target, py::object device, const std::string& invalid);

	void publish() {
		if (outputs.size() != sources.size())
			throw std::logic_error("Incomplete sparse batch");
		size_t written = 0;
		try {
			for (; written < sources.size(); ++written)
				sources[written].destination[sources[written].feature] = outputs[written];
		} catch (...) {
			while (written) {
				auto& src = sources[--written];
				if (src.present)
					src.destination[src.feature] = src.previous;
				else
					src.destination.attr("pop")(src.feature);
			}
			throw;
		}
	}
};
void validateCOO(const at::Tensor& tensor, Size rank, const std::string& invalid) {
	TORCH_CHECK(tensor.device().is_cpu() && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat && tensor.sparse_dim() == rank && !tensor.dense_dim() && tensor.is_coalesced() && !tensor.requires_grad(), invalid);
}
void checkRecords(const py::array& a, const py::dtype& dtype, const std::string& invalid) {
	TORCH_CHECK(a.ndim() == 1 && (a.flags() & py::array::c_style) && a.writeable() && (a.dtype().is(dtype) || a.dtype().equal(dtype)), invalid);
}
py::object wrapped(const py::object& factory, const py::object& cls, py::array data, const py::tuple& shape) {
	py::object source = factory(cls);
	source.attr("entries") = std::move(data);
	source.attr("tensorSize") = shape;
	return source;
}
void Pending::prepareOutstanding(py::dict groups, py::tuple target, py::object device, const std::string& invalid) {
	for (auto item : groups) {
		const auto group = py::cast<py::tuple>(item.second);
		py::object column = group[0];
		TORCH_CHECK(column.attr("isDatabaseResidentConnectionStorage")().cast<bool>() && column.attr("getFeatureConnectionsTargetSize")().cast<py::tuple>().equal(target), invalid);
		auto features = group[1].cast<py::set>();
		auto loaded = column.attr("loadedSourceFeatureIndices").cast<py::set>();
		auto remaining = updatedFeatures.contains(item.first) ? features.attr("difference")(updatedFeatures[item.first]).cast<py::set>() : features;
		if (remaining.size())
			column.attr("prepareRequiredSourceFeatureConnectionsTrain")(remaining, device);
		loaded.attr("update")(features);
		column.attr("trainPreparedSourceFeatureIndices") = group[2].attr("copy")();
	}
}
int requiredBits(Key largest) {
	return largest ? 64 - __builtin_clzll(largest) : 0;
}

struct Geometry {
	std::vector<Size> shape, logicalShape;
	bool neurons;
	Key extent = 1;
	int featureBits = 0, conceptBits = 0, targetBits = 0;
	bool packedBits = false;

	Geometry(py::tuple target, bool packed, bool neuronBatch, Size bucketAxis, const py::dtype& dtype, const std::string& invalid)
		: shape(target.cast<std::vector<Size>>()), logicalShape(shape), neurons(neuronBatch) {
		if (neurons) {
			TORCH_CHECK(!packed && shape.size() == sourceRank - 1, invalid);
			shape.insert(shape.begin() + conceptAxis, 1);
		}
		TORCH_CHECK(shape.size() == sourceRank && bucketAxis == conceptAxis, invalid);
		TORCH_CHECK(!packed || dtype.itemsize() == recordWidth * sizeof(int32_t), invalid);
		for (Size n : shape) {
			TORCH_CHECK(n > 0 && extent <= Key(std::numeric_limits<Size>::max()) / n, invalid);
			TORCH_CHECK(!packed || n <= Size(std::numeric_limits<int32_t>::max()) + 1, invalid);
			extent *= n;
		}
		featureBits = requiredBits(shape[featureAxis] - 1);
		conceptBits = requiredBits(shape[conceptAxis] - 1);
		targetBits = featureBits + conceptBits + requiredBits(shape[segmentAxis] - 1);
		packedBits = packed && shape[propertyAxis] == 1 && shape[branchAxis] == 1 && targetBits < 63;
		if (packedBits)
			extent = Key(1) << targetBits;
	}

	inline Key encode(Size property, Size branch, Size segment, Size concept, Size feature, const std::string& invalid) const {
		TORCH_CHECK(Key(property) < Key(shape[propertyAxis]) && Key(branch) < Key(shape[branchAxis]) && Key(segment) < Key(shape[segmentAxis]) && Key(concept) < Key(shape[conceptAxis]) && Key(feature) < Key(shape[featureAxis]), invalid);
		Key key;
		if (packedBits)
			key = ((Key(segment) << conceptBits | Key(concept)) << featureBits) | Key(feature);
		else
			key = (((Key(property) * shape[branchAxis] + branch) * shape[segmentAxis] + segment) * shape[conceptAxis] + concept) * shape[featureAxis] + feature;
		return key;
	}
};

struct Packet {
	at::Tensor indices, values;
	std::vector<Size> sources;
};
struct Tile {
	size_t first, end;
	Size historical = 0, updates = 0, updateStart = 0;
};

std::vector<Packet> readPackets(py::list packets, const Geometry& geometry, std::vector<Size>& allKeys, const std::string& invalid) {
	std::unordered_map<Size, size_t> positions;
	std::vector<Packet> result;
	for (auto args : packets) {
		auto pair = py::cast<py::tuple>(args);
		const auto keyTensor = pair[0].cast<at::Tensor>();
		TORCH_CHECK(keyTensor.device().is_cpu() && keyTensor.layout() == at::kStrided && keyTensor.scalar_type() == at::kLong && keyTensor.dim() == 1, invalid);
		auto contiguous = keyTensor.contiguous();
		std::vector<Size> local(contiguous.numel());
		if (!local.empty())
			std::memcpy(local.data(), contiguous.data_ptr<Size>(), local.size() * sizeof(Size));
		auto t = pair[1].cast<at::Tensor>();
		validateCOO(t, geometry.logicalShape.size() + 1, invalid);
		TORCH_CHECK(t.size(conceptAxis) == Size(local.size()), invalid);
		for (size_t d = 0; d < geometry.logicalShape.size(); ++d)
			TORCH_CHECK(t.size(d + (d >= conceptAxis)) == geometry.logicalShape[d], invalid);
		for (size_t j = 0; j < local.size(); ++j) {
			const Size key = local[j];
			TORCH_CHECK(key >= 0 && (geometry.neurons || key / geometry.shape[featureAxis] < geometry.shape[conceptAxis]) && (!j || local[j - 1] < key), invalid);
			if (positions.emplace(key, 0).second)
				allKeys.push_back(key);
		}
		result.push_back({t.indices(), t.values(), std::move(local)});
	}
	std::sort(allKeys.begin(), allKeys.end());
	for (size_t i = 0; i < allKeys.size(); ++i)
		positions[allKeys[i]] = i;
	for (auto& packet : result)
		for (auto& key : packet.sources)
			key = positions.at(key);
	return result;
}

void collectNeurons(Pending& pending, py::dict columns, const std::vector<Size>& names, const Geometry& geo, bool allowHostMerge, const std::string& invalid) {
	pending.sources.reserve(names.size());
	for (auto key : names) {
		auto column = columns[py::int_(key)];
		auto tensor = column.attr("featureNeurons").cast<at::Tensor>().coalesce();
		validateCOO(tensor, geo.logicalShape.size(), invalid);
		TORCH_CHECK(tensor.sizes().vec() == geo.logicalShape, invalid);
		Source source;
		if (allowHostMerge)
			source.previous = py::cast(tensor);
		source.count = tensor._nnz();
		source.indices = tensor.indices();
		source.weights = tensor.values();
		pending.sources.push_back(std::move(source));
	}
}

bool collectSources(Pending& pending, py::dict columns, py::dict deferred, const std::vector<Size>& keys, const Geometry& geo, py::tuple target, bool packed, py::object cls, const py::dtype& dtype, const std::string& invalid, bool allowPackedGrowth = false) {
	Size activeColumn = -1;
	py::dict dest;
	py::set loaded, allowed, touched;
	bool compatible = true;
	pending.sources.reserve(keys.size());
	for (const auto key : keys) {
		const Size conceptIndex = key / geo.shape[featureAxis];
		if (conceptIndex != activeColumn) {
			activeColumn = conceptIndex;
			py::int_ conceptKey(conceptIndex);
			auto col = columns[conceptKey];
			dest = col.attr("featureConnectionsBySourceFeature").cast<py::dict>();
			loaded = col.attr("loadedSourceFeatureIndices").cast<py::set>();
			allowed = deferred.contains(conceptKey) ? deferred[conceptKey].cast<py::tuple>()[1].cast<py::set>() : py::set();
			touched = py::set();
			if (deferred.size())
				pending.updatedFeatures[conceptKey] = touched;
		}
		py::int_ feature(key % geo.shape[featureAxis]);
		const bool requested = deferred.size() && allowed.contains(feature);
		if (requested)
			touched.add(feature);
		TORCH_CHECK(requested || loaded.contains(feature), invalid);
		PyObject* object = PyDict_GetItemWithError(dest.ptr(), feature.ptr());
		if (!object && PyErr_Occurred())
			throw py::error_already_set();
		Source source;
		source.destination = dest;
		source.feature = feature;
		source.present = object != nullptr;
		source.previous = source.present ? py::reinterpret_borrow<py::object>(object) : py::none();
		if (!object || object == Py_None) {
			TORCH_CHECK(requested, invalid);
		} else if (packed) {
			TORCH_CHECK(py::isinstance(source.previous, cls), invalid);
			auto priorShape = source.previous.attr("tensorSize").cast<py::tuple>();
			if (!priorShape.is(target)) {
				TORCH_CHECK(priorShape.size() == target.size(), invalid);
				for (auto dimension : priorShape)
					TORCH_CHECK(PyLong_CheckExact(dimension.ptr()) && dimension.cast<Size>() > 0 && dimension.cast<Size>() <= Size(std::numeric_limits<int32_t>::max()) + 1, invalid);
			}
			if (!priorShape.equal(target)) {
				if (!allowPackedGrowth) {
					compatible = false;
					break;
				}
				// A packed record stores coordinates independently of the vocabulary size.
				// Publish the expanded shape with the result, leaving the old column untouched until commit.
				for (Size d = 0; d < sourceRank; ++d) {
					const Size oldSize = priorShape[d].cast<Size>();
					TORCH_CHECK((d == conceptAxis || d == featureAxis) ? oldSize <= geo.shape[d] : oldSize == geo.shape[d], invalid);
				}
			}
			auto a = source.previous.attr("entries").cast<py::array>();
			checkRecords(a, dtype, invalid);
			source.records = static_cast<const char*>(a.data());
			source.count = a.size();
		} else {
			auto t = source.previous.cast<at::Tensor>().coalesce();
			validateCOO(t, sourceRank, invalid);
			if (t.sizes().vec() != geo.shape && deferred.size()) {
				compatible = false;
				break;
			}
			TORCH_CHECK(t.sizes().vec() == geo.shape, invalid);
			source.count = t._nnz();
			source.indices = t.indices();
			source.weights = t.values();
		}
		pending.sources.push_back(std::move(source));
	}
	return compatible;
}

struct NonfiniteHistory : std::exception {};

void packHistory(const Source& source, const Geometry& geo, Key prefix, Key* keys, float* values, const std::string& invalid) {
	Key previous = 0;
	if (source.records) {
		for (Size i = 0; i < source.count; ++i) {
			int32_t row[recordWidth];
			std::memcpy(row, source.records + i * sizeof(row), sizeof(row));
			const auto key = geo.encode(row[0], row[1], row[2], row[3], row[4], invalid);
			float value;
			std::memcpy(&value, &row[sourceRank], sizeof(value));
			TORCH_CHECK((!i || key > previous) && std::isfinite(value), invalid);
			keys[i] = prefix + key;
			values[i] = value;
			previous = key;
		}
	} else if (source.count) {
		auto ix = source.indices.accessor<Size, 2>();
		auto weights = source.weights.accessor<float, 1>();
		for (Size i = 0; i < source.count; ++i) {
			const auto key = geo.encode(ix[0][i], ix[1][i], ix[2][i], geo.neurons ? 0 : ix[3][i], ix[geo.neurons ? 3 : 4][i], invalid);
			TORCH_CHECK(!i || key > previous, invalid);
			// CPU COO can carry NaNs from an external checkpoint. CUDA and CPU may
			// choose different NaN payloads even in the same addition order.
			if (!std::isfinite(weights[i]))
				throw NonfiniteHistory{};
			keys[i] = prefix + key;
			values[i] = weights[i];
			previous = key;
		}
	}
}

// Lanes own consecutive sequence intervals. The prefix for a lane is after all
// earlier lanes, independent of thread scheduling or the number of ATen threads.
struct TransferPlan {
	std::vector<Tile> tiles;
	std::vector<size_t> tileOf;
	std::vector<Size> historyOffset;
	std::vector<Key> sourcePrefix, updateKeys;
	std::vector<float> updateValues;
	Size capacity = 0;

	bool build(const std::vector<Packet>& packets, const std::vector<Source>& sources, const Geometry& geo, Size budget, Size sourceLimit, Key domainLimit, bool packed, const std::string& invalid) {
		const Size lanes = std::max<Size>(1, std::min<Size>(at::get_num_threads(), packets.size()));
		const size_t rows = sources.size();
		std::vector<std::vector<Size>> counts(lanes, std::vector<Size>(rows));
		auto byLane = [&](auto fn) {
			at::parallel_for(0, lanes, 1, [&](Size begin, Size end) {
				for (Size l = begin; l < end; ++l) {
					for (size_t p = packets.size() * l / lanes; p < packets.size() * (l + 1) / lanes; ++p)
						fn(l, packets[p]);
				}
			});
		};
		byLane([&](Size lane, const Packet& packet) {
			auto ix = packet.indices.accessor<Size, 2>();
			for (Size i = 0; i < packet.values.numel(); ++i) {
				const auto local = ix[conceptAxis][i];
				TORCH_CHECK(Key(local) < packet.sources.size(), invalid);
				++counts[lane][packet.sources[local]];
			}
		});
		tileOf.resize(rows);
		sourcePrefix.resize(rows);
		historyOffset.resize(rows);
		bool fits = true;
		for (size_t i = 0; i < rows; ++i) {
			Size updates = 0;
			for (const auto& lane : counts)
				updates += lane[i];
			const Size history = sources[i].count;
			if (history > sourceLimit || updates > sourceLimit - history) {
				fits = false;
				break;
			}
			const Size n = history + updates;
			if (tiles.empty() || tiles.back().historical + tiles.back().updates + n > budget || i - tiles.back().first >= size_t(budget) || i - tiles.back().first + 1 > domainLimit / geo.extent)
				tiles.push_back({i, i});
			auto& tile = tiles.back();
			tileOf[i] = tiles.size() - 1;
			sourcePrefix[i] = (i - tile.first) * geo.extent;
			historyOffset[i] = tile.historical;
			tile.end = i + 1;
			tile.historical += history;
			tile.updates += updates;
			capacity = std::max(capacity, std::max<Size>(tile.end - tile.first, tile.historical + tile.updates));
		}
		if (fits) {
			std::vector<std::vector<Size>> cursor(lanes, std::vector<Size>(tiles.size()));
			Size entries = 0;
			for (size_t t = 0; t < tiles.size(); ++t) {
				auto& tile = tiles[t];
				tile.updateStart = entries;
				for (Size l = 0; l < lanes; ++l) {
					cursor[l][t] = entries;
					entries = std::accumulate(counts[l].begin() + tile.first, counts[l].begin() + tile.end, entries);
				}
				TORCH_CHECK(entries - tile.updateStart == tile.updates, invalid);
			}
			counts.clear();
			updateKeys.resize(entries);
			updateValues.resize(entries);
			byLane([&](Size lane, const Packet& packet) {
				auto ix = packet.indices.accessor<Size, 2>();
				auto weights = packet.values.accessor<float, 1>();
				std::array<Size, sourceRank + 1> previous{};
				for (Size i = 0; i < packet.values.numel(); ++i) {
					const std::array<Size, sourceRank + 1> row{ix[0][i], ix[1][i], ix[2][i], ix[3][i], geo.neurons ? 0 : ix[4][i], ix[geo.neurons ? 4 : 5][i]};
					TORCH_CHECK(!packed || !i || previous < row, invalid);
					const auto bucket = packet.sources[row[conceptAxis]];
					const auto key = geo.encode(row[0], row[1], row[2], row[4], row[5], invalid);
					const float value = weights[i];
					TORCH_CHECK(std::isfinite(value), invalid);
					const Size destination = cursor[lane][tileOf[bucket]]++;
					updateKeys[destination] = sourcePrefix[bucket] + key;
					updateValues[destination] = value;
					previous = row;
				}
			});
		}
		return fits;
	}
};

void emitGeneric(Pending& batch, const Tile& tile, const Geometry& geo, py::tuple target, bool packed, py::object cls, py::object factory, const py::dtype& dtype, const Key* keys, const float* values, int count, const std::string& invalid) {
	int cursor = 0;
	for (size_t j = tile.first; j < tile.end; ++j) {
		const int start = cursor;
		const Key base = (j - tile.first) * geo.extent;
		while (cursor < count && keys[cursor] < base + geo.extent)
			++cursor;
		const int length = cursor - start;
		if (packed) {
			py::array a(dtype, py::array::ShapeContainer{length});
			auto* dest = static_cast<int32_t*>(a.mutable_data());
			for (int i = 0; i < length; ++i) {
				Key key = keys[start + i] - base;
				for (Size d = sourceRank; d--;) {
					dest[i * recordWidth + d] = key % geo.shape[d];
					key /= geo.shape[d];
				}
				TORCH_CHECK(std::isfinite(values[start + i]), invalid);
				std::memcpy(dest + i * recordWidth + sourceRank, &values[start + i], sizeof(float));
			}
			batch.outputs.append(wrapped(factory, cls, std::move(a), target));
		} else {
			const Size rank = geo.logicalShape.size();
			auto ix = at::empty({rank, length}, at::kLong), v = at::empty({length}, at::kFloat);
			auto* data = ix.data_ptr<Size>();
			for (int i = 0; i < length; ++i) {
				Key key = keys[start + i] - base;
				for (Size d = rank; d--;) {
					data[d * length + i] = key % geo.logicalShape[d];
					key /= geo.logicalShape[d];
				}
			}
			if (length)
				std::memcpy(v.data_ptr<float>(), values + start, length * sizeof(float));
			batch.outputs.append(py::cast(at::sparse_coo_tensor(ix, v, geo.logicalShape, v.options().layout(at::kSparse), true)));
		}
	}
	TORCH_CHECK(cursor == count, invalid);
}

void stageTile(const Tile& tile, const Pending& batch, const TransferPlan& plan, const Geometry& geo, std::vector<Key>& input, std::vector<float>& values, const std::string& invalid) {
	if (tile.historical) {
		at::parallel_for(tile.first, tile.end, 1, [&](Size begin, Size end) {
			for (Size i = begin; i < end; ++i) {
				const auto offset = plan.historyOffset[i];
				packHistory(batch.sources[i], geo, plan.sourcePrefix[i], input.data() + offset, values.data() + offset, invalid);
			}
		});
	}
	if (tile.updates) {
		std::copy_n(plan.updateKeys.data() + tile.updateStart, tile.updates, input.data() + tile.historical);
		std::copy_n(plan.updateValues.data() + tile.updateStart, tile.updates, values.data() + tile.historical);
	}
}

void appendPacked(Pending& batch, const py::object& factory, const py::object& cls, const py::dtype& dtype, const py::tuple& target, const int32_t* records, const int32_t* ends, size_t sources, int count, const std::string& invalid) {
	int cursor = 0;
	for (size_t i = 0; i < sources; ++i) {
		int end = ends[i];
		if (!end)
			end = cursor;
		TORCH_CHECK(end >= cursor && end <= count, invalid);
		py::array array(dtype, py::array::ShapeContainer{end - cursor});
		if (end > cursor)
			std::memcpy(array.mutable_data(), records + Size(cursor) * recordWidth, Size(end - cursor) * recordWidth * sizeof(int32_t));
		batch.outputs.append(wrapped(factory, cls, std::move(array), target));
		cursor = end;
	}
	TORCH_CHECK(cursor == count, invalid);
}

void bindPending(py::module_& module) {
	py::class_<Pending, std::shared_ptr<Pending>>(module, "Pending").def("publish", &Pending::publish).def("prepareOutstanding", &Pending::prepareOutstanding).def_readonly("outputs", &Pending::outputs).def_readonly("tiles", &Pending::tiles).def_readonly("inputEntries", &Pending::inputEntries).def_readonly("outputEntries", &Pending::outputEntries);
}
} // namespace
