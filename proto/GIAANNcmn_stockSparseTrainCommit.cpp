// Only CPU staging lives in this extension. The Python callback runs stock PyTorch on CUDA.
#include "GIAANNcmn_trainCommitTransport.hpp"

namespace {
void reduceTiles(Pending& batch, const TransferPlan& plan, const Geometry& geo, py::tuple target, py::object cls, py::dtype dtype, py::object reduceTile, const std::string& invalid) {
	const bool packed = !geo.neurons;
	auto factory = packed ? cls.attr("__new__").cast<py::object>() : py::none().cast<py::object>();
	std::vector<Key> input(plan.capacity);
	std::vector<float> values(plan.capacity);
	for (const auto& tile : plan.tiles) {
		const Size n = tile.historical + tile.updates;
		const Size sources = tile.end - tile.first;
		TORCH_CHECK(Key(sources) <= Key(std::numeric_limits<Size>::max()) / geo.extent, invalid);
		stageTile(tile, batch, plan, geo, input, values, invalid);
		// from_blob borrows the tile until the callback returns its completed CPU tensors.
		auto keys = n ? at::from_blob(input.data(), {n}, at::kLong) : at::empty({0}, at::kLong);
		auto weights = n ? at::from_blob(values.data(), {n}, at::kFloat) : at::empty({0}, at::kFloat);
		auto outputs = reduceTile(keys, weights, geo.extent, sources, geo.packedBits, geo.conceptBits + geo.featureBits, geo.featureBits).cast<py::tuple>();
		auto first = outputs[0].cast<at::Tensor>(), second = outputs[1].cast<at::Tensor>();
		TORCH_CHECK(first.is_cpu() && second.is_cpu() && first.is_contiguous() && second.is_contiguous(), invalid);
		const Size count = first.size(0);
		TORCH_CHECK(count <= n && count <= std::numeric_limits<int>::max(), invalid);
		if (geo.packedBits) {
			TORCH_CHECK(first.dim() == 2 && first.size(1) == recordWidth && first.scalar_type() == at::kInt && second.dim() == 1 && second.numel() == sources && second.scalar_type() == at::kInt, invalid);
			appendPacked(batch, factory, cls, dtype, target, first.const_data_ptr<int32_t>(), second.const_data_ptr<int32_t>(), sources, int(count), invalid);
		} else {
			TORCH_CHECK(first.dim() == 1 && first.scalar_type() == at::kLong && second.dim() == 1 && second.numel() == count && second.scalar_type() == at::kFloat, invalid);
			emitGeneric(batch, tile, geo, target, packed, cls, factory, dtype, reinterpret_cast<const Key*>(first.const_data_ptr<Size>()), second.const_data_ptr<float>(), int(count), invalid);
		}
		batch.inputEntries += n;
		batch.outputEntries += count;
		batch.tiles += bool(n);
	}
}

py::tuple merge(py::dict columns, py::dict deferred, py::list packets, py::tuple target, Size sourceAxis, bool neurons, py::object cls, py::dtype dtype, Size budget, const std::string& invalid, py::object reduceTile) {
	TORCH_CHECK(budget > 0 && budget <= std::numeric_limits<int>::max(), invalid);
	const Geometry geo(target, !neurons, neurons, sourceAxis, dtype, invalid);
	std::vector<Size> keys;
	auto inputs = readPackets(packets, geo, keys, invalid);
	auto pending = std::make_shared<Pending>();
	if (neurons)
		collectNeurons(*pending, columns, keys, geo, false, invalid);
	else
		TORCH_CHECK(collectSources(*pending, columns, deferred, keys, geo, target, true, cls, dtype, invalid, true), invalid);
	TransferPlan plan;
	// Keep a single large source intact so duplicate coordinates are reduced together.
	TORCH_CHECK(plan.build(inputs, pending->sources, geo, budget, std::numeric_limits<int>::max(), std::numeric_limits<Size>::max(), !neurons, invalid), "Packed sparse source exceeds the supported int32 record count");
	reduceTiles(*pending, plan, geo, target, cls, dtype, reduceTile, invalid);
	return py::make_tuple(keys, pending);
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
	bindPending(module);
	module.def("merge", &merge);
}
