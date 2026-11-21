// Bulk transfer between GIAANN's CPU-resident sources and stable CUDA radix runs.
#include "GIAANNcmn_trainCommitTransport.hpp"
#include <c10/core/DeviceGuard.h>

extern "C" const char* sorter_create(void**, int);
extern "C" void sorter_close(void*);
extern "C" const char* sorter_run_chronological(void*, const uint64_t*, const float*, int, uint64_t, int, uint64_t*, float*, int*);
extern "C" const char* sorter_run_training_records(void*, const uint64_t*, const float*, int, int, int, int, int, int, int32_t*, int*, int*);
extern "C" int coo_gather_host(const int64_t*, int64_t, int64_t, const float*, int64_t, int64_t, const int64_t*, int64_t, int64_t, int64_t, const int64_t*, int, float*, int);
extern "C" const char* coo_gather_device(const int64_t*, int64_t, int64_t, const float*, int64_t, int64_t, const int64_t*, int64_t, int64_t, int64_t, const int64_t*, int, float*, int*, void*);
namespace {
void checkedCuda(const char* error) {
	TORCH_CHECK(!error, error ? error : "");
}
void execute(Pending& batch, const TransferPlan& plan, const Geometry& geo, py::tuple target, bool packed, py::object cls, py::dtype dtype, const std::string& invalid) {
	// A tile can contain named sources that have no stored or pending entries.
	const Size capacity = std::max<Size>(1, plan.capacity);
	void* ctx = nullptr;
	checkedCuda(sorter_create(&ctx, int(capacity)));
	std::unique_ptr<void, decltype(&sorter_close)> workspace(ctx, &sorter_close);
	std::vector<Key> input(capacity), out(geo.packedBits ? 0 : capacity);
	std::vector<float> values(capacity), reduced(geo.packedBits ? 0 : capacity);
	std::vector<int32_t> records(geo.packedBits ? capacity * recordWidth : 0);
	std::vector<int> ends;
	auto factory = packed ? cls.attr("__new__").cast<py::object>() : py::none().cast<py::object>();
	for (const auto& tile : plan.tiles) {
		const Size n = tile.historical + tile.updates;
		const size_t sources = tile.end - tile.first;
		stageTile(tile, batch, plan, geo, input, values, invalid);
		int count = 0;
		const Key domain = sources * geo.extent;
		const int bits = std::max(1, requiredBits(domain - 1));
		if (geo.packedBits) {
			ends.assign(sources, 0);
			if (n)
				checkedCuda(sorter_run_training_records(ctx, input.data(), values.data(), int(n), bits, geo.targetBits, geo.conceptBits, geo.featureBits, int(sources), records.data(), ends.data(), &count));
			appendPacked(batch, factory, cls, dtype, target, records.data(), ends.data(), sources, count, invalid);
		} else {
			if (n)
				checkedCuda(sorter_run_chronological(ctx, input.data(), values.data(), int(n), domain, bits, out.data(), reduced.data(), &count));
			emitGeneric(batch, tile, geo, target, packed, cls, factory, dtype, out.data(), reduced.data(), count, invalid);
		}
		batch.inputEntries += n;
		batch.outputEntries += count;
		batch.tiles += bool(n);
	}
}

bool ieeeHost() {
	bool supported = std::fegetround() == FE_TONEAREST;
#if defined(__SSE2__)
	// MXCSR bits 15 and 6 select flush-to-zero and denormals-are-zero.
	constexpr unsigned flushModes = (1u << 15) | (1u << 6);
	supported = supported && !(_mm_getcsr() & flushModes);
#endif
	return supported;
}
// For small changes to large, CPU-resident COO histories the existing CPU
// merge binary-searches only the changed coordinates and copies untouched spans.
bool preferHost(const std::vector<Source>& sources, const std::vector<Packet>& inputs, Size ratio) {
	bool selected = false;
	if (ratio > 0) {
		Size updates = 0, history = 0;
		for (const auto& packet : inputs)
			updates += packet.values.numel();
		for (const auto& source : sources)
			history += source.count;
		selected = updates > 0 && history / updates >= ratio;
	}
	return selected;
}
void mergeHost(Pending& pending, const std::vector<Packet>& inputs, py::list packets, py::tuple target, py::object hostMerge) {
	py::list sources, updates, mappings;
	at::Tensor empty;
	for (const auto& source : pending.sources) {
		if (!source.previous.is_none())
			sources.append(py::cast(source.previous.cast<at::Tensor>().coalesce()));
		else {
			if (!empty.defined()) {
				const auto shape = target.cast<std::vector<Size>>();
				auto indices = at::empty({Size(shape.size()), 0}, at::kLong), values = at::empty({0}, at::kFloat);
				empty = at::sparse_coo_tensor(indices, values, shape, values.options().layout(at::kSparse), true);
			}
			sources.append(py::cast(empty));
		}
	}
	for (size_t i = 0; i < inputs.size(); ++i) {
		updates.append(packets[i].cast<py::tuple>()[1]);
		mappings.append(py::cast(at::tensor(inputs[i].sources, at::TensorOptions().dtype(at::kLong))));
	}
	pending.outputs = hostMerge(sources, updates, mappings, target).cast<py::list>();
}
py::tuple merge(bool enabled, py::dict columns, py::dict deferred, py::list packets, py::tuple target, Size sourceAxis, bool packed, bool neurons, py::object cls, py::dtype dtype, Size budget, const std::string& invalid, Size historyRatio, py::object hostMerge) {
	TORCH_CHECK(historyRatio >= 0 && (!historyRatio || (!packed && !hostMerge.is_none())), invalid);
	TORCH_CHECK(enabled && budget > 0 && budget <= std::numeric_limits<int>::max(), invalid);
	py::tuple result = py::make_tuple(py::none(), py::none());
	if (enabled && ieeeHost()) {
		const Geometry geo(target, packed, neurons, sourceAxis, dtype, invalid);
		std::vector<Size> keys;
		const auto inputs = readPackets(packets, geo, keys, invalid);
		auto pending = std::make_shared<Pending>();
		if (geo.neurons)
			collectNeurons(*pending, columns, keys, geo, historyRatio > 0, invalid);
		if (geo.neurons || collectSources(*pending, columns, deferred, keys, geo, target, packed, cls, dtype, invalid)) {
			TransferPlan plan;
			if (preferHost(pending->sources, inputs, historyRatio)) {
				mergeHost(*pending, inputs, packets, target, hostMerge);
				result = py::make_tuple(keys, pending);
			} else if (plan.build(inputs, pending->sources, geo, budget, budget, std::numeric_limits<Key>::max(), packed, invalid)) {
				try {
					execute(*pending, plan, geo, target, packed, cls, dtype, invalid);
					result = py::make_tuple(keys, pending);
				} catch (const NonfiniteHistory&) {
					// No source has been published; preserve stock COO semantics.
				}
			}
		}
	}
	return result;
}
at::Tensor gatherCanonical(bool enabled, const at::Tensor& tensor, const at::Tensor& query, Size minEntries, Size maxEntries, Size maxRank, int maxThreads, uintptr_t stream) {
	at::Tensor result;
	if (enabled && tensor.layout() == at::kSparse && tensor.scalar_type() == at::kFloat && !tensor.requires_grad() && tensor.is_coalesced() && !tensor.dense_dim() && tensor.dim() > 0 && tensor.dim() <= maxRank && tensor._nnz() >= minEntries && tensor._nnz() <= maxEntries && query.layout() == at::kStrided && query.scalar_type() == at::kLong && query.dim() == 2 && query.size(0) == tensor.dim() && query.size(1) <= maxEntries && tensor.device() == query.device() && (tensor.is_cpu() || tensor.is_cuda())) {
		auto indices = tensor.indices(), values = tensor.values();
		if (!indices.is_neg() && !values.is_neg() && !query.is_neg()) {
			const auto m = query.size(1), n = tensor._nnz();
			c10::DeviceGuard guard(tensor.device());
			auto output = at::empty({m + int(tensor.is_cuda())}, values.options());
			int invalid = 0;
			if (tensor.is_cuda()) {
				checkedCuda(coo_gather_device(indices.data_ptr<Size>(), indices.stride(0), indices.stride(1), values.data_ptr<float>(), values.stride(0), n, query.data_ptr<Size>(), query.stride(0), query.stride(1), m, tensor.sizes().data(), tensor.dim(), output.data_ptr<float>(), &invalid, reinterpret_cast<void*>(stream)));
				output = output.narrow(0, 0, m);
			} else {
				invalid = coo_gather_host(indices.data_ptr<Size>(), indices.stride(0), indices.stride(1), values.data_ptr<float>(), values.stride(0), n, query.data_ptr<Size>(), query.stride(0), query.stride(1), m, tensor.sizes().data(), tensor.dim(), output.data_ptr<float>(), std::min(at::get_num_threads(), maxThreads));
			}
			if (!invalid)
				result = std::move(output);
		}
	}
	return result;
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
	bindPending(module);
	module.def("merge", &merge);
	module.def("gather", &gatherCanonical, py::call_guard<py::gil_scoped_release>());
}
