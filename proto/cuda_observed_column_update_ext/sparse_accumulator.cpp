#include <torch/extension.h>

#include <stdexcept>
#include <string>

void build_sparse_hash_cuda(torch::Tensor targetLinearKey, torch::Tensor hashKeys, torch::Tensor hashSlots, int64_t maxProbe);
void accumulate_sparse_updates_cuda(torch::Tensor targetValues, torch::Tensor updateLinearKey, torch::Tensor updateValues, torch::Tensor hashKeys, torch::Tensor hashSlots, torch::Tensor overflowKeys, torch::Tensor overflowValues, torch::Tensor overflowCount, torch::Tensor hashHits, int64_t maxProbe);

static void validate_cuda_tensor(const torch::Tensor& tensor, const std::string& tensorName) {
	if(!tensor.defined()) {
		throw std::runtime_error("validate_cuda_tensor error: undefined tensor '" + tensorName + "'");
	}
	if(!tensor.is_cuda()) {
		throw std::runtime_error("validate_cuda_tensor error: expected CUDA tensor for '" + tensorName + "'");
	}
	return;
}

static void validate_long_vector(const torch::Tensor& tensor, const std::string& tensorName) {
	validate_cuda_tensor(tensor, tensorName);
	if(tensor.scalar_type() != torch::kInt64) {
		throw std::runtime_error("validate_long_vector error: expected int64 tensor for '" + tensorName + "'");
	}
	if(tensor.dim() != 1) {
		throw std::runtime_error("validate_long_vector error: expected 1D tensor for '" + tensorName + "'");
	}
	return;
}

void build_sparse_hash(torch::Tensor targetLinearKey, torch::Tensor hashKeys, torch::Tensor hashSlots, int64_t maxProbe) {
	validate_long_vector(targetLinearKey, "targetLinearKey");
	validate_long_vector(hashKeys, "hashKeys");
	validate_long_vector(hashSlots, "hashSlots");
	if(hashKeys.numel() != hashSlots.numel()) {
		throw std::runtime_error("build_sparse_hash error: hashKeys and hashSlots must have identical length");
	}
	if(maxProbe <= 0) {
		throw std::runtime_error("build_sparse_hash error: maxProbe must be > 0");
	}
	build_sparse_hash_cuda(targetLinearKey, hashKeys, hashSlots, maxProbe);
	return;
}

void accumulate_sparse_updates(torch::Tensor targetValues, torch::Tensor updateLinearKey, torch::Tensor updateValues, torch::Tensor hashKeys, torch::Tensor hashSlots, torch::Tensor overflowKeys, torch::Tensor overflowValues, torch::Tensor overflowCount, torch::Tensor hashHits, int64_t maxProbe) {
	validate_cuda_tensor(targetValues, "targetValues");
	validate_long_vector(updateLinearKey, "updateLinearKey");
	validate_cuda_tensor(updateValues, "updateValues");
	validate_long_vector(hashKeys, "hashKeys");
	validate_long_vector(hashSlots, "hashSlots");
	validate_long_vector(overflowKeys, "overflowKeys");
	validate_cuda_tensor(overflowValues, "overflowValues");
	validate_long_vector(overflowCount, "overflowCount");
	validate_long_vector(hashHits, "hashHits");
	if(updateValues.dim() != 1) {
		throw std::runtime_error("accumulate_sparse_updates error: updateValues must be 1D");
	}
	if(overflowValues.dim() != 1) {
		throw std::runtime_error("accumulate_sparse_updates error: overflowValues must be 1D");
	}
	if(overflowCount.numel() != 1) {
		throw std::runtime_error("accumulate_sparse_updates error: overflowCount must contain exactly one value");
	}
	if(hashHits.numel() != 1) {
		throw std::runtime_error("accumulate_sparse_updates error: hashHits must contain exactly one value");
	}
	if(updateLinearKey.numel() != updateValues.numel()) {
		throw std::runtime_error("accumulate_sparse_updates error: updateLinearKey and updateValues must have the same number of elements");
	}
	if(hashKeys.numel() != hashSlots.numel()) {
		throw std::runtime_error("accumulate_sparse_updates error: hashKeys and hashSlots must have identical length");
	}
	if(overflowKeys.numel() != overflowValues.numel()) {
		throw std::runtime_error("accumulate_sparse_updates error: overflowKeys and overflowValues must have identical length");
	}
	if(maxProbe <= 0) {
		throw std::runtime_error("accumulate_sparse_updates error: maxProbe must be > 0");
	}
	accumulate_sparse_updates_cuda(targetValues, updateLinearKey, updateValues, hashKeys, hashSlots, overflowKeys, overflowValues, overflowCount, hashHits, maxProbe);
	return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("build_sparse_hash", &build_sparse_hash, "Build sparse hash map (CUDA)");
	m.def("accumulate_sparse_updates", &accumulate_sparse_updates, "Accumulate sparse updates (CUDA)");
}
