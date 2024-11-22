"""GIAANNproto_sparseTensors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive sparse Tensors

"""

import torch as pt

from GIAANNproto_globalDefs import *

def createEmptySparseTensor(shape):
	sparse_zero_tensor = pt.sparse_coo_tensor(indices=pt.empty((len(shape), 0), dtype=pt.long), values=pt.empty(0), size=shape)
	return sparse_zero_tensor

def modify_sparse_tensor(sparse_tensor, indices_to_update, new_value):
	sparse_tensor = sparse_tensor.coalesce()
	
	# Transpose indices_to_update to match dimensions
	indices_to_update = indices_to_update.t()  # Shape: (3, N)
	
	# Get sparse tensor indices
	sparse_indices = sparse_tensor.indices()   # Shape: (3, nnz)
	
	# Expand dimensions to enable broadcasting
	sparse_indices_expanded = sparse_indices.unsqueeze(2)	   # Shape: (3, nnz, 1)
	indices_to_update_expanded = indices_to_update.unsqueeze(1) # Shape: (3, 1, N)
	
	# Compare indices
	matches = (sparse_indices_expanded == indices_to_update_expanded).all(dim=0)  # Shape: (nnz, N)
	
	# Identify matches
	match_mask = matches.any(dim=1)  # Shape: (nnz,)
	
	# Update the values at the matched indices
	sparse_tensor.values()[match_mask] = new_value
	
	return sparse_tensor

	
def merge_tensor_slices_sum(original_sparse_tensor, sparse_slices, d):
	# Extract indices and values from the original tensor
	original_indices = original_sparse_tensor._indices()
	original_values = original_sparse_tensor._values()

	# Prepare lists for new indices and values
	all_indices = [original_indices]
	all_values = [original_values]

	# Process each slice and adjust for the d dimension
	for index, tensor_slice in sparse_slices.items():
		# Create the index tensor for dimension 'd'
		num_nonzero = tensor_slice._indices().size(1)
		d_indices = pt.full((1, num_nonzero), index, dtype=tensor_slice._indices().dtype)

		# Build the new indices by inserting d_indices at position 'd'
		slice_indices = tensor_slice._indices()
		before = slice_indices[:d, :]
		after = slice_indices[d:, :]
		new_indices = pt.cat([before, d_indices, after], dim=0)

		# Collect the adjusted indices and values
		all_indices.append(new_indices)
		all_values.append(tensor_slice._values())

	# Concatenate all indices and values, including the original tensor's
	final_indices = pt.cat(all_indices, dim=1)
	final_values = pt.cat(all_values)

	# Define the final size of the merged tensor, matching the original
	final_size = original_sparse_tensor.size()

	# Create the updated sparse tensor and coalesce to handle duplicates
	merged_sparse_tensor = pt.sparse_coo_tensor(final_indices, final_values, size=final_size)

	merged_sparse_tensor = merged_sparse_tensor.coalesce()
	merged_sparse_tensor.values().clamp_(min=0)

	return merged_sparse_tensor


def slice_sparse_tensor_multi(sparse_tensor, slice_dim, slice_indices):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at given indices,
	without reducing the number of dimensions.

	Args:
		sparse_tensor (pt.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_indices (pt.Tensor): A 1D tensor of indices to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor with the same number of dimensions.
	"""
	import torch

	# Ensure slice_indices is a 1D tensor and sorted
	slice_indices = slice_indices.view(-1).long()
	slice_indices_sorted, _ = pt.sort(slice_indices)

	# Get the indices and values from the sparse tensor
	indices = sparse_tensor.indices()  # Shape: (ndim, nnz)
	values = sparse_tensor.values()	# Shape: (nnz, ...)

	# Get indices along the slicing dimension
	indices_along_dim = indices[slice_dim]  # Shape: (nnz,)

	# Use searchsorted to find positions in slice_indices
	positions = pt.searchsorted(slice_indices_sorted, indices_along_dim)

	# Check if indices_along_dim are actually in slice_indices
	in_bounds = positions < len(slice_indices_sorted)
	matched = in_bounds & (slice_indices_sorted[positions.clamp(max=len(slice_indices_sorted)-1)] == indices_along_dim)

	# Mask to select relevant indices and values
	mask = matched

	# Select the indices and values where mask is True
	selected_indices = indices[:, mask]
	selected_values = values[mask]

	# Adjust indices along slice_dim
	new_indices_along_dim = positions[mask]

	# Update the indices along slice_dim
	selected_indices[slice_dim] = new_indices_along_dim

	# Adjust the size of the tensor
	new_size = list(sparse_tensor.size())
	new_size[slice_dim] = len(slice_indices)

	# Create the new sparse tensor
	new_sparse_tensor = pt.sparse_coo_tensor(selected_indices, selected_values, size=new_size)

	return new_sparse_tensor
	
	

def slice_sparse_tensor(sparse_tensor, slice_dim, slice_index):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparse_tensor (pt.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_index (int): The index at which to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparse_tensor = sparse_tensor.coalesce()	
	
	# Step 1: Extract indices and values
	indices = sparse_tensor._indices()  # Shape: (ndim, nnz)
	values = sparse_tensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match slice_index at slice_dim
	mask = (indices[slice_dim, :] == slice_index)

	# Step 3: Filter indices and values using the mask
	filtered_indices = indices[:, mask]
	filtered_values = values[mask]

	# Step 4: Remove the slice_dim from indices
	new_indices = pt.cat((filtered_indices[:slice_dim, :], filtered_indices[slice_dim+1:, :]), dim=0)

	# Step 5: Adjust the size of the new sparse tensor
	original_size = sparse_tensor.size()
	new_size = original_size[:slice_dim] + original_size[slice_dim+1:]

	# Step 6: Create the new sparse tensor
	new_sparse_tensor = pt.sparse_coo_tensor(new_indices, filtered_values, size=new_size)
	new_sparse_tensor = new_sparse_tensor.coalesce()  # Ensure the tensor is in canonical form

	return new_sparse_tensor

'''
def slice_sparse_tensor(sparse_tensor, slice_dim, slice_index):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparse_tensor (pt.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_index (int): The index at which to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparse_tensor = sparse_tensor.coalesce()

	# Step 1: Extract indices and values
	indices = sparse_tensor._indices()  # Shape: (ndim, nnz)
	values = sparse_tensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match slice_index at slice_dim
	mask = (indices[slice_dim, :] == slice_index)

	# Step 3: Filter indices and values using the mask
	filtered_indices = indices[:, mask]
	filtered_values = values[mask]

	# Step 4: Remove the slice_dim from indices using boolean indexing
	keep_dims = pt.arange(indices.size(0)) != slice_dim
	new_indices = filtered_indices[keep_dims, :]

	# Step 5: Adjust the size of the new sparse tensor
	new_size = list(sparse_tensor.size())
	del new_size[slice_dim]
	new_size = tuple(new_size)

	# Step 6: Create the new sparse tensor
	new_sparse_tensor = pt.sparse_coo_tensor(new_indices, filtered_values, size=new_size)
	# Remove the following line if coalesce is unnecessary
	new_sparse_tensor = new_sparse_tensor.coalesce()

	return new_sparse_tensor
'''
