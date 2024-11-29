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

def subtract_value_from_sparse_tensor_values(sparse_tensor, value):
	sparse_tensor = add_value_to_sparse_tensor_values(sparse_tensor, -value)
	sparse_tensor.values().clamp_(min=0)
	return sparse_tensor
	
def add_value_to_sparse_tensor_values(sparse_tensor, value):
	sparse_tensor = sparse_tensor.coalesce()
	sparse_tensor = pt.sparse_coo_tensor(sparse_tensor.indices(), sparse_tensor.values() + value, sparse_tensor.size())
	sparse_tensor = sparse_tensor.coalesce()
	return sparse_tensor

def modify_sparse_tensor(sparse_tensor, indices_to_update, new_value):
	sparse_tensor = sparse_tensor.coalesce()
	
	# Transpose indices_to_update to match dimensions
	indices_to_update = indices_to_update.t()  # Shape: (batch_size, N)
	
	# Get sparse tensor indices
	sparse_indices = sparse_tensor.indices()   # Shape: (batch_size, nnz)
	
	# Expand dimensions to enable broadcasting
	sparse_indices_expanded = sparse_indices.unsqueeze(2)	   # Shape: (batch_size, nnz, 1)
	indices_to_update_expanded = indices_to_update.unsqueeze(1) # Shape: (batch_size, 1, N)
	
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

def addSparseTensorToFirstDimIndex(A, B, index):

	# Assume A is a sparse 4D tensor and B is a sparse 3D tensor
	A_indices = A._indices()
	A_values = A._values()

	# Step 1: Create a mask for entries where the first dimension index is index
	mask = (A_indices[0] == index)

	# Step 2: Extract indices and values for A[index]
	A_index_indices = A_indices[1:, mask]
	A_index_values = A_values[mask]

	# Step 3: Create sparse tensor A[index]
	A_index = pt.sparse_coo_tensor(A_index_indices, A_index_values, size=B.shape)

	# Step 4: Perform the addition A[index] + B
	C = A_index + B

	# Step 5: Adjust indices to include the first dimension index index
	C_indices = pt.cat([pt.full((1, C._indices().shape[1]), index, dtype=pt.long), C._indices()], dim=0)
	C_values = C._values()

	# Step 6: Remove old entries and add new entries to A
	A_remaining_indices = A_indices[:, ~mask]
	A_remaining_values = A_values[~mask]
	A_new_indices = pt.cat([A_remaining_indices, C_indices], dim=1)
	A_new_values = pt.cat([A_remaining_values, C_values], dim=0)

	# Step 7: Update A
	A = pt.sparse_coo_tensor(A_new_indices, A_new_values, size=A.size())
	
	return A

def replaceAllSparseTensorElementsAtFirstDimIndex(A, B, index):

	# Get indices and values of A
	A_indices = A._indices()
	A_values = A._values()

	# Create a mask to filter out entries where the first index is index
	mask = A_indices[0] != index

	# Keep only the entries where the first index is not index
	A_indices_filtered = A_indices[:, mask]
	A_values_filtered = A_values[mask]

	# Get indices and values of B
	B_indices = B._indices()
	B_values = B._values()

	# Adjust B's indices to align with A's dimensions by prepending a row of index's
	B_indices_adjusted = pt.cat((pt.full((1, B_indices.size(1)), index, dtype=pt.long, device=B.device), B_indices), dim=0)

	# Concatenate the filtered A indices/values with the adjusted B indices/values
	new_indices = pt.cat((A_indices_filtered, B_indices_adjusted), dim=1)
	new_values = pt.cat((A_values_filtered, B_values), dim=0)

	# Create a new sparse tensor with the updated indices and values
	A_new = pt.sparse_coo_tensor(new_indices, new_values, size=A.size(), device=A.device)
	
	return A_new

def sparse_assign(A, B, *indices):
	"""
	Assigns sparse tensor B into sparse tensor A at positions specified by indices.

	Args:
		A (pt.sparse.Tensor): The target sparse tensor.
		B (pt.sparse.Tensor): The sparse tensor to assign into A.
		*indices: An arbitrary number of indices (int, slice, or 1D tensor).

	Returns:
		pt.sparse.Tensor: The updated sparse tensor.
	"""
	# Ensure A and B are sparse tensors
	if not A.is_sparse or not B.is_sparse:
		raise ValueError("Both A and B must be sparse tensors.")

	# Convert indices to a list for easier manipulation
	indices_list = list(indices)

	# Validate the number of indices
	if len(indices_list) > A.ndim:
		raise IndexError("Too many indices for tensor of dimension {}".format(A.ndim))

	# Pad indices with empty slices if fewer indices are provided
	if len(indices_list) < A.ndim:
		indices_list.extend([slice(None)] * (A.ndim - len(indices_list)))

	# Process indices to compute the new positions for B's indices in A
	new_indices = []
	for dim, idx in enumerate(indices_list):
		if isinstance(idx, int):
			# If idx is an integer, add it to B's indices in this dimension
			new_idx = B.indices()[dim] + idx
		elif isinstance(idx, slice):
			# If idx is a slice, calculate the start index
			start = idx.start or 0
			new_idx = B.indices()[dim] + start
		elif pt.is_tensor(idx):
			# If idx is a tensor of indices, index into it using B's indices
			idx = idx.to(B.indices().device)
			new_idx = idx[B.indices()[dim]]
		else:
			raise TypeError("Invalid index type: {}".format(type(idx)))
		new_indices.append(new_idx)

	# Stack the new indices
	new_indices = pt.stack(new_indices)

	# Concatenate A's and B's indices and values
	combined_indices = pt.cat([A.indices(), new_indices], dim=1)
	combined_values = pt.cat([A.values(), B.values()], dim=0)

	# Create a new sparse tensor with the combined indices and values
	new_A = pt.sparse_coo_tensor(combined_indices, combined_values, size=A.shape, dtype=A.dtype, device=A.device)

	# Coalesce the tensor to sum duplicate indices
	new_A = new_A.coalesce()

	return new_A

def expand_sparse_tensor(tensor, q, y, new_dim_size=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y.

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2264 x).
		y (int): Index in the new dimension to set.
		new_dim_size (int, optional): Size of the new dimension. If None,
			it defaults to y + 1.

	Returns:
		pt.Tensor: New sparse tensor with x+1 dimensions.
	"""
	if not tensor.is_sparse:
		raise ValueError("Input tensor must be a sparse tensor.")

	indices = tensor.indices()  # Shape: (x, nnz)
	values = tensor.values()	# Shape: (nnz,)

	x, nnz = indices.shape

	# Create a row with the new indices set to y
	y_row = pt.full(
		(1, nnz), y, dtype=indices.dtype, device=indices.device
	)

	# Insert the new dimension at position q
	new_indices = pt.cat((indices[:q], y_row, indices[q:]), dim=0)

	# Determine the new size
	original_size = list(tensor.size())

	if new_dim_size is None:
		if y >= 0:
			new_dim_size = y + 1
		else:
			raise ValueError(
				"Negative index y requires specifying new_dim_size."
			)

	if y >= new_dim_size or y < -new_dim_size:
		raise ValueError(
			f"Index y={y} is out of bounds for dimension of size {new_dim_size}"
		)

	new_size = original_size[:q] + [new_dim_size] + original_size[q:]

	# Create the new sparse tensor
	new_tensor = pt.sparse_coo_tensor(new_indices, values, size=new_size)

	return new_tensor

def expand_sparse_tensor_multi(tensor, q, y, new_dim_size=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y (list or tensor of indices).

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2264 x).
		y (list or pt.Tensor): Indices in the new dimension to set, length nnz.
		new_dim_size (int, optional): Size of the new dimension. If None,
			it defaults to max(y) + 1.

	Returns:
		pt.Tensor: New sparse tensor with x+1 dimensions.
	"""
	if not tensor.is_sparse:
		raise ValueError("Input tensor must be a sparse tensor.")

	indices = tensor.indices()  # Shape: (x, nnz)
	values = tensor.values()	# Shape: (nnz,)

	x, nnz = indices.shape

	# Convert y to a tensor if it's a list
	y = pt.tensor(y, dtype=indices.dtype, device=indices.device).view(1, nnz)

	if y.shape[1] != nnz:
		raise ValueError(f"Length of y ({y.shape[1]}) must be equal to number of non-zero elements (nnz={nnz}).")

	# Insert the new dimension at position q
	new_indices = pt.cat((indices[:q], y, indices[q:]), dim=0)

	# Determine the new size
	original_size = list(tensor.size())

	if new_dim_size is None:
		new_dim_size = int(y.max().item()) + 1

	if y.min().item() < -new_dim_size or y.max().item() >= new_dim_size:
		raise ValueError(f"Indices in y are out of bounds for dimension of size {new_dim_size}.")

	new_size = original_size[:q] + [new_dim_size] + original_size[q:]

	# Create the new sparse tensor
	new_tensor = pt.sparse_coo_tensor(new_indices, values, size=new_size)

	return new_tensor
