"""GIAANNcmn_sparseTensors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common predictive sparse Tensors

"""

import torch as pt

from GIAANNcmn_globalDefs import *

def createEmptySparseTensor(shape):
	sparseZeroTensor = pt.sparse_coo_tensor(indices=pt.empty((len(shape), 0), dtype=pt.long), values=pt.empty(0), size=shape, device=deviceSparse)
	return sparseZeroTensor

def buildSparseTensorIndexKeys(indices, size):
	result = None
	if(indices is None):
		raise RuntimeError("buildSparseTensorIndexKeys error: indices is None")
	if(indices.dim() != 2):
		raise RuntimeError("buildSparseTensorIndexKeys error: indices must be rank 2")
	if(len(size) != indices.shape[0]):
		raise RuntimeError("buildSparseTensorIndexKeys error: size rank mismatch")
	sizeTensor = pt.tensor(tuple(size), dtype=pt.long, device=indices.device)
	if(sizeTensor.numel() == 0):
		raise RuntimeError("buildSparseTensorIndexKeys error: size must not be empty")
	if(indices.numel() > 0):
		if(bool(pt.any(indices < 0).item()) or bool(pt.any(indices >= sizeTensor.view(-1, 1)).item())):
			raise RuntimeError("buildSparseTensorIndexKeys error: index out of range")
	stridesCumulative = pt.flip(pt.cumprod(pt.flip(sizeTensor, dims=(0,)), dim=0), dims=(0,))
	strides = pt.cat((stridesCumulative[1:], pt.ones((1,), dtype=pt.long, device=indices.device)), dim=0)
	result = (indices*strides.view(-1, 1)).sum(dim=0)
	return result

def gatherSparseTensorValuesAtIndices(tensor, indices, dtype):
	result = None
	if(indices is None):
		raise RuntimeError("gatherSparseTensorValuesAtIndices error: indices is None")
	if(indices.dim() != 2):
		raise RuntimeError("gatherSparseTensorValuesAtIndices error: indices must be rank 2")
	if(tensor is None):
		result = pt.zeros((indices.shape[1],), dtype=dtype, device=indices.device)
	else:
		if(len(tensor.size()) != indices.shape[0]):
			raise RuntimeError("gatherSparseTensorValuesAtIndices error: tensor/index rank mismatch")
		queryKeys = buildSparseTensorIndexKeys(indices, tensor.size())
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			if(tensor._nnz() == 0):
				result = pt.zeros((indices.shape[1],), dtype=dtype, device=indices.device)
			else:
				tensorKeys = buildSparseTensorIndexKeys(tensor.indices(), tensor.size())
				sortedTensorKeys, sortOrder = pt.sort(tensorKeys)
				sortedTensorValues = tensor.values().index_select(0, sortOrder)
				positions = pt.searchsorted(sortedTensorKeys, queryKeys)
				validMask = positions < sortedTensorKeys.shape[0]
				safePositions = positions.clamp(max=sortedTensorKeys.shape[0]-1)
				matchMask = validMask & (sortedTensorKeys[safePositions] == queryKeys)
				result = pt.zeros((queryKeys.shape[0],), dtype=sortedTensorValues.dtype, device=sortedTensorValues.device)
				if(matchMask.any()):
					result[matchMask] = sortedTensorValues.index_select(0, safePositions[matchMask])
		else:
			flatValues = tensor.reshape(-1)
			result = flatValues.index_select(0, queryKeys.to(flatValues.device))
		if(result.dtype != dtype):
			result = result.to(dtype)
	return result

def maximumSparseTensorValues(targetSparse, updateSparse):
	result = targetSparse
	if(targetSparse is None):
		raise RuntimeError("maximumSparseTensorValues error: targetSparse is None")
	if(updateSparse is None):
		raise RuntimeError("maximumSparseTensorValues error: updateSparse is None")
	if(not targetSparse.is_sparse or not updateSparse.is_sparse):
		raise RuntimeError("maximumSparseTensorValues error: tensors must be sparse")
	targetSparse = targetSparse.coalesce()
	updateSparse = updateSparse.coalesce()
	if(targetSparse.dim() != updateSparse.dim() or tuple(targetSparse.size()) != tuple(updateSparse.size())):
		raise RuntimeError("maximumSparseTensorValues error: tensor size mismatch")
	if(updateSparse._nnz() > 0):
		combinedIndices = pt.cat((targetSparse.indices(), updateSparse.indices()), dim=1)
		combinedValues = pt.cat((targetSparse.values(), updateSparse.values()), dim=0)
		combinedKeys = buildSparseTensorIndexKeys(combinedIndices, targetSparse.size())
		sortedKeys, sortOrder = pt.sort(combinedKeys)
		sortedIndices = combinedIndices.index_select(1, sortOrder)
		sortedValues = combinedValues.index_select(0, sortOrder)
		uniqueKeys, counts = pt.unique_consecutive(sortedKeys, return_counts=True)
		segmentStarts = pt.cumsum(counts, dim=0) - counts
		uniqueIndices = sortedIndices.index_select(1, segmentStarts)
		segmentIds = pt.repeat_interleave(pt.arange(uniqueKeys.shape[0], dtype=pt.long, device=sortedValues.device), counts)
		uniqueValues = pt.empty((uniqueKeys.shape[0],), dtype=sortedValues.dtype, device=sortedValues.device)
		uniqueValues.scatter_reduce_(0, segmentIds, sortedValues, reduce="amax", include_self=False)
		result = pt.sparse_coo_tensor(uniqueIndices, uniqueValues, size=targetSparse.size(), dtype=targetSparse.dtype, device=targetSparse.device).coalesce()
	return result

def subtractValueFromSparseTensorValues(sparseTensor, value):
	sparseTensor = addValueToSparseTensorValues(sparseTensor, -value)
	sparseTensor.values().clamp_(min=0)
	return sparseTensor
	
def addValueToSparseTensorValues(sparseTensor, value):
	sparseTensor = sparseTensor.coalesce()
	sparseTensor = pt.sparse_coo_tensor(sparseTensor.indices(), sparseTensor.values() + value, sparseTensor.size(), device=deviceSparse)
	sparseTensor = sparseTensor.coalesce()
	return sparseTensor

def scaleSparseTensorByBranchValues(sparseTensor, branchValues):
	if(sparseTensor is None):
		return sparseTensor
	if(branchValues is None):
		return sparseTensor
	sparseTensor = sparseTensor.coalesce()
	if(sparseTensor._nnz() == 0):
		return sparseTensor
	indices = sparseTensor.indices()
	values = sparseTensor.values()
	branchValues = branchValues.to(values.device)
	branchIndices = indices[0]
	scaledValues = values * branchValues[branchIndices]
	return pt.sparse_coo_tensor(indices, scaledValues, size=sparseTensor.size(), device=sparseTensor.device).coalesce()

def collapseSparseBranchDimension(sparseTensor):
	if(sparseTensor is None):
		return sparseTensor
	sparseTensor = sparseTensor.coalesce()
	if(sparseTensor._nnz() == 0):
		newSize = sparseTensor.size()[1:]
		emptyIndices = pt.empty((len(newSize), 0), dtype=pt.long, device=sparseTensor.device)
		emptyValues = pt.empty((0,), dtype=sparseTensor.dtype, device=sparseTensor.device)
		return pt.sparse_coo_tensor(emptyIndices, emptyValues, size=newSize, device=sparseTensor.device)
	indices = sparseTensor.indices()
	values = sparseTensor.values()
	if(indices.shape[0] <= 1):
		return sparseTensor
	newIndices = indices[1:]
	newSize = sparseTensor.size()[1:]
	return pt.sparse_coo_tensor(newIndices, values, size=newSize, device=sparseTensor.device).coalesce()

def reduceSparseBranchMax(sparseTensor):
	result = None
	sparseTensor = sparseTensor.coalesce()
	indices = sparseTensor.indices()
	values = sparseTensor.values()
	newSize = sparseTensor.size()[1:]
	if(indices.numel() == 0):
		emptyIndices = pt.empty((len(newSize), 0), dtype=pt.long, device=sparseTensor.device)
		emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
		result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=newSize, device=sparseTensor.device)
	else:
		indicesTail = indices[1:]
		device = indicesTail.device
		strides = pt.ones((len(newSize),), dtype=pt.long, device=device)
		for i in range(len(newSize)-2, -1, -1):
			strides[i] = strides[i+1] * newSize[i+1]
		keys = (indicesTail * strides.unsqueeze(1)).sum(dim=0)
		uniqueKeys, inverseIndices = pt.unique(keys, sorted=True, return_inverse=True)
		if(uniqueKeys.numel() > 0):
			newValues = pt.empty((uniqueKeys.shape[0],), dtype=values.dtype, device=values.device)
			newValues.scatter_reduce_(0, inverseIndices, values, reduce="amax", include_self=False)
			newIndicesList = []
			for i in range(len(newSize)):
				newIndicesList.append(pt.remainder(pt.div(uniqueKeys, strides[i], rounding_mode="floor"), int(newSize[i])))
			newIndices = pt.stack(newIndicesList, dim=0)
			result = pt.sparse_coo_tensor(newIndices, newValues, size=newSize, device=sparseTensor.device)
		else:
			emptyIndices = pt.empty((len(newSize), 0), dtype=pt.long, device=sparseTensor.device)
			emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
			result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=newSize, device=sparseTensor.device)
	result = result.coalesce()
	return result


#replace or multiply element(s) at index with/by new_value
#preconditions: if replace (ie !multiply), assume that the sparse array contains non-zero values at indices_to_update
def modifySparseTensor(sparseTensor, indicesToUpdate, newValue, multiply=False):
	indicesToUpdate = indicesToUpdate.to(sparseTensor.device)
	
	sparseTensor = sparseTensor.coalesce()
	
	# Transpose indicesToUpdate to match dimensions
	indicesToUpdate = indicesToUpdate.t()  # Shape: (batch_size, N)
	
	# Get sparse tensor indices
	sparseIndices = sparseTensor.indices()   # Shape: (batch_size, nnz)
	
	# Expand dimensions to enable broadcasting
	sparseIndicesExpanded = sparseIndices.unsqueeze(2)	   # Shape: (batch_size, nnz, 1)
	indicesToUpdateExpanded = indicesToUpdate.unsqueeze(1) # Shape: (batch_size, 1, N)
	
	# Compare indices
	matches = (sparseIndicesExpanded == indicesToUpdateExpanded).all(dim=0)  # Shape: (nnz, N)
	
	# Identify matches
	matchMask = matches.any(dim=1)  # Shape: (nnz,)
	
	# Update the values at the matched indices
	if(multiply):
		sparseTensor.values()[matchMask] *= newValue
	else:	#replace
		sparseTensor.values()[matchMask] = newValue
		
	return sparseTensor

	
def mergeTensorSlicesSum(originalSparseTensor, sparseSlices, d):
	# Extract indices and values from the original tensor
	originalIndices = originalSparseTensor._indices()
	originalValues = originalSparseTensor._values()

	# Prepare lists for new indices and values
	allIndices = [originalIndices]
	allValues = [originalValues]

	# Process each slice and adjust for the d dimension
	for index, tensorSlice in sparseSlices.items():
		# Create the index tensor for dimension 'd'
		numNonzero = tensorSlice._indices().size(1)
		dIndices = pt.full((1, numNonzero), index, dtype=tensorSlice._indices().dtype, device=deviceSparse)

		# Build the new indices by inserting dIndices at position 'd'
		sliceIndices = tensorSlice._indices()
		before = sliceIndices[:d, :]
		after = sliceIndices[d:, :]
		newIndices = pt.cat([before, dIndices, after], dim=0)

		# Collect the adjusted indices and values
		allIndices.append(newIndices)
		allValues.append(tensorSlice._values())

	# Concatenate all indices and values, including the original tensor's
	finalIndices = pt.cat(allIndices, dim=1)
	finalValues = pt.cat(allValues)

	# Define the final size of the merged tensor, matching the original
	finalSize = originalSparseTensor.size()

	# Create the updated sparse tensor and coalesce to handle duplicates
	mergedSparseTensor = pt.sparse_coo_tensor(finalIndices, finalValues, size=finalSize, device=deviceSparse)

	mergedSparseTensor = mergedSparseTensor.coalesce()
	mergedSparseTensor.values().clamp_(min=0)

	return mergedSparseTensor


def sliceSparseTensorMulti(sparseTensor, sliceDim, sliceIndices):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at given indices,
	without reducing the number of dimensions.

	Args:
		sparseTensor (pt.sparse.FloatTensor): The input sparse tensor.
		sliceDim (int): The dimension along which to slice.
		sliceIndices (pt.Tensor): A 1D tensor of indices to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor with the same number of dimensions.
	"""
	import torch

	sparseTensor = sparseTensor.coalesce()
	
	# Ensure sliceIndices is a 1D tensor and sorted
	sliceIndices = sliceIndices.view(-1).long()
	sliceIndicesSorted, _ = pt.sort(sliceIndices)

	# Get the indices and values from the sparse tensor
	indices = sparseTensor.indices()  # Shape: (ndim, nnz)
	values = sparseTensor.values()	# Shape: (nnz, ...)

	# Get indices along the slicing dimension
	indicesAlongDim = indices[sliceDim]  # Shape: (nnz,)

	# Use searchsorted to find positions in sliceIndices
	positions = pt.searchsorted(sliceIndicesSorted, indicesAlongDim)

	# Check if indicesAlongDim are actually in sliceIndices
	inBounds = positions < len(sliceIndicesSorted)
	matched = inBounds & (sliceIndicesSorted[positions.clamp(max=len(sliceIndicesSorted)-1)] == indicesAlongDim)

	# Mask to select relevant indices and values
	mask = matched

	# Select the indices and values where mask is True
	selectedIndices = indices[:, mask]
	selectedValues = values[mask]

	# Adjust indices along sliceDim
	newIndicesAlongDim = positions[mask]

	# Update the indices along sliceDim
	selectedIndices[sliceDim] = newIndicesAlongDim

	# Adjust the size of the tensor
	newSize = list(sparseTensor.size())
	newSize[sliceDim] = len(sliceIndices)

	# Create the new sparse tensor
	newSparseTensor = pt.sparse_coo_tensor(selectedIndices, selectedValues, size=newSize, device=deviceSparse)

	return newSparseTensor
	
	

def sliceSparseTensor(sparseTensor, sliceDim, sliceIndex):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparseTensor (pt.sparse.FloatTensor): The input sparse tensor.
		sliceDim (int): The dimension along which to slice.
		sliceIndex (int): The index at which to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparseTensor = sparseTensor.coalesce()	
	
	# Step 1: Extract indices and values
	indices = sparseTensor._indices()  # Shape: (ndim, nnz)
	values = sparseTensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match sliceIndex at sliceDim
	mask = (indices[sliceDim, :] == sliceIndex)

	# Step 3: Filter indices and values using the mask
	filteredIndices = indices[:, mask]
	filteredValues = values[mask]

	# Step 4: Remove the sliceDim from indices
	newIndices = pt.cat((filteredIndices[:sliceDim, :], filteredIndices[sliceDim+1:, :]), dim=0)

	# Step 5: Adjust the size of the new sparse tensor
	originalSize = sparseTensor.size()
	newSize = originalSize[:sliceDim] + originalSize[sliceDim+1:]

	# Step 6: Create the new sparse tensor
	newSparseTensor = pt.sparse_coo_tensor(newIndices, filteredValues, size=newSize, device=deviceSparse)
	newSparseTensor = newSparseTensor.coalesce()  # Ensure the tensor is in canonical form

	return newSparseTensor

def addSparseTensorToFirstDimIndex(A, B, index):

	# Assume A is a sparse 4D tensor and B is a sparse 3D tensor
	AIndices = A._indices()
	AValues = A._values()

	# Step 1: Create a mask for entries where the first dimension index is index
	mask = (AIndices[0] == index)

	# Step 2: Extract indices and values for A[index]
	AIndexIndices = AIndices[1:, mask]
	AIndexValues = AValues[mask]

	# Step 3: Create sparse tensor A[index]
	AIndex = pt.sparse_coo_tensor(AIndexIndices, AIndexValues, size=B.shape, device=deviceSparse)

	# Step 4: Perform the addition A[index] + B
	C = AIndex + B

	# Step 5: Adjust indices to include the first dimension index index
	CIndices = pt.cat([pt.full((1, C._indices().shape[1]), index, dtype=pt.long), C._indices()], dim=0)
	CValues = C._values()

	# Step 6: Remove old entries and add new entries to A
	ARemainingIndices = AIndices[:, ~mask]
	ARemainingValues = AValues[~mask]
	ANewIndices = pt.cat([ARemainingIndices, CIndices], dim=1)
	ANewValues = pt.cat([ARemainingValues, CValues], dim=0)

	# Step 7: Update A
	A = pt.sparse_coo_tensor(ANewIndices, ANewValues, size=A.size(), device=deviceSparse)
	
	return A

def replaceAllSparseTensorElementsAtFirstDimIndex(A, B, index):

	# Get indices and values of A
	AIndices = A._indices()
	AValues = A._values()

	# Create a mask to filter out entries where the first index is index
	mask = AIndices[0] != index

	# Keep only the entries where the first index is not index
	AIndicesFiltered = AIndices[:, mask]
	AValuesFiltered = AValues[mask]

	# Get indices and values of B
	BIndices = B._indices()
	BValues = B._values()

	# Adjust B's indices to align with A's dimensions by prepending a row of index's
	BIndicesAdjusted = pt.cat((pt.full((1, BIndices.size(1)), index, dtype=pt.long, device=B.device), BIndices), dim=0)

	# Concatenate the filtered A indices/values with the adjusted B indices/values
	newIndices = pt.cat((AIndicesFiltered, BIndicesAdjusted), dim=1)
	newValues = pt.cat((AValuesFiltered, BValues), dim=0)

	# Create a new sparse tensor with the updated indices and values
	ANew = pt.sparse_coo_tensor(newIndices, newValues, size=A.size(), device=deviceSparse)
	
	return ANew

def sparseAssign(A, B, *indices):
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
	indicesList = list(indices)

	# Validate the number of indices
	if len(indicesList) > A.ndim:
		raise IndexError("Too many indices for tensor of dimension {}".format(A.ndim))

	# Pad indices with empty slices if fewer indices are provided
	if len(indicesList) < A.ndim:
		indicesList.extend([slice(None)] * (A.ndim - len(indicesList)))

	# Process indices to compute the new positions for B's indices in A
	newIndices = []
	for dim, idx in enumerate(indicesList):
		if isinstance(idx, int):
			newIdx = B.indices()[dim] + idx
		elif isinstance(idx, slice):
			start = idx.start or 0
			newIdx = B.indices()[dim] + start
		elif pt.is_tensor(idx):
			idx = idx.to(B.indices().device)
			newIdx = idx[B.indices()[dim]]
		else:
			raise TypeError("Invalid index type: {}".format(type(idx)))
		newIndices.append(newIdx)

	# Stack the new indices
	newIndices = pt.stack(newIndices)

	# Concatenate A's and B's indices and values
	combinedIndices = pt.cat([A.indices(), newIndices], dim=1)
	combinedValues = pt.cat([A.values(), B.values()], dim=0)

	# Create a new sparse tensor with the combined indices and values
	newA = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=A.shape, dtype=A.dtype, device=deviceSparse)

	# Coalesce the tensor to sum duplicate indices
	newA = newA.coalesce()

	return newA

def expandSparseTensor(tensor, q, y, newDimSize=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y.

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2265 x).
		y (int): Index in the new dimension to set.
		newDimSize (int, optional): Size of the new dimension. If None,
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
	yRow = pt.full((1, nnz), y, dtype=indices.dtype, device=indices.device)

	# Insert the new dimension at position q
	newIndices = pt.cat((indices[:q], yRow, indices[q:]), dim=0)

	# Determine the new size
	originalSize = list(tensor.size())

	if newDimSize is None:
		if y >= 0:
			newDimSize = y + 1
		else:
			raise ValueError("Negative index y requires specifying newDimSize.")

	if y >= newDimSize or y < -newDimSize:
		raise ValueError(f"Index y={y} is out of bounds for dimension of size {newDimSize}")

	newSize = originalSize[:q] + [newDimSize] + originalSize[q:]

	# Create the new sparse tensor
	newTensor = pt.sparse_coo_tensor(newIndices, values, size=newSize, device=deviceSparse)

	return newTensor

def expandSparseTensorMulti(tensor, q, y, newDimSize=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y (list or pt.Tensor of indices).

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2265 x).
		y (list or pt.Tensor): Indices in the new dimension to set, length nnz.
		newDimSize (int, optional): Size of the new dimension. If None,
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
	newIndices = pt.cat((indices[:q], y, indices[q:]), dim=0)

	# Determine the new size
	originalSize = list(tensor.size())

	if newDimSize is None:
		newDimSize = int(y.max().item()) + 1

	if y.min().item() < -newDimSize or y.max().item() >= newDimSize:
		raise ValueError(f"Indices in y are out of bounds for dimension of size {newDimSize}.")

	newSize = originalSize[:q] + [newDimSize] + originalSize[q:]

	# Create the new sparse tensor
	newTensor = pt.sparse_coo_tensor(newIndices, values, size=newSize, device=deviceSparse)

	return newTensor

def addElementValueToSparseTensor(spTensor, dimensions, v):

	# Example setup: create a sparse len(dimensions) dimensional tensor
	# spTensor is a sparse_coo_tensor with shape [dimensions]
	# Make sure spTensor is in COO format (coalesce it if necessary)
	spTensor = spTensor.coalesce()

	indices = spTensor._indices()  # Shape: [len(dimensions), nnz]
	values = spTensor._values()	# Shape: [nnz]

	targetIndex = pt.tensor(dimensions, dtype=pt.long, device=spTensor.device).unsqueeze(1)  # shape [len(dimensions), 1]

	# Check if this index already exists in the sparse tensor
	mask = (indices == targetIndex).all(dim=0)  # Boolean mask indicating where the match occurs

	if mask.any():
		# The element already exists.
		idx = mask.nonzero().item()  # Extract that single index
		values[idx] += v
	else:
		# The element does not exist; we need to add it.
		newIndices = pt.cat([indices, targetIndex], dim=1)
		newValues = pt.cat([values, pt.tensor([v], dtype=values.dtype, device=values.device)])
		spTensor = pt.sparse_coo_tensor(newIndices, newValues, spTensor.shape, device=spTensor.device).coalesce()

	return spTensor

def sparse_rowwise_max(x):
	"""
	Computes max over dim=1 (columns) for a 2D sparse COO tensor 'x'.
	Returns a 1D tensor of size [x.size(0)] with the max values.
	"""

	# Make sure the tensor is in COO format and has unique indices
	x = x.coalesce()

	# Extract COO indices and values
	indices = x._indices()  # shape: [2, nnz]
	values  = x._values()   # shape: [nnz]

	# Row indices (which row each non-zero belongs to)
	row_idx = indices[0]

	# Initialize output with -inf so we can do a max-reduction
	out = pt.full((x.size(0),), float('-inf'), dtype=values.dtype, device=values.device)

	# Use index_reduce_ (available in newer PyTorch versions) to compute max
	out.index_reduce_(dim=0, index=row_idx, source=values, reduce="amax")

	return out

def selectAindicesContainedInB(A, B):
	A = A.coalesce()
	B = B.coalesce()
	A_indices = A.indices()
	A_values = A.values()
	B_indices = B.indices()
	if(A.size() != B.size()):
		raise RuntimeError("selectAindicesContainedInB error: tensor sizes do not match")
	if(A_indices.shape[0] != B_indices.shape[0]):
		raise RuntimeError("selectAindicesContainedInB error: tensor ranks do not match")
	if(A_indices.numel() == 0 or B_indices.numel() == 0):
		A_indices_in_B = pt.empty((A_indices.shape[0], 0), dtype=A_indices.dtype, device=A_indices.device)
		A_values_in_B = pt.empty((0,), dtype=A_values.dtype, device=A_values.device)
	else:
		A_keys = buildSparseTensorIndexKeys(A_indices, A.size())
		B_keys = buildSparseTensorIndexKeys(B_indices, B.size())
		B_keys = pt.sort(pt.unique(B_keys)).values
		positions = pt.searchsorted(B_keys, A_keys)
		valid = positions < B_keys.shape[0]
		safePositions = positions.clamp(max=B_keys.shape[0]-1)
		mask = valid & (B_keys[safePositions] == A_keys)
		A_indices_in_B = A_indices[:, mask]
		A_values_in_B = A_values[mask]
	result = pt.sparse_coo_tensor(A_indices_in_B, A_values_in_B, size=A.shape, device=A.device)
	return result

def buildSparseTensorIndexKeys(indices, size):
	result = None
	if(indices.dim() != 2):
		raise RuntimeError("buildSparseTensorIndexKeys error: indices must be rank 2")
	if(len(size) != indices.shape[0]):
		raise RuntimeError("buildSparseTensorIndexKeys error: index rank does not match tensor size")
	strides = pt.ones((len(size),), dtype=pt.long, device=indices.device)
	for i in range(len(size)-2, -1, -1):
		strides[i] = strides[i+1] * int(size[i+1])
	result = (indices * strides.unsqueeze(1)).sum(dim=0)
	return result

def selectAindicesContainedInBBinaryTree(A, B):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(not A.is_sparse or not B.is_sparse):
			raise RuntimeError("selectAindicesContainedInBBinaryTree error: A and B must be sparse")
		A = A.coalesce()
		B = B.coalesce()
		if(A.size() != B.size()):
			raise RuntimeError("selectAindicesContainedInBBinaryTree error: A and B sizes must match")
		if(A.dim() != 3):
			raise RuntimeError("selectAindicesContainedInBBinaryTree error: A and B must have three dimensions")
		if(A._nnz() == arrayIndexSegmentFirst or B._nnz() == arrayIndexSegmentFirst):
			emptyIndices = pt.empty((A.dim(), arrayIndexSegmentFirst), dtype=pt.long, device=A.device)
			emptyValues = pt.empty((arrayIndexSegmentFirst,), dtype=A.dtype, device=A.device)
			result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=A.size(), dtype=A.dtype, device=A.device).coalesce()
		else:
			AlinearIndices = calculateBinaryTreeSparseTensorLinearIndices(A.indices(), A.size())
			BlinearIndices = calculateBinaryTreeSparseTensorLinearIndices(B.indices(), B.size())
			sortedBlinearIndices = pt.sort(BlinearIndices).values
			positions = pt.searchsorted(sortedBlinearIndices, AlinearIndices)
			validPositions = positions < sortedBlinearIndices.shape[0]
			safePositions = positions.clamp(max=sortedBlinearIndices.shape[0]-1)
			containedMask = validPositions & (sortedBlinearIndices[safePositions] == AlinearIndices)
			result = pt.sparse_coo_tensor(A.indices()[:, containedMask], A.values()[containedMask], size=A.size(), dtype=A.dtype, device=A.device).coalesce()
	else:
		raise RuntimeError("selectAindicesContainedInBBinaryTree error: requires multipleDendriticBranchesBinaryTree")
	return result

def calculateBinaryTreeSparseTensorLinearIndices(indices, size):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(indices.dim() != 2):
			raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: indices must have two dimensions")
		if(len(size) != indices.shape[0]):
			raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: indices and size dimensions must match")
		if(len(size) != 3):
			raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: binary tree neuron activations must have three dimensions")
		maximumLinearIndex = multipleDendriticBranchesBinaryTreeBranchingFactor**arrayIndexSegmentFirst
		for dimensionSize in size:
			if(dimensionSize <= arrayIndexSegmentFirst):
				raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: size dimensions must be positive")
			maximumLinearIndex = maximumLinearIndex*dimensionSize
		if(maximumLinearIndex > pt.iinfo(pt.long).max):
			raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: sparse tensor shape exceeds supported linear index range")
		result = indices[arrayIndexSegmentFirst].clone()
		for dimensionIndex in range(arrayIndexSegmentFirst+1, indices.shape[0]):
			result = result*size[dimensionIndex] + indices[dimensionIndex]
	else:
		raise RuntimeError("calculateBinaryTreeSparseTensorLinearIndices error: requires multipleDendriticBranchesBinaryTree")
	return result

def neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod):
	hasBranchDim = globalFeatureNeuronsActivation.dim() == 4
	isSparse = globalFeatureNeuronsActivation.is_sparse
	if(multipleDendriticBranchesBinaryTree):
		featureNeuronsActive = neuronActivationSparseBinaryTree(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	else:
		# Sparse tensors cannot be sliced with native indexing on CPU; use sliceSparseTensor.
		def sliceSegment(tensor, segmentIndex):
			if(hasBranchDim):
				if(isSparse):
					return sliceSparseTensor(tensor, 1, segmentIndex)
				return tensor[:, segmentIndex]
			if(isSparse):
				return sliceSparseTensor(tensor, 0, segmentIndex)
			return tensor[segmentIndex]
		if(useSANI):
			if(algorithmMatrixSANImethod=="doNotEnforceActivationAcrossSegments"):
				if(hasBranchDim):
					featureNeuronsActive = globalFeatureNeuronsActivation.sum(dim=1) 	#sum activations across all segments
				else:
					featureNeuronsActive = globalFeatureNeuronsActivation.sum(dim=0) 	#sum activations across all segments
			elif(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
				if(hasBranchDim):
					featureNeuronsActive = globalFeatureNeuronsActivation.sum(dim=1) 	#sum activations across all segments
				else:
					featureNeuronsActive = globalFeatureNeuronsActivation.sum(dim=0) 	#sum activations across all segments
				if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
					lastSegmentConstraint = arrayIndexSegmentAdjacentColumn	#ignore internal column activation requirement
				else:
					lastSegmentConstraint = arrayIndexSegmentLast
				if(algorithmMatrixSANIenforceRequirement=="enforceAnySegmentMustBeActive"):
					pass
				elif(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
					# Only require that the last constraint segment (ie adjacent column or adjacent feature) is active; the internal column segment is ignored
					lastConstraintSegmentActive = sliceSegment(globalFeatureNeuronsActivation, lastSegmentConstraint)
					featureNeuronsActive = selectAindicesContainedInB(featureNeuronsActive, lastConstraintSegmentActive)
				elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
					for s in calculateAllSegmentConstraintIndexRange(lastSegmentConstraint):	#ignore internal column activation requirement
						featureNeuronsActive = selectAindicesContainedInB(featureNeuronsActive, sliceSegment(globalFeatureNeuronsActivation, s))
		else:
			#select last (most proximal) segment activation
			featureNeuronsActive = sliceSegment(globalFeatureNeuronsActivation, arrayIndexSegmentLast)
	return featureNeuronsActive	

def neuronActivationSparseBinaryTree(globalFeatureNeuronsActivation, algorithmMatrixSANImethod):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(not multipleDendriticBranches):
			raise RuntimeError("neuronActivationSparseBinaryTree error: multipleDendriticBranches is required")
		if(globalFeatureNeuronsActivation.dim() < 3):
			raise RuntimeError("neuronActivationSparseBinaryTree error: activation tensor must include branch and segment dimensions")
		if(globalFeatureNeuronsActivation.shape[0] != multipleDendriticBranchesNumber or globalFeatureNeuronsActivation.shape[1] != arrayNumberOfSegments):
			raise RuntimeError("neuronActivationSparseBinaryTree error: activation tensor tree dimensions are invalid")
		if(multipleDendriticBranchesBinaryTreeDepth != arrayNumberOfSegments):
			raise RuntimeError("neuronActivationSparseBinaryTree error: binary tree depth must equal arrayNumberOfSegments")
		wasSparse = globalFeatureNeuronsActivation.is_sparse
		activationSparse = globalFeatureNeuronsActivation.coalesce() if wasSparse else globalFeatureNeuronsActivation.to_sparse_coo().coalesce()
		segmentActivations = []
		for segmentIndex in range(arrayNumberOfSegments):
			segmentActivations.append(projectBinaryTreeSegmentActivation(activationSparse, segmentIndex))
		combinedIndices = pt.empty((activationSparse.dim()-1, 0), dtype=pt.long, device=activationSparse.device)
		combinedValues = pt.empty((0,), dtype=activationSparse.dtype, device=activationSparse.device)
		if(len(segmentActivations) > 0):
			combinedIndices = pt.cat([segmentActivation.indices() for segmentActivation in segmentActivations], dim=1)
			combinedValues = pt.cat([segmentActivation.values() for segmentActivation in segmentActivations], dim=0)
		featureNeuronsActive = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=(multipleDendriticBranchesNumber, *activationSparse.size()[2:]), dtype=activationSparse.dtype, device=activationSparse.device).coalesce()
		if(useSANI):
			if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
				if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
					lastSegmentConstraint = arrayIndexSegmentAdjacentColumn
				else:
					lastSegmentConstraint = arrayIndexSegmentLast
				if(lastSegmentConstraint < arrayIndexSegmentFirst or lastSegmentConstraint >= arrayNumberOfSegments):
					raise RuntimeError("neuronActivationSparseBinaryTree error: lastSegmentConstraint out of range")
				if(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
					featureNeuronsActive = selectAindicesContainedInBBinaryTree(featureNeuronsActive, segmentActivations[lastSegmentConstraint])
				elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
					for segmentIndex in calculateAllSegmentConstraintIndexRange(lastSegmentConstraint):
						featureNeuronsActive = selectAindicesContainedInBBinaryTree(featureNeuronsActive, segmentActivations[segmentIndex])
				elif(algorithmMatrixSANIenforceRequirement!="enforceAnySegmentMustBeActive"):
					raise RuntimeError("neuronActivationSparseBinaryTree error: algorithmMatrixSANIenforceRequirement is invalid")
			elif(algorithmMatrixSANImethod!="doNotEnforceActivationAcrossSegments"):
				raise RuntimeError("neuronActivationSparseBinaryTree error: algorithmMatrixSANImethod is invalid")
		else:
			featureNeuronsActive = segmentActivations[arrayIndexSegmentLast]
		result = featureNeuronsActive if wasSparse else featureNeuronsActive.to_dense()
	else:
		raise RuntimeError("neuronActivationSparseBinaryTree error: requires multipleDendriticBranchesBinaryTree")
	return result

def calculateAllSegmentConstraintIndexRange(lastSegmentConstraint):
	result = None
	if(lastSegmentConstraint < arrayIndexSegmentFirst or lastSegmentConstraint >= arrayNumberOfSegments):
		raise RuntimeError("calculateAllSegmentConstraintIndexRange error: lastSegmentConstraint out of range")
	firstSegmentConstraint = arrayIndexSegmentFirst
	if(useSANIfeaturesAndColumns and enforceAllSegmentsMustBeActiveFeatureSegmentsOnly and lastSegmentConstraint >= arrayNumberOfSegmentsColumnDistance):
		if(arrayNumberOfSegmentsColumnDistance <= arrayIndexSegmentFirst or arrayNumberOfSegmentsColumnDistance >= arrayNumberOfSegments):
			raise RuntimeError("calculateAllSegmentConstraintIndexRange error: arrayNumberOfSegmentsColumnDistance out of range")
		firstSegmentConstraint = arrayNumberOfSegmentsColumnDistance
	if(firstSegmentConstraint < arrayIndexSegmentFirst or firstSegmentConstraint > lastSegmentConstraint):
		raise RuntimeError("calculateAllSegmentConstraintIndexRange error: firstSegmentConstraint out of range")
	if(useSANIfeaturesAndColumns and not enforceAllSegmentsMustBeActiveFeatureSegmentsOnly and lastSegmentConstraint >= arrayNumberOfSegmentsColumnDistance):
		if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= arrayNumberOfSegmentsColumnDistance):
			raise RuntimeError("calculateAllSegmentConstraintIndexRange error: arrayIndexSegmentInternalColumn out of range")
		result = []
		for segmentIndex in range(firstSegmentConstraint, lastSegmentConstraint+1):
			if(segmentIndex != arrayIndexSegmentInternalColumn):
				result.append(segmentIndex)
	else:
		result = range(firstSegmentConstraint, lastSegmentConstraint+1)
	return result

def requiresLastSegmentConnectionConstraint():
	result = False
	if(not useSANI):
		result = False
	elif(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
		if(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
			result = True
		elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
			if(enforceSequentialActivation and useSANIfeaturesAndColumns and enforceSequentialActivationFeatureSegmentsOnly and enforceAllSegmentsMustBeActiveFeatureSegmentsOnly):
				result = True
		elif(algorithmMatrixSANIenforceRequirement=="enforceAnySegmentMustBeActive"):
			result = False
		else:
			raise RuntimeError("requiresLastSegmentConnectionConstraint error: algorithmMatrixSANIenforceRequirement is invalid")
	elif(algorithmMatrixSANImethod=="doNotEnforceActivationAcrossSegments"):
		result = False
	else:
		raise RuntimeError("requiresLastSegmentConnectionConstraint error: algorithmMatrixSANImethod is invalid")
	return result

def projectBinaryTreeSegmentActivation(activationSparse, segmentIndex):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(not activationSparse.is_sparse or not activationSparse.is_coalesced()):
			raise RuntimeError("projectBinaryTreeSegmentActivation error: activationSparse must be a coalesced sparse tensor")
		if(segmentIndex < arrayIndexSegmentFirst or segmentIndex >= arrayNumberOfSegments):
			raise RuntimeError("projectBinaryTreeSegmentActivation error: segmentIndex out of range")
		rootBranchesPerSegmentBranch = multipleDendriticBranchesBinaryTreeBranchingFactor**segmentIndex
		segmentBranchCount = multipleDendriticBranchesNumber//rootBranchesPerSegmentBranch
		if(segmentBranchCount < 1 or multipleDendriticBranchesNumber % rootBranchesPerSegmentBranch != arrayIndexSegmentFirst):
			raise RuntimeError("projectBinaryTreeSegmentActivation error: binary tree branch dimensions are invalid")
		activationIndices = activationSparse.indices()
		activationValues = activationSparse.values()
		segmentMask = activationIndices[1] == segmentIndex
		segmentIndices = activationIndices[:, segmentMask]
		segmentValues = activationValues[segmentMask]
		resultSize = (multipleDendriticBranchesNumber, *activationSparse.size()[2:])
		if(segmentIndices.shape[1] == arrayIndexSegmentFirst):
			emptyIndices = pt.empty((len(resultSize), 0), dtype=pt.long, device=activationSparse.device)
			emptyValues = pt.empty((0,), dtype=activationSparse.dtype, device=activationSparse.device)
			result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=resultSize, dtype=activationSparse.dtype, device=activationSparse.device).coalesce()
		else:
			if(bool(pt.any(segmentIndices[0] >= segmentBranchCount).item())):
				raise RuntimeError("projectBinaryTreeSegmentActivation error: active branch index is invalid for segmentIndex")
			rootBranchOffsets = pt.arange(rootBranchesPerSegmentBranch, dtype=pt.long, device=activationSparse.device)
			rootBranchIndices = (segmentIndices[0].unsqueeze(1)*rootBranchesPerSegmentBranch + rootBranchOffsets.unsqueeze(0)).reshape(-1)
			repeatedTailIndices = segmentIndices[2:].repeat_interleave(rootBranchesPerSegmentBranch, dim=1)
			expandedIndices = pt.cat((rootBranchIndices.unsqueeze(0), repeatedTailIndices), dim=0)
			expandedValues = segmentValues.repeat_interleave(rootBranchesPerSegmentBranch)
			result = pt.sparse_coo_tensor(expandedIndices, expandedValues, size=resultSize, dtype=activationSparse.dtype, device=activationSparse.device).coalesce()
	else:
		raise RuntimeError("projectBinaryTreeSegmentActivation error: requires multipleDendriticBranchesBinaryTree")
	return result
			
def insertSequenceObservedColumnIntoObservedColumnFeatures(self, cIdx, fIdxTensor, featureIndicesInObserved, featureNeuronsSparse, observedColumn, storeDatabaseGlobalFeatureNeuronsInRam=False):
	# feature neurons
	indices = featureNeuronsSparse.indices()
	values = featureNeuronsSparse.values()
	#if(indices.shape[1] > 0):
	mask = (indices[3] == cIdx) & pt.isin(indices[4], fIdxTensor)
	#if pt.any(mask):
	filteredIndices = indices[:, mask]
	filteredValues = values[mask]
	filteredIndices = pt.stack((
		filteredIndices[0],	# property index
		filteredIndices[1],	# branch index
		filteredIndices[2],	# segment index
		filteredIndices[4],	# feature index
	), dim=0)
	if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
		filteredIndices[3] = featureIndicesInObserved[filteredIndices[3]]
	if not storeDatabaseGlobalFeatureNeuronsInRam:
		observedColumn.featureNeurons = observedColumn.featureNeurons + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=observedColumn.featureNeurons.size(), dtype=arrayType, device=deviceSparse)
		observedColumn.featureNeurons = observedColumn.featureNeurons.coalesce()
		observedColumn.featureNeurons.values().clamp_(min=0)
	else:
		self.featureNeuronChanges[cIdx] = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=observedColumn.featureNeurons.size(), dtype=arrayType, device=deviceSparse)

def insertSequenceObservedColumnIntoObservedColumnConnections(self, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsSparse, featureConnections, featureConnectionsOutput=True):
	# feature connections;
	indices = featureConnectionsSparse.indices()
	values = featureConnectionsSparse.values()
	#if(indices.shape[1] > 0):
	if(featureConnectionsOutput):
		mask = (indices[3] == cIdx)
	else:
		mask = (indices[5] == cIdx)
	#if pt.any(mask):
	filteredIndices = indices[:, mask]
	filteredValues = values[mask]
	if(featureConnectionsOutput):
		'''#orig;
		filteredIndices[2] = filteredIndices[3]
		filteredIndices[3] = filteredIndices[4]
		filteredIndices[4] = filteredIndices[5]
		filteredIndices = filteredIndices[0:5]
		'''
		filteredIndices = pt.stack((
			filteredIndices[0],	# property index
			filteredIndices[1],	# branch index
			filteredIndices[2],	# segment index
			filteredIndices[4],	# source feature index
			filteredIndices[5],	# target column index (sequence)
			filteredIndices[6],	# target feature index
		), dim=0)
	else:	#featureConnectionsInput (inhibitory connections only)
		filteredIndices = pt.stack((
			filteredIndices[0],	# property index
			filteredIndices[1],	# branch index
			filteredIndices[2],	# segment index
			filteredIndices[6],	# inhibitory feature index (target)
			filteredIndices[3],	# source column index
			filteredIndices[4],	# source feature index
		), dim=0)
	filteredIndices[4] = self.conceptIndicesInSequenceObservedTensor[filteredIndices[4]]
	if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
		filteredIndices[3] = featureIndicesInObserved[filteredIndices[3]]
		filteredIndices[5] = featureIndicesInObserved[filteredIndices[5]]
	featureConnections = featureConnections + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=featureConnections.size(), dtype=arrayType, device=deviceSparse)
	featureConnections = featureConnections.coalesce()
	featureConnections.values().clamp_(min=0)
	return featureConnections	
