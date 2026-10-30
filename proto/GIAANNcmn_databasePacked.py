"""Zero-copy views of resident int32 connection records; COO remains the disk format."""

from GIAANNcmn_globalDefs import *

if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
	import numpy as np


def prepareSources(column, requiredSourceFeatureIndices, targetDevice):
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam or not column.databaseNetworkObject.observedColumnsRAMLoaded or not column.isDatabaseResidentConnectionStorage() or column.databaseNetworkObject.inferenceMode):
			raise RuntimeError(databaseResidentPackedInvalidConfiguration)
		resolvedDevice = pt.device(targetDevice if targetDevice is not None else column.getDefaultConnectionTargetDevice())
		if(resolvedDevice.type != databaseResidentPackedDeviceType or pt.device(deviceDatabase).type != databaseResidentPackedDeviceType):
			raise RuntimeError(databaseResidentPackedInvalidConfiguration)
		sourceIndices = column.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		if(sourceIndices is None or any(type(index) is not int or index < databaseResidentPackedZero or index >= column.databaseNetworkObject.f for index in sourceIndices)):
			raise RuntimeError(databaseResidentPackedInvalidInput)
		targetSize = column.getFeatureConnectionsTargetSize()
		validateShape(targetSize)
		for sourceIndex in sourceIndices:
			source = column.featureConnectionsBySourceFeature.get(sourceIndex)
			if(source is None):
				records = pt.empty((databaseResidentPackedZero, databaseResidentPackedSourceRank + databaseResidentPackedOne), dtype=databaseResidentCoordinateDtype, device=deviceDatabase)
				column.featureConnectionsBySourceFeature[sourceIndex] = wrapValidatedRecords(records, targetSize)
			else:
				getRecords(source)
				shape = source.shape
				if(shape[:databaseResidentPackedConceptDimension] != targetSize[:databaseResidentPackedConceptDimension]):
					raise RuntimeError(databaseResidentPackedInvalidInput)
				if(shape[databaseResidentPackedConceptDimension] < targetSize[databaseResidentPackedConceptDimension] or shape[databaseResidentPackedFeatureDimension] < targetSize[databaseResidentPackedFeatureDimension]):
					expandedSize = shape[:databaseResidentPackedConceptDimension] + (max(shape[databaseResidentPackedConceptDimension], targetSize[databaseResidentPackedConceptDimension]), max(shape[databaseResidentPackedFeatureDimension], targetSize[databaseResidentPackedFeatureDimension]))
					source.expandTensorSize(expandedSize)
		column.loadedSourceFeatureIndices.update(sourceIndices)
	return


def getRecords(source):
	result = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections
		if(not isinstance(source, PackedDatabaseResidentFeatureConnections)):
			raise RuntimeError(databaseResidentPackedInvalidInput)
		validateShape(source.shape)
		entries = source.entries
		if(not isinstance(entries, np.ndarray) or entries.ndim != databaseResidentPackedArrayRank or entries.dtype != getEntryDtype() or not entries.flags.c_contiguous or not entries.flags.writeable):
			raise RuntimeError(databaseResidentPackedInvalidInput)
		result = pt.from_numpy(entries.view(np.int32).reshape((entries.size, databaseResidentPackedSourceRank + databaseResidentPackedOne)))
	return result


def wrapValidatedRecords(records, shape):
	result = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		from GIAANNcmn_databaseNetworkObservedColumn import PackedDatabaseResidentFeatureConnections
		validateShape(shape)
		if(not pt.is_tensor(records) or records.layout != pt.strided or records.device.type != databaseResidentPackedDeviceType or records.dtype != databaseResidentCoordinateDtype or records.dim() != databaseResidentPackedRecordRank or records.shape[databaseResidentPackedOne] != databaseResidentPackedSourceRank + databaseResidentPackedOne or not records.is_contiguous()):
			raise RuntimeError(databaseResidentPackedInvalidInput)
		# Only empty records or native-validated outputs enter here. NumPy retains the tensor owner.
		result = PackedDatabaseResidentFeatureConnections.__new__(PackedDatabaseResidentFeatureConnections)
		result.entries = records.numpy().view(getEntryDtype()).reshape((records.shape[databaseResidentPackedZero],))
		result.tensorSize = tuple(shape)
	return result


def validateShape(shape):
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		if(len(shape) != databaseResidentPackedSourceRank or any(type(size) is not int or size < databaseResidentPackedOne or size > databaseResidentCoordinateDimensionSizeMaximum for size in shape)):
			raise RuntimeError(databaseResidentPackedInvalidInput)
	return


def getEntryDtype():
	result = None
	if(optimisationStoreDatabaseResidentCoordinatesAsInt32):
		result = np.dtype([(databaseResidentPackedIndicesFieldName, np.int32, (databaseResidentPackedSourceRank,)), (databaseResidentPackedValuesFieldName, np.float32)])
	return result
