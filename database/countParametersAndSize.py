'''
example usage;
assume database folder has been zipped to database.zip
conda activate pytorchsenv
set layoutType = "legacy"
python countParametersAndSize.py databaseOscar5000-numSeedTokensInference8-spacyPipelineOptimisations-1k.zip
set layoutType = "v1l"
python countParametersAndSize.py databaseOscar5000-numSeedTokensInference8-spacyPipelineOptimisations-1l.zip
'''

import argparse
import io
import os
import zipfile
import torch as pt

layoutType = "v1l"	#supported: "v1l", "legacy"

def countNonZero(t):
	result = 0
	if(isinstance(t, pt.Tensor)):
		if(t.is_sparse or (hasattr(t, "layout") and t.layout in (pt.sparse_coo, pt.sparse_csr, pt.sparse_csc, pt.sparse_bsr, pt.sparse_bsc))):
			result = int(t._nnz())
		else:
			result = int(pt.count_nonzero(t).item())
	return result

def getObservedColumnsPrefix(fileName):
	result = None
	marker = "/observedColumns/"
	if(fileName.startswith("observedColumns/")):
		result = "observedColumns/"
	elif(marker in fileName):
		markerIndex = fileName.index(marker)
		result = fileName[:markerIndex + len(marker)]
	return result

def getObservedColumnsPrefixFromZip(zipInfoList):
	prefixes = set()
	result = None
	for info in zipInfoList:
		if(info.is_dir()):
			continue
		prefix = getObservedColumnsPrefix(info.filename)
		if(prefix is not None):
			prefixes.add(prefix)
	if(len(prefixes) == 0):
		raise RuntimeError("countParametersAndSize error: no observedColumns/ folder found inside zip")
	if(len(prefixes) != 1):
		raise RuntimeError("countParametersAndSize error: multiple observedColumns/ roots found inside zip: " + ", ".join(sorted(prefixes)))
	for prefix in prefixes:
		result = prefix
	return result

def isUnsignedIntegerText(text):
	result = text != "" and text.isdigit()
	return result

def isObservedColumnFolderName(pathPart):
	result = False
	if(pathPart.startswith("cIndex")):
		indexText = pathPart[len("cIndex"):]
		if(isUnsignedIntegerText(indexText)):
			result = True
	return result

def isObservedColumnSourceFeatureTensorFileName(pathPart):
	result = False
	tensorExtension = ".pt"
	if(pathPart.startswith("fIndex") and pathPart.endswith(tensorExtension)):
		indexText = pathPart[len("fIndex"):-len(tensorExtension)]
		if(isUnsignedIntegerText(indexText)):
			result = True
	return result

def isLegacyMetadataFileName(relativeName):
	result = False
	if("/" not in relativeName and relativeName.lower().endswith("_data.pkl")):
		result = True
	return result

def isLegacyFeatureConnectionsTensorFileName(relativeName):
	result = False
	if("/" not in relativeName and relativeName.lower().endswith("_featureconnections.pt")):
		result = True
	return result

def isLegacyFeatureNeuronsTensorFileName(relativeName):
	result = False
	if("/" not in relativeName and relativeName.lower().endswith("_featureneurons.pt")):
		result = True
	return result

def loadTensorPayloadFromZip(zf, info):
	payload = None
	with zf.open(info, "r") as f:
		data = f.read()
	payload = pt.load(io.BytesIO(data), map_location="cpu")
	return payload

def getTensorPayloadStats(payload):
	resultNnz = 0
	resultLoaded = False
	if(isinstance(payload, pt.Tensor)):
		resultNnz = countNonZero(payload)
		resultLoaded = True
	elif(isinstance(payload, dict)):
		for value in payload.values():
			if(isinstance(value, pt.Tensor)):
				resultNnz += countNonZero(value)
		resultLoaded = True
	return resultNnz, resultLoaded

def main():
	parser = argparse.ArgumentParser(description="Scan .pt files inside zipFileName.zip/zipFileName/observedColumns/")
	parser.add_argument("zipFileName", help="Base name (without .zip), or a full path to the .zip")
	args = parser.parse_args()

	zipPath = args.zipFileName
	if(not zipPath.lower().endswith(".zip")):
		zipPath = zipPath + ".zip"

	totalNnz = 0
	totalPtBytesUncompressed = 0
	totalPtBytesCompressed = 0
	totalDataPklBytesUncompressed = 0
	totalZipBytesUncompressed = 0
	numFiles = 0
	numLoaded = 0
	numSkipped = 0
	numMetadataFiles = 0
	numFeatureNeuronsTensorFiles = 0
	prefix = None
	if(layoutType == "v1l"):
		numSourceFeatureTensorFiles = 0
		with zipfile.ZipFile(zipPath, "r") as zf:
			prefix = getObservedColumnsPrefixFromZip(zf.infolist())
			for info in zf.infolist():
				if(not info.is_dir()):
					totalZipBytesUncompressed += int(info.file_size)
				name = info.filename
				if(not name.startswith(prefix)):
					continue
				if(info.is_dir()):
					continue
				relativeName = name[len(prefix):]
				pathParts = relativeName.split("/")
				isMetadata = False
				isFeatureNeuronsTensor = False
				isSourceFeatureTensor = False
				isRecognisedEntry = False
				if(len(pathParts) == 2 and isObservedColumnFolderName(pathParts[0]) and pathParts[1] == "data.pkl"):
					isMetadata = True
					isRecognisedEntry = True
				if(len(pathParts) == 2 and isObservedColumnFolderName(pathParts[0]) and pathParts[1] == "featureNeurons.pt"):
					isFeatureNeuronsTensor = True
					isRecognisedEntry = True
				if(len(pathParts) == 3 and isObservedColumnFolderName(pathParts[0]) and pathParts[1] == "featureConnections" and isObservedColumnSourceFeatureTensorFileName(pathParts[2])):
					isSourceFeatureTensor = True
					isRecognisedEntry = True
				if(not isRecognisedEntry):
					if(relativeName.lower().endswith(".pt") or relativeName.lower().endswith(".pkl")):
						raise RuntimeError("countParametersAndSize error: unsupported v1l observedColumns entry = " + name)
					continue
				if(isMetadata):
					totalDataPklBytesUncompressed += int(info.file_size)
					numMetadataFiles += 1
				if(not name.lower().endswith(".pt")):
					continue
				if(isSourceFeatureTensor):
					numSourceFeatureTensorFiles += 1
				if(isFeatureNeuronsTensor):
					numFeatureNeuronsTensorFiles += 1
				numFiles += 1
				totalPtBytesUncompressed += int(info.file_size)
				totalPtBytesCompressed += int(info.compress_size)
				try:
					payload = loadTensorPayloadFromZip(zf, info)
					payloadNnz, payloadLoaded = getTensorPayloadStats(payload)
					if(payloadLoaded):
						totalNnz += payloadNnz
						numLoaded += 1
					else:
						numSkipped += 1
				except Exception as e:
					print(f"WARNING: failed to load {name}: {e}")
					numSkipped += 1
	elif(layoutType == "legacy"):
		numLegacyFeatureConnectionsTensorFiles = 0
		with zipfile.ZipFile(zipPath, "r") as zf:
			prefix = getObservedColumnsPrefixFromZip(zf.infolist())
			for info in zf.infolist():
				if(not info.is_dir()):
					totalZipBytesUncompressed += int(info.file_size)
				name = info.filename
				if(not name.startswith(prefix)):
					continue
				if(info.is_dir()):
					continue
				relativeName = name[len(prefix):]
				isMetadata = isLegacyMetadataFileName(relativeName)
				isLegacyFeatureConnectionsTensor = isLegacyFeatureConnectionsTensorFileName(relativeName)
				isFeatureNeuronsTensor = isLegacyFeatureNeuronsTensorFileName(relativeName)
				isRecognisedEntry = isMetadata or isLegacyFeatureConnectionsTensor or isFeatureNeuronsTensor
				if(not isRecognisedEntry):
					if(relativeName.lower().endswith(".pt") or relativeName.lower().endswith(".pkl")):
						raise RuntimeError("countParametersAndSize error: unsupported legacy observedColumns entry = " + name)
					continue
				if(isMetadata):
					totalDataPklBytesUncompressed += int(info.file_size)
					numMetadataFiles += 1
				if(not name.lower().endswith(".pt")):
					continue
				if(isLegacyFeatureConnectionsTensor):
					numLegacyFeatureConnectionsTensorFiles += 1
				if(isFeatureNeuronsTensor):
					numFeatureNeuronsTensorFiles += 1
				numFiles += 1
				totalPtBytesUncompressed += int(info.file_size)
				totalPtBytesCompressed += int(info.compress_size)
				try:
					payload = loadTensorPayloadFromZip(zf, info)
					payloadNnz, payloadLoaded = getTensorPayloadStats(payload)
					if(payloadLoaded):
						totalNnz += payloadNnz
						numLoaded += 1
					else:
						numSkipped += 1
				except Exception as e:
					print(f"WARNING: failed to load {name}: {e}")
					numSkipped += 1
	else:
		raise RuntimeError("countParametersAndSize error: unsupported layoutType = " + str(layoutType))

	totalPtGiB = totalPtBytesUncompressed / (1024 ** 3)
	totalZipGiBUncompressed = totalZipBytesUncompressed / (1024 ** 3)
	
	print(f"Zip: {zipPath}")
	print(f"Observed-columns prefix scanned: {prefix}")
	print(f"Observed-columns layout: {layoutType}")
	print(f"Observed-column metadata files: {numMetadataFiles}")
	print(f"Total columns: {numMetadataFiles}")
	if(layoutType == "v1l"):
		print(f"Observed-column source-feature tensors: {numSourceFeatureTensorFiles}")
	if(layoutType == "legacy"):
		print(f"Observed-column legacy feature-connection tensors: {numLegacyFeatureConnectionsTensorFiles}")
	print(f"Observed-column feature-neuron tensors (when lowMem=True): {numFeatureNeuronsTensorFiles}")
	print(f".pt files found: {numFiles}")
	print(f".pt files loaded: {numLoaded}")
	print(f".pt files skipped/failed: {numSkipped}")
	print(f"Total .pt size (compressed bytes in zip): {totalPtBytesCompressed}")
	print(f"Total .pt size (uncompressed bytes): {totalPtBytesUncompressed}")
	print(f"Total .pt size (uncompressed GiB): {totalPtGiB:.3f}")
	print("")
	print(f"Total non-zero values (nnz): {totalNnz}")
	print(f"Total zip size (uncompressed GiB): {totalZipGiBUncompressed:.3f}")
	#print(f"Total zip size trainStoreFeatureMapsGlobally (uncompressed GiB): {totalZipGiBUncompressed_trainStoreFeatureMapsGlobally:.3f}")

if __name__ == "__main__":
	main()
