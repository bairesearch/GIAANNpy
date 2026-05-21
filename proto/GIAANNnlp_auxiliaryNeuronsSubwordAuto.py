"""GIAANNnlp_auxiliaryNeuronsSubwordAuto.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP auxiliary neurons subword auto

"""

from GIAANNcmn_globalDefs import *


if(auxiliaryNeurons and auxiliaryNeuronsTokenisationSubwordAuto):

	import GIAANNnlp_auxiliaryNeuronsAuto

	def updateAutoAuxiliaryConnections(databaseNetworkObject):
		GIAANNnlp_auxiliaryNeuronsAuto.updateAutoAuxiliaryConnections(databaseNetworkObject, subwordSimilarity=True)
		return
