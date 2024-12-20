"""GIAANNproto_predictiveNetworkMLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive Network MLP

FUTURE: consider pytorch implementation for sparse tensors.

"""

import torch as pt
import torch.nn as nn
import torch.optim as optim


from GIAANNproto_globalDefs import *
	
class NextWordPredictionMLPmodel(nn.Module):
	def __init__(self, databaseNetworkObject):
		super(NextWordPredictionMLPmodel, self).__init__()
		self.s = databaseNetworkObject.s
		self.c = databaseNetworkObject.c
		self.f = databaseNetworkObject.f
		self.p = databaseNetworkObject.p

		input_size = databaseNetworkObject.s * databaseNetworkObject.c * databaseNetworkObject.f #* databaseNetworkObject.p
		output_size = databaseNetworkObject.c * databaseNetworkObject.f
		hidden_size_multiplier = 2	#TODO: requires testing
		hidden_size = (input_size+output_size) * hidden_size_multiplier

		self.fc1 = nn.Linear(input_size, hidden_size, device=devicePredictiveNetworkModel)
		self.relu = nn.ReLU(device=devicePredictiveNetworkModel)
		self.fc2 = nn.Linear(hidden_size, output_size, device=devicePredictiveNetworkModel)

	def forward(self, x):
		x = x.view(x.size(0), -1)	#Flatten the input: (batch_size, c, f) -> (batch_size, c*f)
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)	#Output shape: (batch_size, c*f)
		out = out.view(batch_size, self.c, self.f)
		return out

def nextWordPredictionMLPcreate(databaseNetworkObject):
	global model, criterion, optimizer, batch_size

	model = NextWordPredictionMLPmodel(databaseNetworkObject)
	model = model.to(devicePredictiveNetworkModel)
	model.train()  # set model to training mode

	if(multipleTargets):
		criterion = nn.BCEWithLogitsLoss()
	else:
		criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0005)
	batch_size = 1
	
def nextWordPredictionMLPtrainStep(global_feature_neurons_activation, targets):
	global model, criterion, optimizer, batch_size
	
	targets = targets.unsqueeze(0)	#add batch dim
	targets = targets.to(devicePredictiveNetworkModel)

	global_feature_neurons_activation = global_feature_neurons_activation.unsqueeze(0)	#add batch dim	
	global_feature_neurons_activation = global_feature_neurons_activation.to(devicePredictiveNetworkModel)
	global_feature_neurons_activation = global_feature_neurons_activation.to_dense()
		
	outputs = model(global_feature_neurons_activation)  # Outputs shape: (batch_size, c, f)

	loss = criterion(outputs, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss_value = loss.item()
	print("loss_value = ", loss_value)

	return getTopkPredictions(outputs)

def getTopkPredictions(outputs):

	with pt.no_grad():

		if(multipleTargets):
			# Apply sigmoid to get probabilities
			probs = pt.sigmoid(outputs)  # Shape: (batch_size, c, f)
		else:
			probs = outputs

		column_probs = probs.mean(dim=2)  # Shape: (batch_size, c)
		_, concept_columns_indices_next = pt.topk(column_probs, kcNetwork, dim=1)

		# For each of the top kcNetwork columns, compute top kf features
		top_kf_indices = []
		for column_idx in concept_columns_indices_next:
			column_data = probs[:, column_idx, :]  # Shape: (batch_size, f)
			feature_probs = column_data.mean(dim=0)  # Shape: (f,)
			topk_feature_probs, topk_feature_indices = pt.topk(feature_probs, kf)  # Shapes: (kf,), (kf,)
			top_kf_indices.append(topk_feature_indices)

		concept_columns_feature_indices_next = pt.stack(top_kf_indices)  # Shape: (batch_size, kcNetwork, kf)

	concept_columns_indices_next = concept_columns_indices_next[0]	#select first sample of batch
	concept_columns_feature_indices_next = concept_columns_feature_indices_next[0]	#select first sample of batch
		
	return concept_columns_indices_next, concept_columns_feature_indices_next

def save_model(model, path, filename="model.pt"):
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Construct full filepath
    filepath = os.path.join(path, filename)
    
    # Save the model state_dict
    pt.save(model.state_dict(), filepath)
    return filepath


def load_model(model, filepath, map_location=None):
    # Load state_dict from file
    state_dict = pt.load(filepath, map_location=map_location)
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    
    return model


