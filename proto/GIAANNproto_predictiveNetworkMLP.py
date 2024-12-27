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
import GIAANNproto_predictiveNetworkOperations

def nextWordPredictionMLPcreate(databaseNetworkObject):
	global model, criterion, criterion_c, criterion_f, optimizer, batch_size

	model = NextWordPredictionMLPmodel(databaseNetworkObject)
	model = model.to(devicePredictiveNetworkModel)
	model.train()  # set model to training mode

	if(inferencePredictiveNetworkIndependentFCpredictions):
		if(multipleTargets):
			criterion_c = nn.BCEWithLogitsLoss()
			criterion_f = nn.BCEWithLogitsLoss()
		else:
			criterion_c = nn.MSELoss()
			criterion_f = nn.MSELoss()
	else:
		if(multipleTargets):
			criterion = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.MSELoss()
			
	optimizer = optim.Adam(model.parameters(), lr=inferencePredictiveNetworkLearningRate)
	batch_size = 1

def nextWordPredictionMLPtrainStep(global_feature_neurons, targets, targets_c, targets_f):
	global model, criterion, optimizer, batch_size
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targets_c = targets_c.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
		targets_f = targets_f.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	else:
		targets = targets.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim

	global_feature_neurons = global_feature_neurons.unsqueeze(0)	#add batch dim	
	global_feature_neurons = global_feature_neurons.to(devicePredictiveNetworkModel)
	global_feature_neurons = global_feature_neurons.to_dense()
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		outputs_c, outputs_f = model(global_feature_neurons) # Outputs shape: (batch_size, c, f)
		loss_c = criterion_c(outputs_c, targets_c)
		loss_f = criterion_f(outputs_f, targets_f)
		loss = loss_c + loss_f  # Combine losses
	else:	
		outputs = model(global_feature_neurons)  # Outputs shape: (batch_size, c, f)
		loss = criterion(outputs, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss_value = loss.item()
	print("loss_value = ", loss_value)

	if(inferencePredictiveNetworkIndependentFCpredictions):
		topk_c = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsC(outputs_c)
		topk_f = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsF(outputs_f)
		return topk_c, topk_f
	else:
		topk = GIAANNproto_predictiveNetworkOperations.getTopkPredictions(outputs)
		return topk
		
class NextWordPredictionMLPmodel(nn.Module):
	def __init__(self, databaseNetworkObject):
		super(NextWordPredictionMLPmodel, self).__init__()
		self.s = databaseNetworkObject.s
		self.c = databaseNetworkObject.c
		self.f = databaseNetworkObject.f
		self.p = databaseNetworkObject.p
		
		if(inferencePredictiveNetworkUseInputAllProperties):
			input_size = databaseNetworkObject.s * databaseNetworkObject.c * databaseNetworkObject.f * databaseNetworkObject.p
			hidden_size_multiplier = 1	#default: 1	#TODO: requires testing
		else:
			input_size = databaseNetworkObject.s * databaseNetworkObject.c * databaseNetworkObject.f
			hidden_size_multiplier = 2	#default: 2	#TODO: requires testing
		output_size = databaseNetworkObject.c * databaseNetworkObject.f
		hidden_size = input_size * hidden_size_multiplier

		self.fc1 = nn.Linear(input_size, hidden_size, device=devicePredictiveNetworkModel)
		self.relu = nn.ReLU()
		if(inferencePredictiveNetworkIndependentFCpredictions):
			self.fc_c = nn.Linear(hidden_size, self.c, device=devicePredictiveNetworkModel)
			self.fc_f = nn.Linear(hidden_size, self.f, device=devicePredictiveNetworkModel)
		else:
			self.fc2 = nn.Linear(hidden_size, output_size, device=devicePredictiveNetworkModel)
	
	def forward(self, x):
		x = x.view(x.size(0), -1)	#Flatten the input: (batch_size, c, f) -> (batch_size, c*f)
		out = self.fc1(x)
		out = self.relu(out)
		if(inferencePredictiveNetworkIndependentFCpredictions):
			out_c = self.fc_c(out)  # Shape: (batch_size, c)
			out_f = self.fc_f(out)  # Shape: (batch_size, f)
			out_c = out_c.view(batch_size, self.c)
			out_f = out_f.view(batch_size, self.f)
			return out_c, out_f
		else:
			out = self.fc2(out)	#Output shape: (batch_size, c*f)
			out = out.view(batch_size, self.c, self.f)
			return out



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


