# %% Define the graph data structure

import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Example data for multiple patients
num_patients = 10
patient_data = []
for _ in range(num_patients):
    lesion_features = np.random.rand(5, 10)  # 5 lesions, 10 features each
    distances = np.random.rand(5, 5)  # Distance matrix between lesions

    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []
    for i in range(len(distances)):
        for j in range(len(distances)):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append([distances[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create node features
    x = torch.tensor(lesion_features, dtype=torch.float)

    # Example survival time and event
    survival_time = torch.tensor([365], dtype=torch.float)  # 1 year
    event = torch.tensor([1], dtype=torch.float)  # Event occurred

    # Create a Data object for the patient
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=survival_time, event=event)
    patient_data.append(data)

# Split the data into training and testing sets
train_data, test_data = train_test_split(patient_data, test_size=0.2, random_state=42)

# Create DataLoaders for batching
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# %% Define the GNN model

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Aggregate node features
        x = self.fc(x)
        return x

# Example usage
model = GNN(in_channels=10, hidden_channels=16, out_channels=8)
output = model(patient_data[0])
print(output)

# %% Custom loss function for survival prediction

def survival_loss(pred, time, event):
    # Cox proportional hazards loss
    hazard_ratio = torch.exp(pred)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = pred - log_risk
    censored_likelihood = uncensored_likelihood * event
    neg_log_likelihood = -torch.sum(censored_likelihood)
    return neg_log_likelihood

# %% Train the model

from torch.optim import Adam

# Define the model, optimizer, and loss function
model = GNN(in_channels=10, hidden_channels=16, out_channels=8)
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = survival_loss(output, batch.y, batch.event)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# %% Evaluate the model

model.eval()
preds = []
true_times = []
true_events = []

with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        preds.append(output.item())
        true_times.append(batch.y.item())
        true_events.append(batch.event.item())

# Calculate the concordance index
c_index = concordance_index(true_times, preds, true_events)
print(f'Concordance Index: {c_index}')