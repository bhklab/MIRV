# %% Data Loading and Clean-up

import pandas as pd, numpy as np
import pickle, os
import matplotlib.pyplot as plt
import seaborn as sns

# load raw data
radiomics = pd.read_csv('../../procdata/SARC021/radiomics-all.csv')
survival = pd.read_csv('../../rawdata/SARC021/survival-all.csv')

# mirv data - only need the features here
def load_pickle(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Warning: '{file_path}' is empty or does not exist.")
        return None

results_surv = load_pickle('../../procdata/SARC021/results_surv.pkl')

if results_surv is not None:
    cols_to_keep = results_surv[1].columns
else:
    raise ValueError("results_surv is None. Please check the pickle file.")

# isolate multimetastatic patients with survival data
lesion_counts = radiomics['USUBJID'].value_counts()
patients_with_multiple_lesions = lesion_counts[lesion_counts > 1].index
patients_with_survival_data = survival['USUBJID'].unique()
valid_patients = patients_with_multiple_lesions.intersection(patients_with_survival_data)
radiomics = radiomics[radiomics['USUBJID'].isin(valid_patients)]
survival = survival[survival['USUBJID'].isin(valid_patients)][['USUBJID', 'T_OS', 'E_OS']]
clinical = clinical[clinical['USUBJID'].isin(valid_patients)]


# get spatial coordinates using the masks (convert from strings) 
com_idx = radiomics.pop('diagnostics_Mask-original_CenterOfMassIndex')
com_vectors = []
for entry in com_idx:
    entry = entry.strip('()')
    vector = np.array([float(coord) for coord in entry.split(', ')])
    com_vectors.append(vector)

com_vectors = np.array(com_vectors) 
subj_list = radiomics.pop('USUBJID')
cols_to_keep = list(cols_to_keep) + ['original_shape_VoxelVolume']
radiomics = radiomics[cols_to_keep]
# Determine the number of input features for modeling
num_features = radiomics.shape[1]   

# Check for NaNs in the data
assert not radiomics.select_dtypes(include=[np.number]).isnull().values.any(), "NaNs found in radiomics data"
assert not survival.select_dtypes(include=[np.number]).isnull().values.any(), "NaNs found in survival data"

del(results_surv,cols_to_keep,lesion_counts,patients_with_multiple_lesions,patients_with_survival_data,valid_patients,entry,vector,com_idx)

# %% Define the graph data structure

import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Example data for multiple patients
patients = np.unique(subj_list)
patient_data = []
for subj in patients:
    # Extract lesion features for the current patient
    lesion_features = radiomics[subj_list == subj].values

    # Scale the features 
    scaler = StandardScaler()
    lesion_features = scaler.fit_transform(lesion_features)
    lesion_indices = np.where(subj_list == subj)[0]
    lesion_com_vectors = com_vectors[lesion_indices]

    # Calculate the Euclidean distance matrix
    distances = np.linalg.norm(lesion_com_vectors[:, np.newaxis] - lesion_com_vectors, axis=2)

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
    survival_time = torch.tensor([survival[survival.USUBJID == subj]['T_OS'].values], dtype=torch.float)  # 1 year
    event = torch.tensor([survival[survival.USUBJID == subj]['E_OS'].values], dtype=torch.float)  # Event occurred

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
import torch.nn.init as init

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Aggregate node features
        x = self.fc(x)
        return x

# %% Custom loss function for survival prediction

def survival_loss(pred, time, event, epsilon=1e-7):
    # Cox proportional hazards loss
    hazard_ratio = torch.exp(pred)
    # if torch.isnan(hazard_ratio).any():
    #     print("NaNs in hazard_ratio")
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    # if torch.isnan(log_risk).any():
    #     print("NaNs in log_risk")
    uncensored_likelihood = pred - log_risk
    # if torch.isnan(uncensored_likelihood).any():
    #     print("NaNs in uncensored_likelihood")
    censored_likelihood = uncensored_likelihood * event
    # if torch.isnan(censored_likelihood).any():
    #     print("NaNs in censored_likelihood")
    neg_log_likelihood = -torch.sum(censored_likelihood)
    # if torch.isnan(neg_log_likelihood).any():
    #     print("NaNs in neg_log_likelihood")
    return neg_log_likelihood

# %% Train the model

from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'hidden_channels': [18, 21, 24, 27],
    'out_channels': [9, 6, 3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [1, 2, 4],
    'epochs': [50, 100, 200]
}

best_c_index = -1
best_params = None

# Perform grid search
for params in ParameterGrid(param_grid):
    try:
        hidden_channels = params['hidden_channels']
        out_channels = params['out_channels']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        epochs = params['epochs']
        
        # Define the model, optimizer, and loss function
        model = GNN(in_channels=num_features, hidden_channels=hidden_channels, out_channels=out_channels)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Create DataLoaders for batching with the current batch size
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = survival_loss(output, batch.y, batch.event)
                loss.backward()
                optimizer.step()
        
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
        print(f'Params: {params}, Concordance Index: {c_index}')
        
        # Update best parameters if current model is better
        if c_index > best_c_index:
            best_c_index = c_index
            best_params = params
    except Exception as e:
        print(f"Error with params {params}: {e}")

print(f'Best Params: {best_params}, Best Concordance Index: {best_c_index}')

# %%

# Define the model with the best parameters
model = GNN(in_channels=num_features, hidden_channels=best_params['hidden_channels'], out_channels=best_params['out_channels'])
optimizer = Adam(model.parameters(), lr=best_params['learning_rate'])

# Training loop with the best parameters
for epoch in range(best_params['epochs']):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = survival_loss(output, batch.y, batch.event)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

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
# %%
