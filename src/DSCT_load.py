import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import random
import anndata
import cosg

from torch.utils.data import DataLoader, TensorDataset
seed = 4
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        
        self.attention_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        attention_weights = self.softmax(self.attention_layer(x))
        attended_features = attention_weights * x
        
        hidden = self.relu(self.fc(attended_features))
        output = self.out(hidden)
        
        return output, attention_weights
def fmap_load(sc_data,st_data,anno,gene_number,device):

    sc_data.obs['total_counts'] = sc_data.X.sum(axis=1)
    st_data.obs['total_counts'] = st_data.X.sum(axis=1)
    sc.pp.filter_cells(sc_data, min_counts=0)
    sc.pp.filter_cells(st_data, min_counts=0)  
    sc.pp.filter_genes(sc_data, min_cells=20)
    sc.pp.filter_genes(st_data, min_cells=20)
    corrected_sc_data = sc_data 
    corrected_st_data = st_data 
 
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names

    common_genes = set(sc_genes).intersection(set(st_genes))

    sc_data = sc_data[:, list(common_genes)]
    st_data = st_data[:, list(common_genes)]

    sc.pp.normalize_total(st_data, target_sum=1e4)
    sc.pp.log1p(st_data)
    groupby=anno
    cosg.cosg(sc_data,
        key_added='cosg',
        # use_raw=False, layer='log1p', ## e.g., if you want to use the log1p layer in adata
        mu=300,
        expressed_pct=0.1,
        remove_lowly_expressed=True,
         n_genes_user=gene_number,#之前是175
                   groupby=groupby)


    sc.pp.highly_variable_genes(st_data, n_top_genes=10000)
    high_var_genes = st_data.var[st_data.var['highly_variable']].index
    

    st_data = corrected_st_data
    markers_df = pd.DataFrame(sc_data.uns["cosg"]["names"]).iloc[0:500, :]

    marker_genes = set(markers_df.values.flatten().tolist())
    intersect_genes = high_var_genes.intersection(marker_genes)
    sc_data = sc_data[:, list(intersect_genes)]
    st_data = st_data[:, list(intersect_genes)]
    
    sorted_genes = sorted(st_data.var_names)
    st_data = st_data[:, sorted_genes]
    sc_data = sc_data[:, sorted_genes]
    
    X = sc_data.X.todense()
    y = sc_data.obs[anno].astype('category').cat.codes.values 

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    input_dim = X.shape[1]
    hidden_dim = 256
    output_dim = len(set(y))


    model = AttentionModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        outputs, attention_weights = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

 
    gene_importances = attention_weights.mean(dim=0).detach().cpu().numpy()  # 注意将tensor从GPU移回CPU
    genes = sc_data.var_names.tolist()
    sorted_indices = gene_importances.argsort()[::-1]  # 从高到低排序
    sorted_genes = [genes[i] for i in sorted_indices]
    sorted_importances = gene_importances[sorted_indices]


    num_to_keep = int(len(sorted_genes) * 0.9)
    selected_genes = sorted_genes[:num_to_keep]
    selected_importances = sorted_importances[:num_to_keep]
    sc_data = sc_data[:, selected_genes]
    st_data = st_data[:, selected_genes]
    
    

    
    return sc_data,st_data
    
