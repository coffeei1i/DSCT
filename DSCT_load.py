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
    # 获取 sc_data 和 st_data 的基因列表
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names
    # 使用集合操作找到共同基因
    common_genes = set(sc_genes).intersection(set(st_genes))
    # 仅保留 sc_data 和 st_data 中共同的基因
    sc_data = sc_data[:, list(common_genes)]
    st_data = st_data[:, list(common_genes)]
    # # 首先，对数据进行预处理，例如归一化和对数缩放
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

    # 获取高变基因
    sc.pp.highly_variable_genes(st_data, n_top_genes=10000)#原本是6000
    high_var_genes = st_data.var[st_data.var['highly_variable']].index
    
    #取log之后要还回去
    st_data = corrected_st_data
    
    # 获取每个类别的前个marker基因
    # sc.tl.rank_genes_groups(sc_data, groupby="annotation", use_raw=False)
    markers_df = pd.DataFrame(sc_data.uns["cosg"]["names"]).iloc[0:500, :]#原本是3000

    marker_genes = set(markers_df.values.flatten().tolist())

    # 计算高变基因和marker基因的交集
    intersect_genes = high_var_genes.intersection(marker_genes)
    
        # 在 sc_data 和 st_data 中保留交集基因
    sc_data = sc_data[:, list(intersect_genes)]
    st_data = st_data[:, list(intersect_genes)]
    
    X = sc_data.X.todense()
    # 此处仅为示例，您需要确保您的标签是适当的
    y = sc_data.obs[anno].astype('category').cat.codes.values 

    # 2. 将数据移动到设备上
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    input_dim = X.shape[1]
    hidden_dim = 256
    output_dim = len(set(y))

    # 将模型移动到设备上
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

    # 查看基因的注意力权重
    gene_importances = attention_weights.mean(dim=0).detach().cpu().numpy()  # 注意将tensor从GPU移回CPU
    # 获取基因的名称
    genes = sc_data.var_names.tolist()

    # 对基因按权重进行排序
    sorted_indices = gene_importances.argsort()[::-1]  # 从高到低排序
    sorted_genes = [genes[i] for i in sorted_indices]
    sorted_importances = gene_importances[sorted_indices]

    # 选择前90%的基因
    num_to_keep = int(len(sorted_genes) * 0.9)
    selected_genes = sorted_genes[:num_to_keep]
    selected_importances = sorted_importances[:num_to_keep]

    # 使用selected_genes过滤sc_data中的基因
    sc_data = sc_data[:, selected_genes]
    st_data = st_data[:, selected_genes]
    
    

    
    return sc_data,st_data
    