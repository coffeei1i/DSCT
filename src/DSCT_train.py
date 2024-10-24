from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import diopy
import scanpy as sc
import random
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size4, num_classes)

        self.fc6 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)


        out = self.fc5(out)+self.fc6(x)
        return out
def fmap_train(sc_data,st_data,result_save_path,model_save_path,plot_save_path,train_num,num_classes,anno):

    type_num=len(set(sc_data.obs[anno]))
    num_classes = len(set(sc_data.obs[anno]))
    cell_types = sc_data.obs[anno]
    cell_types_categorical = cell_types.astype("category")
    cell_types_integer = cell_types_categorical.cat.codes
    sc_data.obs["type_integer"] = cell_types_integer
#     num_list = list(range(1, len(st_data.obs)))
    num_list = list(range(1, len(sc_data.obs)))
    split_idx = int(len(num_list) * 0.8)

    train_data = sc_data[:split_idx, :]
    valid_data = sc_data[split_idx:, :]


    X_train = train_data.X.todense()
    X_valid = valid_data.X.todense()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_train = train_data.obs["type_integer"].values
    y_valid = valid_data.obs["type_integer"].values
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 81
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    input_size = sc_data.X.shape[1]
    hidden_size1 = 256
    hidden_size2 = 128
    hidden_size3 = 64
    hidden_size4 = 32
    learning_rate = 0.01
    num_epochs = train_num
    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    
    labels = sc_data.obs[anno] 
    label_freq = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_freq)
    alpha = 1.08  
    weights = {label: (total_samples / (num_classes * count)) ** alpha for label, count in label_freq.items()}
    weights_tensor = torch.tensor([weights[label] for label in sorted(weights)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)


    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_valid_tensor = X_valid_tensor.to(device)
    y_valid_tensor = y_valid_tensor.to(device)

    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                outputs = model(X_valid_tensor)
                _, predicted = torch.max(outputs.data, 1)
                total += y_valid_tensor.size(0)
                correct += (predicted == y_valid_tensor).sum().item()
                valid_accuracy = 100 * correct / total

            model.train()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model_save_path2=model_save_path+".pth"
    torch.save(model.state_dict(), model_save_path2)



    new_data_X = st_data.X.todense()
    new_data_tensor = torch.tensor(new_data_X, dtype=torch.float32).to(device)  # Move the new data tensor to the device (GPU or CPU)

    model = model.to(device)  # Make sure the model is on the device (GPU or CPU)
    model.eval()



    with torch.no_grad():
        
        outputs = model(new_data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)


    probabilities_np = probabilities.cpu().numpy()
    probabilities_df = pd.DataFrame(probabilities_np, columns=cell_types_categorical.cat.categories)
    result_save_path2=result_save_path+"_2.csv"
    probabilities_df.to_csv(result_save_path2, index=False)
    predicted_labels = cell_types_categorical.cat.categories[predicted.cpu()]
    st_data.obs['predicted_classes'] = predicted_labels


       
