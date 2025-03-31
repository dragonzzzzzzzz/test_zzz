import torch.nn as nn  # 导入PyTorch的神经网络模块
import os  # 导入操作系统模块
import argparse  # 引入 argparse 模块，用于命令行参数解析
from model import Transformer  # 从model模块导入Transformer类
from testing import *  # 从testing模块导入所有内容
from visualize import *  # 从visualize模块导入所有内容
from loading_data import loading_FD001

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def training(m):
    result, best_rmse =0,0
    best_rmse = float('inf')
    patience = 5
    wait = 0
    best_model_state = None
    total_units = 0

    for epoch in range(num_epochs):
        i = 1
        epoch_loss = 0.0
        model.train()
        while i <= 100:
            x = group.get_group(i).to_numpy()
            if x.shape[0] < 30:
                i += 1
                continue

            valid_steps = x.shape[0] - 2
            unit_loss = 0.0
            optim.zero_grad()

            for t in range(1, x.shape[0] - 25):
                X = x[t - 1:t + 25, 2:-1]
                y = x[t, -1:]
                X_train_tensors = torch.Tensor(X).to(device)  # 将数据移动到 GPU
                y_train_tensors = torch.Tensor(y).to(device)  # 将数据移动到 GPU
                X_train_tensors_final = X_train_tensors.reshape(
                    (1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))

                outputs = model(X_train_tensors_final, t)
                loss = criterion(outputs, y_train_tensors)
                loss.backward()
                unit_loss += loss.item()

            optim.step()
            optim.zero_grad()
            avg_unit_loss = unit_loss / valid_steps
            epoch_loss += avg_unit_loss
            total_units += 1
            i += 1

        avg_epoch_loss = epoch_loss / total_units if total_units > 0 else float('inf')

        model.eval()
        with torch.no_grad():
            rmse, result = testing(group_test, y_test, model, m)

        rmse_value = rmse if isinstance(rmse, float) else rmse[0]
        print(f"Epoch: {epoch}, training loss: {avg_epoch_loss:.5f}, testing rmse: {rmse_value:.5f}")

        if rmse_value < best_rmse:
            best_rmse = rmse_value
            best_model_state = model.state_dict()
            wait = 0
            torch.save(best_model_state, f'best_model_m{m}.pth')
            print("Best model saved with RMSE:", best_rmse)
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}, best RMSE: {best_rmse:.5f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return result, best_rmse


import argparse
import os
import torch
import torch.nn as nn
from model import Transformer
from loading_data import loading_FD001
from testing import testing
from visualize import visualize

def load_model(model, file_path):
    if os.path.exists(file_path):
        try:
            model.load_state_dict(torch.load(file_path))
            model.eval()
            print(f"Model loaded from {file_path}")
        except RuntimeError as e:
            print(f"Error loading model from {file_path}: {e}")
            print("Starting with a new model.")
    else:
        print(f"Model file {file_path} not found. Starting with a new model.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001', help='which dataset to run')
    # 添加 --feature_num 参数，自由设置特征数，比如 6 或 15
    parser.add_argument('--feature_num', type=int, default=15, help='number of sensor features to use')
    opt = parser.parse_args()

    num_epochs = 1
    d_model = 256
    heads = 4
    N = 4
    m = opt.feature_num  # 根据命令行参数设置特征数

    if opt.dataset == 'FD001':
        group, y_test, group_test = loading_FD001(m)
        dropout = 0.3
        model = Transformer(m, d_model, N, heads, dropout).to(device)  # 将模型移动到 GPU
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optim = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss().to(device)  # 将损失函数移动到 GPU

        file_path = f'best_model_m{m}.pth'
        model = load_model(model, file_path)

        result, rmse = training(m)
        visualize(result, rmse)
    else:
        print('Either dataset not implemented or not defined')
