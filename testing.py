import torch  # 导入PyTorch库
import pandas as pd  # 导入Pandas库，用于数据处理
import numpy as np  # 导入NumPy库，用于数值计算
from torch.autograd import Variable  # 从PyTorch导入Variable模块，便于构建可计算的变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def testing(group_test, y_test, model, m):
    rmse = 0
    result = []
    j = 1

    while j <= 100:
        x_test = group_test.get_group(j).to_numpy()
        data_predict = 0
        if x_test.shape[0] < 30:
            i += 1
            continue
        for t in range(1, x_test.shape[0] - 25):
            if t == x_test.shape[0] - 1:
                # 修改：确保最后一个时间步有3行数据：最后两行 + 补一行零
                X_test = np.append(x_test[t - 1:t + 1, 2:], [np.zeros(m)], axis=0)
            else:
                X_test = x_test[t - 1:t + 25, 2:]

            X_test_tensors = torch.Tensor(X_test)
            X_test_tensors_final = X_test_tensors.reshape((1, 1, X_test_tensors.shape[0], X_test_tensors.shape[1])).to(device)
            test_predict = model(X_test_tensors_final, t)
            data_predict = test_predict.data.cpu().numpy()[-1]

            if data_predict - 1 < 0:
                data_predict = 0
            else:
                data_predict -= 1

        result.append(data_predict)
        rmse += np.power((data_predict - y_test.to_numpy()[j - 1]), 2)
        j += 1

    rmse = np.sqrt(rmse / 100)
    result = y_test.join(pd.DataFrame(result))
    result = result.sort_values('RUL', ascending=False)
    return rmse, result



