import pandas as pd  # 导入pandas库，用于数据处理
from add_remaining_useful_life import *  # 从指定模块导入所有内容


def loading_FD001(feature_num):
    dir_path = 'CMAPSSData/'
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv(dir_path + 'train_FD001.txt', sep='\s+', header=None, names=col_names)
    test = pd.read_csv(dir_path + 'test_FD001.txt', sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv(dir_path + 'RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

    # 根据 feature_num 参数决定保留哪些传感器
    if feature_num == 15:
        keep_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    elif feature_num == 6:
        keep_sensors = ['s_1', 's_5', 's_8', 's_13', 's_18', 's_21']
    elif feature_num == 10:
        keep_sensors = ['s_1', 's_5', 's_8', 's_13', 's_18', 's_21']
    else:
        raise ValueError("Unsupported feature number. Only 6,10 or 15 are supported.")

    drop_sensors = [s for s in sensor_names if s not in keep_sensors]
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    test.drop(labels=drop_labels, axis=1, inplace=True)

    title_train = train.iloc[:, 0:2]
    data_train = train.iloc[:, 2:]
    data_norm_train = (data_train - data_train.min()) / (data_train.max() - data_train.min() + 1e-8)
    train_norm = pd.concat([title_train, data_norm_train], axis=1)
    train_norm = add_remaining_useful_life(train_norm)
    train_norm['RUL'].clip(upper=125, inplace=True)
    group = train_norm.groupby(by="unit_nr")

    title_test = test.iloc[:, 0:2]
    data_test = test.iloc[:, 2:]
    data_norm_test = (data_test - data_test.min()) / (data_test.max() - data_test.min() + 1e-8)
    test_norm = pd.concat([title_test, data_norm_test], axis=1)
    group_test = test_norm.groupby(by="unit_nr")

    return group, y_test, group_test

