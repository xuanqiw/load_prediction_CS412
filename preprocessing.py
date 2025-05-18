import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def preprocess_excel(filepath, add_time_features=True, external_scaler=None):
    """
    加载 Excel 文件，进行缺失值填补、标准化、滑动窗口样本构建。
    每条样本输入 shape = (96, n_features)
    包含：负荷、weekday、month、hour_index、前一天mean/max/min
    """
    # 1. 加载数据
    df = pd.read_excel(filepath)

    # 处理 Date 列和负荷数据列
    dates = pd.to_datetime(df['Date'])
    power_cols = [col for col in df.columns if 'Power' in col]

    # 对负荷数据进行缺失值处理
    # 方法1：使用前向填充和后向填充
    df[power_cols] = df[power_cols].ffill().bfill()
    # 如果仍有NaN（例如整列都是NaN），用0填充
    df[power_cols] = df[power_cols].fillna(0)

    df_values = df[power_cols]
    load_data = df_values.values  # (N天, 96)

    # 2. 标准化负荷数据
    if external_scaler is None:
        scaler = StandardScaler()
        load_scaled = scaler.fit_transform(load_data)
    else:
        scaler = external_scaler
        load_scaled = scaler.transform(load_data)

    # 3. 构造时间特征 weekday, month
    if add_time_features:
        weekdays = dates.dt.weekday.values / 6.0
        months = (dates.dt.month.values - 1) / 11.0
        time_features = np.stack([weekdays, months], axis=1)
    else:
        time_features = None

    # 4. 构造 (X, Y) 样本
    X, Y = [], []
    for i in range(1, len(load_scaled)):
        load_input = load_scaled[i - 1].reshape(96, 1)  # 前一天负荷 (96,1)
        hour_index = np.linspace(0, 1, 96).reshape(96, 1)

        # === 新增前一天统计特征（均值/最大/最小） ===
        mean_val = np.mean(load_input)
        max_val = np.max(load_input)
        min_val = np.min(load_input)  # 新增最小值特征
        mean_arr = np.full((96, 1), mean_val)
        max_arr = np.full((96, 1), max_val)
        min_arr = np.full((96, 1), min_val)  # 新增最小值数组

        # === 组合所有特征 ===
        feature_list = [load_input, hour_index, mean_arr, max_arr, min_arr]  # 加入最小值特征

        if add_time_features:
            time_input = np.tile(time_features[i - 1], (96, 1))  # (96,2)
            feature_list.insert(1, time_input)  # 插入 weekday/month 第二列

        full_input = np.concatenate(feature_list, axis=1)

        X.append(full_input)
        Y.append(load_scaled[i])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)

    return dataset, scaler
