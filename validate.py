import torch
from torch.utils.data import DataLoader
from preprocessing import preprocess_excel
from model.transformer_model import TransformerModel
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pandas as pd
import matplotlib.pyplot as plt

# === 配置参数 ===
TRAIN_PATH = "train_data.xlsx"
VALIDATION_PATH = "validation_data.xlsx"
MODEL_PATH = "transformer_model.pth"
ADD_TIME_FEATURES = True
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载训练数据集，只为获取 scaler ===
_, train_scaler = preprocess_excel(TRAIN_PATH, add_time_features=ADD_TIME_FEATURES)

# === 加载验证集，使用训练集的 scaler 进行归一化 & 反归一化 ===
val_dataset, _ = preprocess_excel(
    VALIDATION_PATH,
    add_time_features=ADD_TIME_FEATURES,
    external_scaler=train_scaler,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 获取模型输入维度 ===
input_dim = val_dataset[0][0].shape[-1]

# === 加载模型权重 ===
model = TransformerModel(input_dim=input_dim).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# === 验证过程 ===
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, Y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(Y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)         # shape: (N, 96)
all_targets = np.concatenate(all_targets, axis=0)     # shape: (N, 96)

# === 反归一化 ===
all_preds_inv = train_scaler.inverse_transform(all_preds)
all_targets_inv = train_scaler.inverse_transform(all_targets)

# === 误差指标 ===
rmse = math.sqrt(mean_squared_error(all_targets_inv, all_preds_inv))
mae = mean_absolute_error(all_targets_inv, all_preds_inv)
epsilon = 1.0  # 防止 MAPE 炸裂
safe_denominator = np.clip(all_targets_inv, epsilon, None)
mape = np.mean(np.abs((all_targets_inv - all_preds_inv) / safe_denominator)) * 100

# === 输出误差 ===
print("✅ 验证集评估结果：")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# === 保存预测结果，格式与 validation_data.xlsx 一致 ===
val_df = pd.read_excel(VALIDATION_PATH)
pred_dates = pd.to_datetime(val_df['Date'].values[1:])  # 预测从第2天起

columns = ['Date'] + [f'Power{i}' for i in range(1, 97)]
df_out = pd.DataFrame(columns=columns)
df_out['Date'] = pred_dates
df_out.iloc[:, 1:] = all_preds_inv
df_out.to_excel("validation_predictions.xlsx", index=False)
print("📁 预测结果已保存为 validation_predictions.xlsx")

# === 批量绘图查看多个样本预测偏差趋势 ===
num_plots = 5  # 可根据需要调整
for i in range(num_plots):
    plt.figure(figsize=(12, 4))
    plt.plot(range(96), all_targets_inv[i], label="True", linewidth=2)
    plt.plot(range(96), all_preds_inv[i], label="Predicted", linestyle="--")
    plt.title(f"Prediction vs Ground Truth (Day {i+1})")
    plt.xlabel("Time Step (15-min intervals)")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





