import torch
import pandas as pd
import numpy as np
from preprocessing import preprocess_excel
from model.transformer_model import TransformerModel

# === 配置 ===
DATA_PATH = "train_data.xlsx"
MODEL_PATH = "transformer_model.pth"
OUTPUT_PATH = "check_predictions.xlsx"
ADD_TIME_FEATURES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载全部数据集（含 scaler）===
full_dataset, scaler = preprocess_excel(DATA_PATH, add_time_features=ADD_TIME_FEATURES)

# 检查集范围：从 2024-06-01 开始（从第879天开始）共184天
check_range = range(879, 879 + 184)
print(f"🔍 准备预测检查集，共 {len(check_range)} 天")

# === 加载模型 ===
input_dim = full_dataset[0][0].shape[-1]
model = TransformerModel(input_dim=input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === 开始逐天预测 ===
predictions = []

with torch.no_grad():
    for idx in check_range:
        x = full_dataset[idx - 1][0].unsqueeze(0).to(DEVICE)  # shape: (1, 96, input_dim)
        pred = model(x)  # shape: (1, 96)
        pred_np = pred.squeeze(0).cpu().numpy().reshape(1, -1)
        pred_inv = scaler.inverse_transform(pred_np)  # 反归一化
        predictions.append(pred_inv.flatten())  # shape: (96,)

# === 构造 DataFrame，输出为 Excel ===
columns = [f"Power{i}" for i in range(1, 97)]
df_output = pd.DataFrame(predictions, columns=columns)
df_output.to_excel(OUTPUT_PATH, index=False)

print(f"✅ 检查集预测完成，结果已保存到: {OUTPUT_PATH}")
