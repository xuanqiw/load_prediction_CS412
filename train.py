import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from preprocessing import preprocess_excel
from model.transformer_model import TransformerModel
import numpy as np

# === 配置超参数 ===
EPOCHS = 300
BATCH_SIZE = 32             # 增大批量大小
LEARNING_RATE = 0.001       # 增大学习率
WEIGHT_DECAY = 0.005        # 减小L2正则化强度
DATA_PATH = "train_data.xlsx"
ADD_TIME_FEATURES = True
MODEL_SAVE_PATH = "transformer_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 加载完整数据集 ===
print("📥 加载数据中...")
dataset, scaler = preprocess_excel(DATA_PATH, add_time_features=ADD_TIME_FEATURES)

# === 划分训练集和验证集 ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === 构造模型 ===
input_dim = dataset[0][0].shape[-1]
model = TransformerModel(
    input_dim=input_dim,
    d_model=128,        # 增加到128
    nhead=8,           # 增加到8
    num_layers=3,      # 增加到3
    dim_feedforward=256, # 增加到256
    dropout=0.2        # 适当减小dropout
).to(DEVICE)

# === 优化器和学习率调度器 ===
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)




# === 自定义复合损失函数 ===
def composite_loss(pred, target, alpha=0.6, beta=0.4, epsilon=1e-3):
    # 增加MSE的权重
    mse = F.mse_loss(pred, target)
    denom = torch.clamp(torch.abs(target), min=epsilon)
    mape = torch.mean(torch.abs((pred - target) / denom))
    return alpha * mse + beta * mape



# === 评估函数 ===
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            preds = model(X)
            loss = composite_loss(preds, Y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(data_loader.dataset)


# === 训练过程 ===
best_val_loss = float("inf")
patience = 15  # 减少早停轮数
patience_counter = 0
prev_lr = LEARNING_RATE
train_losses = []
val_losses = []

print("🚀 开始训练...")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = composite_loss(preds, Y_batch)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # 验证阶段
    avg_val_loss = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"训练损失: {avg_train_loss:.6f}")
    print(f"验证损失: {avg_val_loss:.6f}")

    # 更新学习率
    scheduler.step(avg_val_loss)

    # 检查学习率变化
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f"📉 Learning rate adjusted to: {current_lr:.6f}")
        prev_lr = current_lr

    # 早停检查
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ 最佳模型已保存，验证损失={best_val_loss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⚠️ 早停触发！{patience} 个 epoch 验证损失未改善")
            break

print("🎉 训练完成。模型保存至:", MODEL_SAVE_PATH)

# === 绘制损失曲线 ===
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()