# main.py

import os
import sys
import torch

# === 添加当前路径（若直接运行 main.py）===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_excel
from model.transformer_model import TransformerModel
import train
import validate
'''import check_predict'''

DATA_PATH = "train_data.xlsx"
ADD_TIME_FEATURES = True


def run_preprocessing():
    print("📦 Step 1: Data Preprocessing")
    dataset, scaler = preprocess_excel(DATA_PATH, add_time_features=ADD_TIME_FEATURES)
    print(f"✅ 预处理完成，共生成样本数: {len(dataset)}")
    return dataset, scaler



def run_training():
    print("\n🧠 Step 2: Model Training")
    train  # train.py 脚本直接运行时会执行训练流程
    print("✅ 模型训练完成")


def run_validation():
    print("\n🧪 Step 3: Model Validation")
    validate  # validate.py 脚本直接运行时会打印验证结果
    print("✅ 验证完成")


def run_prediction_check():
    print("\n🔍 Step 4: Prediction on Check Set")
    check_predict  # check_predict.py 脚本直接运行时会执行预测
    print("✅ 检查集预测完成")


def main():
    print("🚀 启动全流程：数据预处理 → 训练 → 验证 → 预测检查")

    run_preprocessing()
    run_training()

    run_validation()
    '''
    run_prediction_check()
    '''

    print("\n🎉 全部流程完成！")


if __name__ == "__main__":
    main()
