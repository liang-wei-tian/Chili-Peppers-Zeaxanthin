import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
import time
import optuna
import json

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# 数据预处理模块
# ----------------------
class SafeScaler:
    """安全标准化处理器"""

    def __init__(self):
        self.spectral_scaler = StandardScaler()
        self.chem_scaler = StandardScaler()
        self.label_scaler = MinMaxScaler()

    def fit(self, spectral, chem, labels):
        valid_ratio = chem[:, 0] / (chem[:, 1] + 1e-6)
        self.valid_chem = np.column_stack([chem[:, 0], valid_ratio])

        self.spectral_scaler.fit(spectral)
        self.chem_scaler.fit(self.valid_chem)
        self.label_scaler.fit(labels.reshape(-1, 1))

    def transform(self, spectral, chem, labels):
        valid_ratio = chem[:, 0] / (chem[:, 1] + 1e-6)
        valid_chem = np.column_stack([chem[:, 0], valid_ratio])

        return (
            self.spectral_scaler.transform(spectral),
            self.chem_scaler.transform(valid_chem),
            self.label_scaler.transform(labels.reshape(-1, 1)))


# ----------------------
# 数据集类
# ----------------------
class SafeDataset(Dataset):
    def __init__(self, spectral, chem, labels, scaler=None):
        if scaler is None:
            self.scaler = SafeScaler()
            self.scaler.fit(spectral, chem, labels)
        else:
            self.scaler = scaler

        self.spectral, self.chem, self.labels = self.scaler.transform(
            spectral, chem, labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.spectral[idx]),
            torch.FloatTensor(self.chem[idx]),
            torch.FloatTensor(self.labels[idx]))

    def __len__(self):
        return len(self.labels)


# ----------------------
# 增强模型架构（支持动态通道数）
# ----------------------
class CNNInteractionNet(nn.Module):
    def __init__(self, spectral_dim=19, chem_dim=2, conv_channels1=16, conv_channels2=32):
        super().__init__()
        # 光谱CNN，通道数可调
        self.spectral_cnn = nn.Sequential(
            nn.Conv1d(1, conv_channels1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels1, conv_channels2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

        # 化学特征网络
        self.chem_net = nn.Sequential(
            nn.Linear(chem_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8))

        # 融合模块
        self.interaction = nn.Sequential(
            nn.Linear(conv_channels2 + 8, 24),
            nn.ReLU(),
            nn.LayerNorm(24))

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1))

        # 梯度可视化
        self.gradients = None
        self.activations = None
        self.spectral_cnn[3].register_forward_hook(self.forward_hook)
        self.spectral_cnn[3].register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, spectral, chem):
        spectral = spectral.unsqueeze(1)
        s_feat = self.spectral_cnn(spectral).squeeze(-1)
        c_feat = self.chem_net(chem)
        combined = torch.cat([s_feat, c_feat], dim=1)
        interacted = self.interaction(combined)
        return self.predictor(interacted)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


# ----------------------
# 训练验证模块
# ----------------------
def loo_cross_validation(X_spectral, X_chem, y_labels, epochs=150):
    """留一交叉验证（使用默认超参数，作为基准评估）"""
    n_samples = X_spectral.shape[0]
    all_trues = []
    all_preds = []

    for i in range(n_samples):
        train_idx = np.array([j for j in range(n_samples) if j != i])
        val_idx = np.array([i])

        train_set = SafeDataset(X_spectral[train_idx], X_chem[train_idx], y_labels[train_idx])
        val_set = SafeDataset(X_spectral[val_idx], X_chem[val_idx], y_labels[val_idx], scaler=train_set.scaler)

        model = CNNInteractionNet().to(device)  # 默认超参数
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.HuberLoss()

        model.train()
        for epoch in range(epochs):
            for spec, chem, lbl in DataLoader(train_set, batch_size=8, shuffle=True):
                spec, chem, lbl = spec.to(device), chem.to(device), lbl.to(device)
                optimizer.zero_grad()
                outputs = model(spec, chem)
                loss = criterion(outputs, lbl)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            spec, chem, lbl = val_set[0]
            pred = model(spec.unsqueeze(0).to(device), chem.unsqueeze(0).to(device)).cpu().numpy()

        true = train_set.scaler.label_scaler.inverse_transform(lbl.numpy().reshape(1, -1))[0][0]
        pred = train_set.scaler.label_scaler.inverse_transform(pred.reshape(1, -1))[0][0]
        all_trues.append(true)
        all_preds.append(pred)
        print(f"Sample {i + 1}/{n_samples} | True: {true:.4f} | Pred: {pred:.4f}")

    return {
        'Trues': all_trues,
        'Preds': all_preds,
        'RMSE': np.sqrt(mean_squared_error(all_trues, all_preds)),
        'R2': r2_score(all_trues, all_preds),
        'MAE': mean_absolute_error(all_trues, all_preds)}


def train_model(train_set, val_set=None, epochs=300, **kwargs):
    """通用训练函数，支持验证集和超参数"""
    model = CNNInteractionNet(
        conv_channels1=kwargs.get('conv_channels1', 16),
        conv_channels2=kwargs.get('conv_channels2', 32)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=kwargs.get('lr', 1e-4),
                                  weight_decay=kwargs.get('weight_decay', 1e-5))
    criterion = nn.HuberLoss()
    batch_size = kwargs.get('batch_size', 16)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_set:
        val_loader = DataLoader(val_set, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for spec, chem, lbl in train_loader:
            spec, chem, lbl = spec.to(device), chem.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(spec, chem)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()

        # 可选：打印验证损失
        if val_set and (epoch + 1) % 50 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for spec, chem, lbl in val_loader:
                    spec, chem, lbl = spec.to(device), chem.to(device), lbl.to(device)
                    outputs = model(spec, chem)
                    val_loss += criterion(outputs, lbl).item()
            print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")

    return model


def enhanced_evaluate(model, dataset, scaler):
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    trues = []
    preds = []

    with torch.no_grad():
        for spectral, chem, labels in loader:
            spectral = spectral.to(device)
            chem = chem.to(device)
            outputs = model(spectral, chem)
            preds.extend(outputs.cpu().numpy().flatten())
            trues.extend(labels.cpu().numpy().flatten())

    trues = scaler.label_scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    preds = scaler.label_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    return {
        'RMSE': np.sqrt(mean_squared_error(trues, preds)),
        'R2': r2_score(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'Trues': trues,
        'Preds': preds}


# ----------------------
# 超参数搜索目标函数
# ----------------------
def objective(trial, X_train_spec, X_train_chem, y_train):
    # 建议超参数
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    conv_channels1 = trial.suggest_int('conv_channels1', 8, 32, step=8)
    conv_channels2 = trial.suggest_int('conv_channels2', 16, 64, step=8)

    # 从训练集中划分出验证集（20%）
    dataset = SafeDataset(X_train_spec, X_train_chem, y_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # 训练模型
    model = train_model(
        train_subset, val_subset,
        epochs=100,  # 快速搜索
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        conv_channels1=conv_channels1,
        conv_channels2=conv_channels2
    )

    # 在验证集上评估
    val_metrics = enhanced_evaluate(model, val_subset, dataset.scaler)

    return val_metrics['R2']


# ----------------------
# 可视化与预测保存模块
# ----------------------
def generate_gradcam(model, spectral_input, chem_input, wavelength_range=(365, 970)):
    model.eval()
    spectral = spectral_input.unsqueeze(0).to(device)
    chem = chem_input.unsqueeze(0).to(device)

    pred = model(spectral, chem)
    pred.backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations()

    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    activations = activations.squeeze().cpu().detach().numpy()

    for i in range(activations.shape[0]):
        activations[i, :] *= pooled_gradients[i].cpu().numpy()
    heatmap = np.mean(activations, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.interp(np.linspace(0, len(heatmap) - 1, 100),
                        np.arange(len(heatmap)), heatmap)

    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], len(heatmap))

    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, heatmap, color='darkred', linewidth=2)
    plt.fill_between(wavelengths, heatmap, alpha=0.3, color='indianred')
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Activation Intensity", fontsize=12)
    plt.title("Grad-CAM Spectral Sensitivity", fontsize=14)
    plt.grid(alpha=0.2)
    plt.savefig("grad_cam_spectral.png", dpi=300, bbox_inches='tight')
    plt.close()

    return heatmap


def save_predictions(dataset, model, scaler, filename="train_predictions.csv"):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    trues = []
    preds = []

    with torch.no_grad():
        for spectral, chem, labels in loader:
            spectral, chem = spectral.to(device), chem.to(device)
            outputs = model(spectral, chem).cpu().numpy().flatten()
            trues.extend(labels.numpy().flatten())
            preds.extend(outputs)

    trues = scaler.label_scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    preds = scaler.label_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    result_df = pd.DataFrame({
        "True_Value": trues,
        "Predicted_Value": preds,
        "Absolute_Error": np.abs(trues - preds)
    })
    result_df.to_csv(filename, index=False)
    print(f"\n预测结果已保存至 {filename}")

    print("\n=== 训练集预测性能 ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(trues, preds)):.4f}")
    print(f"R²: {r2_score(trues, preds):.4f}")
    print(f"MAE: {mean_absolute_error(trues, preds):.4f}")
    return result_df


# ----------------------
# 主流程（训练 + 超参数搜索 + LOOCV）
#
# if __name__ == "__main__":
#     # 数据加载
#     train_df = pd.read_csv('C://Users//25726//Desktop//光谱液相//CNN//特征训练集.csv', header=None)
#     test_df = pd.read_csv('C://Users//25726//Desktop//光谱液相//CNN//特征测试集.csv', header=None)
#
#     # 特征提取（根据实际列索引修改）
#     X_train_spec = train_df.iloc[:, :5].values.astype(np.float32)
#     X_train_chem = train_df.iloc[:, [8, 9]].values.astype(np.float32)  # 辣椒红和总类
#     y_train = train_df.iloc[:, 10].values.astype(np.float32)  # 玉米黄质
#
#     X_test_spec = test_df.iloc[:, :5].values.astype(np.float32)
#     X_test_chem = test_df.iloc[:, [8, 9]].values.astype(np.float32)
#     y_test = test_df.iloc[:, 10].values.astype(np.float32)
#
#     # ---------- 1. 执行留一交叉验证----------
#     print("\n正在进行留一交叉验证...")
#     loo_results = loo_cross_validation(X_train_spec, X_train_chem, y_train)
#     print("\n=== 训练集LOOCV结果 ===")
#     print(f"RMSE: {loo_results['RMSE']:.4f}")
#     print(f"R²: {loo_results['R2']:.4f}")
#     print(f"MAE: {loo_results['MAE']:.4f}")
#
#     # ---------- 2. 超参数搜索 ----------
#     print("\n开始超参数搜索（Optuna）...")
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: objective(trial, X_train_spec, X_train_chem, y_train), n_trials=20)
#
#     print("\n最佳超参数:")
#     best_params = study.best_params
#     print(best_params)
#     print(f"最佳验证R²: {study.best_value:.4f}")
#
#     # 保存最佳超参数到文件
#     with open('best_params.json1', 'w') as f:
#         json.dump(best_params, f)
#
#     # ---------- 3. 使用最佳超参数重新训练最终模型 ----------
#     print("\n使用最佳超参数训练最终模型...")
#     full_train_set = SafeDataset(X_train_spec, X_train_chem, y_train)
#     test_set = SafeDataset(X_test_spec, X_test_chem, y_test, scaler=full_train_set.scaler)
#
#     final_model = train_model(
#         full_train_set, val_set=None,
#         epochs=300,
#         **best_params
#     )
#
#     # 保存模型和scaler
#     torch.save(final_model.state_dict(), 'final_model1.pth')
#     with open('final_scaler1.pkl', 'wb') as f:
#         pickle.dump(full_train_set.scaler, f)
#     print("\n模型和标准化参数已保存至 final_model1.pth 和 final_scaler1.pkl")
#
#     # 测试集评估
#     test_metrics = enhanced_evaluate(final_model, test_set, full_train_set.scaler)
#
#     # 提取真实值和预测值用于RPD计算
#     trues = np.array(test_metrics['Trues'])
#     preds = np.array(test_metrics['Preds'])
#     sd_true = np.std(trues, ddof=1)
#     rmse = test_metrics['RMSE']
#     rpd = sd_true / rmse
#
#     print("\n=== 测试集性能 ===")
#     print(f"RMSE: {test_metrics['RMSE']:.4f}")
#     print(f"R²: {test_metrics['R2']:.4f}")
#     print(f"MAE: {test_metrics['MAE']:.4f}")
#     print(f"RPD: {rpd:.4f}")
#
#     # 保存训练集预测结果
#     save_predictions(full_train_set, final_model, full_train_set.scaler)
#
#     # 可视化示例
#     sample_idx = 0
#     spectral, chem, _ = full_train_set[sample_idx]
#     generate_gradcam(final_model, spectral, chem, wavelength_range=(365, 970))
#
#     # 化学特征权重分析
#     chem_weights = final_model.chem_net[0].weight.detach().cpu().numpy()
#     print("\n化学特征权重分析：")
#     print(f"辣椒红素权重: {chem_weights[:, 0].mean():.3f} ± {chem_weights[:, 0].std():.3f}")
#     print(f"辣椒红占比权重: {chem_weights[:, 1].mean():.3f} ± {chem_weights[:, 1].std():.3f}")
#
#     # 打印预测明细
#     print("\n测试集预测明细:")
#     for idx, (true, pred) in enumerate(zip(test_metrics['Trues'], test_metrics['Preds'])):
#         print(f"样本{idx + 1} | 真实值: {true:.4f} | 预测值: {pred:.4f} | 误差: {abs(true - pred):.4f}")



import json
import numpy as np
import torch
import pandas as pd
import pickle
import time
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数统计函数
def count_parameters(model):
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. 加载最佳超参数
with open('MCSF-character.json1', 'r') as f:
    best_params = json.load(f)
print("加载的最佳超参数：", best_params)

# 2. 使用最佳超参数实例化模型
model = CNNInteractionNet(
    conv_channels1=best_params['conv_channels1'],
    conv_channels2=best_params['conv_channels2']
).to(device)

# 3. 加载模型权重
model.load_state_dict(torch.load('MCSF-character.pth'))
model.eval()
print("模型权重加载成功。")

# 4. 计算参数量
param_count = count_parameters(model)
print(f"模型参数量: {param_count:,}")

# 5. 加载标准化器
with open('MCSF-character.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("标准化器加载成功。")

# 6. 加载测试数据
test_df = pd.read_csv('C://Users//25726//Desktop//光谱液相//CNN//特征测试集.csv', header=None)

# 提取特征（必须与训练时使用的列索引一致）
X_test_spec = test_df.iloc[:, :5].values.astype(np.float32)
X_test_chem = test_df.iloc[:, [8, 9]].values.astype(np.float32)
y_test = test_df.iloc[:, 10].values.astype(np.float32)

# 7. 创建测试集 Dataset
test_set = SafeDataset(X_test_spec, X_test_chem, y_test, scaler=scaler)

# 8. 计算推理时间（单样本平均）
print("\n正在计算推理时间...")
model.eval()
with torch.no_grad():
    # GPU预热
    dummy_spec = torch.randn(1, 5).to(device)  # 5个特征波段
    dummy_chem = torch.randn(1, 2).to(device)
    for _ in range(50):
        _ = model(dummy_spec, dummy_chem)

    total_time = 0.0
    n_samples = X_test_spec.shape[0]
    for i in range(n_samples):
        spec = torch.FloatTensor(X_test_spec[i:i+1]).to(device)
        chem = torch.FloatTensor(X_test_chem[i:i+1]).to(device)
        start = time.time()
        _ = model(spec, chem)
        end = time.time()
        total_time += (end - start)
    avg_inference_time = total_time / n_samples * 1000  # 转换为毫秒
    print(f"平均推理时间: {avg_inference_time:.3f} ms/样本")

# 9. 评估性能
def enhanced_evaluate(model, dataset, scaler):
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    trues = []
    preds = []
    with torch.no_grad():
        for spectral, chem, labels in loader:
            spectral = spectral.to(device)
            chem = chem.to(device)
            outputs = model(spectral, chem)
            preds.extend(outputs.cpu().numpy().flatten())
            trues.extend(labels.cpu().numpy().flatten())
    trues = scaler.label_scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    preds = scaler.label_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return {
        'RMSE': np.sqrt(mean_squared_error(trues, preds)),
        'R2': r2_score(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'Trues': trues,
        'Preds': preds
    }

test_metrics = enhanced_evaluate(model, test_set, scaler)

# 10. 计算RPD
trues = test_metrics['Trues']
preds = test_metrics['Preds']
sd_true = np.std(trues, ddof=1)
rmse = test_metrics['RMSE']
rpd = sd_true / rmse

# 11. 输出结果
print("\n=== 测试集性能（优化后模型）===")
print(f"参数量: {param_count:,}")
print(f"平均推理时间: {avg_inference_time:.3f} ms/样本")
print(f"RMSE: {test_metrics['RMSE']:.4f}")
print(f"R²: {test_metrics['R2']:.4f}")
print(f"MAE: {test_metrics['MAE']:.4f}")
print(f"RPD: {rpd:.4f}")

print("\n测试集预测明细:")
for idx, (true, pred) in enumerate(zip(trues, preds)):
    print(f"样本{idx + 1} | 真实值: {true:.4f} | 预测值: {pred:.4f} | 误差: {abs(true - pred):.4f}")