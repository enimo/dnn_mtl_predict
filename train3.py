import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# 添加数据验证和清洗
def validate_data(data):
    # 检查缺失值
    print("Missing values:\n", data.isnull().sum())
    # 检查异常值
    print("\nFeature statistics:\n", data.describe())
    return data

# 添加特征工程
def feature_engineering(features):
    # 创建数值特征的副本
    numerical_features = features[['chain_cost', 'age', 'familynum']].copy()
    
    # 添加人均成本特征
    numerical_features['cost_per_person'] = features['chain_cost'] / features['familynum']
    
    # 处理分类特征
    categorical_features = features[['gender', 'jiazhao', 'baby']].copy()
    
    # 添加年龄分组作为分类特征
    categorical_features['age_group'] = pd.cut(
        features['age'], 
        bins=[0, 18, 30, 50, 70, 100], 
        labels=['0-18', '19-30', '31-50', '51-70', '70+']
    )
    
    return numerical_features, categorical_features

# 读取数据
data = pd.read_csv('traffic_train_data.csv')

# 添加数据验证和清洗
data = validate_data(data)

# 特征工程
numerical_features, categorical_features = feature_engineering(
    data[['chain_cost', 'gender', 'age', 'jiazhao', 'familynum', 'baby']].copy()
)

# 对分类特征进行独热编码
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(categorical_features)
categorical_feature_names = encoder.get_feature_names_out(categorical_features.columns)

# 合并数值特征和编码后的分类特征
X = np.hstack([numerical_features.values, categorical_encoded])

# 提取标签
labels_mode = data['chain_modepattern'].values - 1
labels_purpose = data['chain_type'].values - 1
labels_stops = np.where(data['stop_num'].values >= 3, 2, data['stop_num'].values - 1)

# 数据分割
X_train, X_temp, y_mode_train, y_mode_temp, y_purpose_train, y_purpose_temp, y_stops_train, y_stops_temp = \
    train_test_split(X, labels_mode, labels_purpose, labels_stops, test_size=0.3, random_state=42)
    
X_val, X_test, y_mode_val, y_mode_test, y_purpose_val, y_purpose_test, y_stops_val, y_stops_test = \
    train_test_split(X_temp, y_mode_temp, y_purpose_temp, y_stops_temp, test_size=0.5, random_state=42)

# 标准化数值特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 保存预处理器
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_mode_train_tensor = torch.tensor(y_mode_train, dtype=torch.long)
y_purpose_train_tensor = torch.tensor(y_purpose_train, dtype=torch.long)
y_stops_train_tensor = torch.tensor(y_stops_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_mode_test_tensor = torch.tensor(y_mode_test, dtype=torch.long)
y_purpose_test_tensor = torch.tensor(y_purpose_test, dtype=torch.long)
y_stops_test_tensor = torch.tensor(y_stops_test, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_mode_train_tensor, y_purpose_train_tensor, y_stops_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_mode_test_tensor, y_purpose_test_tensor, y_stops_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 添加验证集的DataLoader
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_mode_val, dtype=torch.long),
                          torch.tensor(y_purpose_val, dtype=torch.long),
                          torch.tensor(y_stops_val, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义模型
class TrafficModel(nn.Module):
    def __init__(self, input_size, num_mode_classes, num_purpose_classes, num_stops_classes):
        super(TrafficModel, self).__init__()
        
        # 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 出行方式分支
        self.mode_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_mode_classes)
        )
        
        # 出行目的分支
        self.purpose_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_purpose_classes)
        )
        
        # 停留次数分支
        self.stops_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_stops_classes)
        )
        
        # 注意力机制
        self.mode_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.purpose_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.stops_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 共享特征提取
        shared_output = self.shared_layer(x)
        
        # 应用注意力机制
        mode_att = self.mode_attention(shared_output)
        purpose_att = self.purpose_attention(shared_output)
        stops_att = self.stops_attention(shared_output)
        
        # 任务特定的预测
        mode_output = self.mode_layer(shared_output * mode_att)
        purpose_output = self.purpose_layer(shared_output * purpose_att)
        stops_output = self.stops_layer(shared_output * stops_att)
        
        return mode_output, purpose_output, stops_output

# 模型实例化
input_size = X_train.shape[1]
num_mode_classes = 3  # len(np.unique(labels_mode))
num_purpose_classes = 6  # len(np.unique(labels_purpose))
num_stops_classes = 3  # len(np.unique(labels_stops))

model = TrafficModel(input_size, num_mode_classes, num_purpose_classes, num_stops_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

# 添加早停机制
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_losses = {'mode': 0, 'purpose': 0, 'stops': 0}
    
    # 训练阶段
    for batch in train_loader:
        X_batch, y_mode_batch, y_purpose_batch, y_stops_batch = batch
        
        optimizer.zero_grad()
        mode_output, purpose_output, stops_output = model(X_batch)
        
        # 计算各任务损失
        loss_mode = criterion(mode_output, y_mode_batch)
        loss_purpose = criterion(purpose_output, y_purpose_batch)
        loss_stops = criterion(stops_output, y_stops_batch)
        
        # 动态权重平衡
        total_loss = loss_mode + loss_purpose + loss_stops
        
        total_loss.backward()
        optimizer.step()
        
        # 记录损失
        train_losses['mode'] += loss_mode.item()
        train_losses['purpose'] += loss_purpose.item()
        train_losses['stops'] += loss_stops.item()
    
    # 验证阶段
    model.eval()
    val_losses = {'mode': 0, 'purpose': 0, 'stops': 0}
    val_metrics = {'mode': 0, 'purpose': 0, 'stops': 0}
    
    with torch.no_grad():
        for batch in val_loader:
            # 验证逻辑...
            pass
    
    # 更新学习率
    val_total_loss = sum(val_losses.values())
    scheduler.step(val_total_loss)
    
    # 早停检查
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_traffic_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = {'mode': [], 'purpose': [], 'stops': []}
    all_labels = {'mode': [], 'purpose': [], 'stops': []}
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_mode_batch, y_purpose_batch, y_stops_batch = batch
            
            mode_output, purpose_output, stops_output = model(X_batch)
            
            _, mode_preds = torch.max(mode_output, 1)
            _, purpose_preds = torch.max(purpose_output, 1)
            _, stops_preds = torch.max(stops_output, 1)
            
            all_preds['mode'].extend(mode_preds.cpu().numpy())
            all_preds['purpose'].extend(purpose_preds.cpu().numpy())
            all_preds['stops'].extend(stops_preds.cpu().numpy())
            
            all_labels['mode'].extend(y_mode_batch.cpu().numpy())
            all_labels['purpose'].extend(y_purpose_batch.cpu().numpy())
            all_labels['stops'].extend(y_stops_batch.cpu().numpy())
    
    for task in ['mode', 'purpose', 'stops']:
        print(f"\n{task.upper()} Classification Report:")
        
        if task == 'mode':
            target_names = ['Mode 1', 'Mode 2', 'Mode 3']
        elif task == 'purpose':
            target_names = [f'Purpose {i+1}' for i in range(6)]
        else:  # stops
            target_names = ['0 stops', '1 stop', '2+ stops']
        
        # 打印类别分布
        print("True label distribution:")
        print(np.bincount(all_labels[task]))
        print("Predicted label distribution:")
        print(np.bincount(all_preds[task]))
        
        # 使用 zero_division=1 来处理未定义的情况
        print(classification_report(all_labels[task], all_preds[task], 
                                    target_names=target_names, zero_division=1))
        
        print(f"\n{task.upper()} Confusion Matrix:")
        print(confusion_matrix(all_labels[task], all_preds[task]))
        
        accuracy = accuracy_score(all_labels[task], all_preds[task])
        print(f"\n{task.upper()} Accuracy: {accuracy:.4f}")

# 在训练循环结束后调用评估函数
print("\nEvaluating model on test set:")
evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'traffic_model.pth')