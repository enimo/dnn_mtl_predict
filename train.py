import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import joblib

# 读取数据
data = pd.read_csv('traffic_train_data.csv')

# 假设特征列为 'feature1', 'feature2', ..., 'featureN'
# 标签列为 'mode' 和 'purpose'
features = data[['chain_cost', 'gender', 'age', 'jiazhao', 'familynum', 'baby']]
# labels_mode = data['mode']
# labels_purpose = data['purpose']

# labels_mode2 = data['stop_num']
labels_mode = data['chain_modepattern']-1 #3
labels_purpose = data['chain_type']-1 #6

# 数据分割
X_train, X_test, y_mode_train, y_mode_test, y_purpose_train, y_purpose_test = train_test_split(
    features, labels_mode, labels_purpose, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 保存标准化器到文件
joblib.dump(scaler, 'scaler.pkl')


# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_mode_train_tensor = torch.tensor(y_mode_train.values, dtype=torch.long)
y_purpose_train_tensor = torch.tensor(y_purpose_train.values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_mode_test_tensor = torch.tensor(y_mode_test.values, dtype=torch.long)
y_purpose_test_tensor = torch.tensor(y_purpose_test.values, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_mode_train_tensor, y_purpose_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_mode_test_tensor, y_purpose_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
class TrafficModel(nn.Module):
    def __init__(self, input_size, num_mode_classes, num_purpose_classes):
        super(TrafficModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mode_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_mode_classes)
        )
        self.purpose_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_purpose_classes)
        )

    def forward(self, x):
        shared_output = self.shared_layer(x)
        mode_output = self.mode_layer(shared_output)
        purpose_output = self.purpose_layer(shared_output)
        return mode_output, purpose_output

# 模型实例化
input_size = X_train.shape[1]
num_mode_classes = 3 #len(labels_mode.unique())
num_purpose_classes = 6 #len(labels_purpose.unique())

model = TrafficModel(input_size, num_mode_classes, num_purpose_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_mode_batch, y_purpose_batch in train_loader:
        optimizer.zero_grad()
        mode_output, purpose_output = model(X_batch)
        loss_mode = criterion(mode_output, y_mode_batch)
        loss_purpose = criterion(purpose_output, y_purpose_batch)
        loss = loss_mode + loss_purpose
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 评估模型
model.eval()
mode_preds = []
purpose_preds = []
mode_labels = []
purpose_labels = []

with torch.no_grad():
    for X_batch, y_mode_batch, y_purpose_batch in test_loader:
        mode_output, purpose_output = model(X_batch)
        _, mode_pred = torch.max(mode_output, 1)
        _, purpose_pred = torch.max(purpose_output, 1)
        
        mode_preds.extend(mode_pred.numpy())
        purpose_preds.extend(purpose_pred.numpy())
        mode_labels.extend(y_mode_batch.numpy())
        purpose_labels.extend(y_purpose_batch.numpy())

# 计算准确率
mode_accuracy = accuracy_score(mode_labels, mode_preds)
purpose_accuracy = accuracy_score(purpose_labels, purpose_preds)

print(f'Mode Prediction Accuracy: {mode_accuracy:.4f}')
print(f'Purpose Prediction Accuracy: {purpose_accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'traffic_model.pth')



# 加载模型
# model = TrafficModel(input_size, num_mode_classes, num_purpose_classes)
# model.load_state_dict(torch.load('traffic_model.pth'))
# model.eval()



