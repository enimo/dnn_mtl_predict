import torch
import joblib
import numpy as np
import torch.nn as nn
import pandas as pd
import argparse

# 定义模型
class TrafficModel(nn.Module):
    def __init__(self, input_size, num_mode_classes, num_purpose_classes, num_stops_classes):
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
        self.stops_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_stops_classes)
        )

    def forward(self, x):
        shared_output = self.shared_layer(x)
        mode_output = self.mode_layer(shared_output)
        purpose_output = self.purpose_layer(shared_output)
        stops_output = self.stops_layer(shared_output)
        return mode_output, purpose_output, stops_output

# 加载训练好的模型
input_size = 6  # 替换为实际的输入特征维度
num_mode_classes = 3  # 替换为实际的出行方式类别数
num_purpose_classes = 6  # 替换为实际的出行目的类别数
num_stops_classes = 3  # 替换为实际的停留次数类别数

model = TrafficModel(input_size, num_mode_classes, num_purpose_classes, num_stops_classes)
model.load_state_dict(torch.load('traffic_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# 加载标准化器
scaler = joblib.load('scaler.pkl')

# 定义预测函数
def predict_travel(features):
    # 将输入特征转换为numpy数组并进行标准化
    # features = np.array(features).reshape(1, -1)
    # features = scaler.transform(features)
    

     # 将输入特征转换为pandas DataFrame并进行标准化
    feature_names = ['chain_cost', 'gender', 'age', 'jiazhao', 'familynum', 'baby']  # 替换为实际的特征名称
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)


    # 转换为PyTorch张量
    # features_tensor = torch.tensor(features, dtype=torch.float32)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    
    # 禁用梯度计算
    with torch.no_grad():
        # 模型推理
        mode_output, purpose_output, stops_output = model(features_tensor)
        
        # 获取预测结果
        _, mode_pred = torch.max(mode_output, 1)
        _, purpose_pred = torch.max(purpose_output, 1)
        _, stops_pred = torch.max(stops_output, 1)
        
        # 将预测结果转换为numpy数组
        mode_pred = mode_pred.numpy()[0]
        purpose_pred = purpose_pred.numpy()[0]
        stops_pred = stops_pred.numpy()[0]
    
    return mode_pred, purpose_pred, stops_pred


# 示例输入特征
# 17.857914  0   56  1   3     0
# input_features = [18, 0, 56, 1, 3, 0]  # 1-1-1
# input_features = [4, 1, 37, 0, 2, 1] # 3-3-1 ok
# input_features = [7, 0, 50, 1, 2, 0] # 1-1-1
# input_features = [4, 0, 36, 1, 4, 1] # 1-1-3
# input_features = [4, 1, 37, 0, 2, 0] # 1-1-3 ok



# 获取预测结果
# mode_pred, purpose_pred, stops_pred = predict_travel(input_features)

# 打印预测结果
# print(f'Predicted Number of Stops: {stops_pred+1}')
# print(f'Predicted Travel Purpose-chain_type: {purpose_pred+1}')
# print(f'Predicted Travel Mode-chain_modepattern: {mode_pred+1}')



def main():
    parser = argparse.ArgumentParser(description='Predict travel mode, purpose, and number of stops.')
    parser.add_argument('--features', nargs=6, type=float, required=True, help='Input features for prediction')
    args = parser.parse_args()
    
    features = args.features
    
    # 获取预测结果
    mode_pred, purpose_pred, stops_pred = predict_travel(features)
    
    # 打印预测结果
    print(f'Predicted Number of Stops: {stops_pred + 1}')
    print(f'Predicted Travel Purpose-chain_type: {purpose_pred + 1}')
    print(f'Predicted Travel Mode-chain_modepattern: {mode_pred + 1}')

if __name__ == '__main__':
    main()