# Deep Neural Networks Multi-Task Learning Demo (DNN_MTL_predict)

The goal of this project is to build a multi-task learning neural network model to predict three different tasks related to travel: travel mode, travel purpose, and number of stops. The project uses deep learning techniques and is implemented based on the PyTorch framework.

#### Key Technical Points

1. **Deep Learning**: Uses multi-layer neural networks to learn data representations and patterns.
2. **Multi-Task Learning**: Solves multiple related tasks simultaneously by sharing part of the network layers, improving the model's generalization ability.
3. **Neural Network Layers**: Constructs the neural network using fully connected layers (`nn.Linear`) and activation functions (`nn.ReLU`).
4. **Data Standardization**: Uses `StandardScaler` to standardize the input features.
5. **Model Inference**: Performs model inference in evaluation mode and disables gradient computation to save resources.
6. **Model Saving and Loading**: Uses `torch.save` and `torch.load` to save and load model parameters.

#### Code Structure

1. **Model Definition**: The `TrafficModel` class defines a multi-task learning neural network model.
2. **Loading Model**: Loads the trained model parameters and the standardizer.
3. **Prediction Function**: The `predict_travel` function takes input features, standardizes them, performs prediction through the model, and returns the predictions for the three tasks.

#### Example Input and Output

- **Input Features**: A list of six features, e.g., `[18, 0, 56, 1, 3, 0]`.
- **Output Results**: Three predicted values representing travel mode, travel purpose, and number of stops.
