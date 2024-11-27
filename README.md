# Deep Neural Networks Multi-Task Learning Demo (DNN_MTL_predict)

Actually this project is a MTL assignsment for Dr. Zhang. The goal of this project is to build a multi-task learning neural network model to predict three different tasks related to travel: travel mode, travel purpose, and number of stops. The project uses deep learning techniques and is implemented based on the PyTorch framework.

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

- **Input Features**: A list of six features, e.g., `[4, 1, 37, 0, 2, 1]`.
- **Output Results**: Three predicted values representing travel mode, travel purpose, and number of stops.



### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/enimo/dnn_mtl_predict.git
    cd dnn_mtl_predict
    ```

2. Create a virtual environment and activate it:

    ```bash
    # if you are mac os
	CONDA_SUBDIR=osx-arm64 conda create -n predict_traffic python=3.11
	conda activate predict_traffic
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

Prepare your dataset with the following columns:

- `feature1`
- `feature2`
- `feature3`
- `feature4`
- `feature5`
- `feature6`
- `mode` (target for travel mode)
- `purpose` (target for travel purpose)
- `stops` (target for number of stops)

Ensure your data is in a CSV file format, and put it in current ROOT directory.

### Training

1. Place your dataset in the project directory.

2. Run the training script:

    ```bash
    python train2.py
    ```

    This will train the model and save the trained model parameters to `traffic_model.pth` and the scaler to `scaler.pkl`.

### Testing

1. Ensure you have the trained model (`traffic_model.pth`) and the scaler (`scaler.pkl`) in the project directory.

2. Run the testing script with sample input features:

    ```bash
    python test2.py  --features 4 1 37 0 2 0
    ```

    This will output the predicted travel mode, travel purpose, and number of stops.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
