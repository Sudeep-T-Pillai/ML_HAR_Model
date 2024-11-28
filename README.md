# Human Activity Recognition (HAR) LSTM Model

## Overview
This project implements a **Human Activity Recognition (HAR)** model using accelerometer data. It builds and trains a Long Short-Term Memory (LSTM)-based neural network to classify human activities such as walking, jogging, and others. The implementation is written in Python and utilizes libraries like TensorFlow, NumPy, Pandas, and Scikit-learn.

## Features
* **Data Preprocessing**: Reads and cleans accelerometer data, preparing it for input into the neural network.
* **LSTM Model Creation**: Implements a two-layer LSTM model for time-series classification.
* **Model Training and Evaluation**: Trains the model, evaluates performance, and plots accuracy and loss metrics.
* **Model Export**: Saves the model in TensorFlow, TensorFlow Lite (TFLite), and includes metadata for further use.

## Requirements
Install the following Python libraries before running the code:

```bash
pip install pandas numpy tensorflow sklearn seaborn matplotlib scipy pickle tflite-support
```

## File Structure
* **Input Data**: `WISDM_ar_v1.1_raw.txt` (Accelerometer data file, assumed to be in a Google Drive folder).
* **Output Directories**:
   * `/content/checkpoint`: Stores the TensorFlow model checkpoints.
   * `/content/saved_model`: Contains the final saved models and metadata.
* **Generated Files**:
   * `HAR.pbtxt`: Protobuf file of the model graph.
   * `HAR.ckpt`: Checkpoint file for TensorFlow.
   * `HAR.tflite`: TensorFlow Lite version of the model.
   * `HAR_metadata.json`: Metadata about the saved TFLite model.
   * `history.p`: Training history for plotting and analysis.
   * `prediction.p`: Model predictions on the test set.

## How It Works

### 1. Data Loading and Preprocessing
* Loads accelerometer data from `WISDM_ar_v1.1_raw.txt`.
* Cleans non-numeric characters and converts the accelerometer axes (`x-axis`, `y-axis`, `z-axis`) into numeric form.
* Segments the data into chunks of 200-time steps with a sliding window approach and assigns a mode-based label for each segment.
* Converts labels into one-hot encoding format.

### 2. Model Definition
* Creates a two-layer LSTM neural network with the following:
   * **Input**: 200 time steps, 3 features (x, y, z accelerometer data).
   * **Hidden Layers**: Two LSTM layers with 64 hidden units each.
   * **Output Layer**: 6-class softmax for activity classification.

### 3. Training and Evaluation
* Splits the dataset into training (80%) and testing (20%) sets.
* Trains the model using the Adam optimizer for 50 epochs with a batch size of 1024.
* Computes training and testing accuracy and loss at each epoch.
* Saves the history of metrics for visualization.

### 4. Model Export
* Exports the trained model in:
   * **TensorFlow format** (`HAR.pbtxt`, `HAR.ckpt`).
   * **TensorFlow Lite format** (`HAR.tflite`) for edge-device deployment.
   * **Metadata**: JSON file containing information about the model input, output, and details.

### 5. Visualization
* Plots the training and testing loss and accuracy over epochs using Matplotlib.

### 6. Deployment
* The TensorFlow Lite (TFLite) model and metadata allow deployment on mobile or embedded devices.
* Metadata includes details about input shape (1, 200, 3), output shape (1, 6), and data types.

## Usage Instructions

### Run the Code
1. Clone or copy the code into a Jupyter Notebook or Google Colab environment.
2. Upload the dataset (`WISDM_ar_v1.1_raw.txt`) to your Google Drive.
3. Update the file path in `pd.read_csv` to point to your dataset location.
4. Execute the notebook cells sequentially to:
   * Preprocess data.
   * Train the LSTM model.
   * Evaluate performance.
   * Save the trained model.

### Visualize Training Progress
* Check the loss and accuracy curves plotted towards the end of the training.

### Deploy the Model
* Use the generated `HAR.tflite` model and `HAR_metadata.json` for deployment in TFLite-compatible environments.

## Outputs and Results
* **Training and Testing Metrics**: Accuracy and loss values logged for each epoch.
* **Final Model Performance**:
   * Final Test Accuracy: Displayed in the terminal after training.
   * Final Test Loss: Displayed in the terminal after training.
* **Saved Files**:
   * `HAR.tflite` and `HAR_metadata.json` for lightweight deployment.
   * Training history and predictions saved for further analysis.
