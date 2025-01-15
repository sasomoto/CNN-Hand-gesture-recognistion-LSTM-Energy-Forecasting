# CNN-Hand-gesture-recognistion-LSTM-Energy-Forecasting
Machine Learning Assignment 5
TEAM MEMBERS
Dhairya Luthra(2022A7PS1377H)
Shashwat Sharma(2022AAPS0508H)
Animesh Agrahari(2022A7PS1367H)
Utkarsh Singhal (2022A7PS1334H)
This notebook outlines the tasks required to complete two deep learning projects:

Hand Gesture Recognition using Convolutional Neural Networks (CNN)
Time Series Forecasting (Energy Consumption Forecasting) using Long Short-Term Memory Networks (LSTM)
Problem 1: Hand Gesture Recognition using Convolutional Neural Network (CNN)
Objective
Develop a Convolutional Neural Network (CNN) model to accurately classify American Sign Language (ASL) hand gestures representing letters A-Z based on image inputs.

Background
American Sign Language (ASL) is a comprehensive language that utilizes hand signs, facial expressions, and body postures. Automating the recognition of ASL gestures can significantly enhance communication accessibility for individuals with hearing impairments. CNNs are well-suited for this image classification task due to their ability to extract hierarchical features from images.

Problem Statement
Create a CNN model to classify sign language letters from grayscale images of hand gestures. The dataset is structured similarly to the classic MNIST dataset, with each image labeled from 0-25 corresponding to letters A-Z (excluding J and Z due to gesture motions).

Tasks
1. Setup and Import Libraries
Import necessary libraries for data processing, model building, and visualization.
2. Data Loading and Preprocessing
Download and Extract Dataset: Ensure the dataset is downloaded and extracted properly.
Load Images and Labels: Load the image data and corresponding labels into appropriate data structures.
Normalize Image Data: Scale pixel values to a suitable range (e.g., [0, 1]).
One-Hot Encode Labels: Convert integer labels to one-hot encoded vectors.
Split Dataset: Divide the dataset into training and testing sets (e.g., 80% training, 20% testing).
3. Exploratory Data Analysis (EDA)
Visualize Sample Images: Display a subset of images from each class to understand the data.
Analyze Class Distribution: Check for class imbalance and address if necessary.
4. Build CNN Architecture
Design the CNN Model:
First Convolutional Layer: Apply a set of learnable filters to extract initial features.
Second Convolutional Layer: Apply another set of filters to capture more abstract features.
Activation Function: Use ReLU or Sigmoid to introduce non-linearity.
Max Pooling Layer: Down-sample feature maps to reduce spatial dimensions.
Flatten Layer: Convert 2D feature maps to a 1D feature vector.
Fully Connected (Dense) Layer: Integrate extracted features for classification.
Dropout Layer: Apply dropout (e.g., rate 0.28) to prevent overfitting.
Output Layer: Use a softmax activation with 26 nodes for classification.
5. Model Visualization
Create a visual representation of the CNN architecture to illustrate the data flow through the layers.
6. Compile the Model
Optimizer: Choose Adam or AdamW.
Loss Function: Use categorical crossentropy.
Metrics: Track accuracy during training.
7. Train the Model
Epochs: Train the model for 20 epochs.
Batch Size: Select an appropriate batch size (e.g., 32).
Validation Split: Allocate a portion of the training data for validation (e.g., 20%).
8. Evaluate the Model
Plot Training and Validation Metrics: Visualize loss and accuracy over epochs.
Generate Classification Report: Assess precision, recall, and F1-score.
Confusion Matrix: Visualize the performance across different classes.
9. Analyze Overfitting
Identify Overfitting Signs: Compare training and validation performance.
Mitigation Strategies: Discuss how CNN handles overfitting and experiment with additional dropout layers (e.g., rate 0.4).
10. Discuss Image Classification Properties
Share Structure Property: Explain how weight sharing in CNNs aids in feature detection.
Invariance Property: Describe how the model achieves invariance to transformations like translation and scaling.
11. Save Observations
Document observations on model performance, overfitting, and how image classification properties are handled. Save these insights in a text file.
Problem 2: Time Series Forecasting (Energy Consumption Forecasting) using LSTMs
Objective
Implement an LSTM-based deep learning model to forecast daily energy consumption for a household, aiding in optimizing energy usage and reducing electric bills.

Background
Energy consumption data often exhibits complex temporal dependencies and recurring patterns. LSTMs are effective in capturing both short-term and long-term dependencies, making them suitable for forecasting tasks.

Tasks
1. Setup and Import Libraries
Import necessary libraries for data processing, model building, and visualization.
2. Data Loading and Preprocessing
Load Dataset: Read the dataset into a suitable data structure.
Handle Missing Values: Identify and manage any missing or anomalous data.
Feature Selection: Select relevant features for forecasting (e.g., Global_active_power).
Resample Data: Aggregate data to obtain daily energy consumption values.
Normalize Features: Scale features using appropriate normalization techniques (e.g., MinMaxScaler).
3. Exploratory Data Analysis (EDA)
Visualize Energy Consumption Trends: Plot daily energy consumption over time.
Identify Patterns and Seasonality: Detect recurring patterns, trends, and seasonal effects.
Detect Anomalies: Identify and analyze any anomalies or outliers in the data.
4. Prepare Data for LSTM
Create Time Series Sequences: Define a window size (e.g., past 30 days) to create input sequences for the model.
Split into Training and Testing Sets: Typically 80% training and 20% testing.
Reshape Data: Reshape data to fit the LSTM input requirements (samples, timesteps, features).
5. Build LSTM Model
Design the LSTM Architecture:
First LSTM Layer: Incorporate LSTM units to capture temporal dependencies.
Dropout Layers: Apply dropout (e.g., rate 0.2) to prevent overfitting.
Second LSTM Layer: Add additional LSTM units for deeper temporal feature extraction.
Dense Output Layer: Use a single neuron for forecasting the next day's energy consumption.
6. Compile the Model
Optimizer: Choose Adam.
Loss Function: Use Mean Squared Error (MSE).
Metrics: Track Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
7. Train the Model
Epochs: Train the model for an appropriate number of epochs (e.g., 50).
Batch Size: Select a suitable batch size (e.g., 32).
Validation Split: Allocate a portion of the training data for validation (e.g., 20%).
8. Evaluate the Model
Forecasting: Predict energy consumption on the test set.
Calculate Metrics: Compute MAE and RMSE to assess model performance.
Visualization: Plot forecasted vs. actual energy usage to visually evaluate accuracy.
9. Save Observations
Document findings, model performance, and potential improvements. Save these insights in a text file.
