
# Podcast Listening Time Prediction

This project aims to predict the **listening time** (in minutes) of podcast episodes using machine learning techniques. The goal is to develop a model that accurately predicts how much time users spend listening to a particular podcast based on several features of the podcast episode.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The project uses a **Random Forest Regressor** model to predict podcast listening times. The data consists of multiple features such as episode name, genre, and sentiment, along with other attributes that might influence how long a listener stays engaged with the podcast episode.

The model is evaluated using **Root Mean Squared Error (RMSE)**, with the aim to minimize prediction error.

## Dataset

The dataset used in this project is synthetic and contains information about various podcast episodes. It includes the following features:

- `Episode`: A unique identifier for each episode.
- `Genre`: The category or genre of the podcast (e.g., Comedy, Education, Technology).
- `Sentiment`: Sentiment associated with the podcast episode (e.g., Positive, Negative, Neutral).
- `Listening_Time_minutes`: The target variable, representing the time (in minutes) spent listening to the episode.
- Other numerical and categorical features related to the podcast.

### Data Files

- **train.csv**: The training dataset containing the features and the target variable (`Listening_Time_minutes`).
- **test.csv**: The test dataset for making predictions (without the target variable).
- **sample_submission.csv**: A sample submission file in the required format for Kaggle.

## Model

The machine learning model used for this project is a **Random Forest Regressor**, which is an ensemble learning method suitable for regression tasks. It was chosen for its ability to handle both numerical and categorical data effectively without requiring feature scaling.

### Model Training

- The data is split into **training** (80%) and **validation** (20%) sets.
- The Random Forest model is trained on the training data and evaluated on the validation set using RMSE as the performance metric.
- The model is then used to predict the listening time on the test dataset.

## Installation

To run the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/podcast-listening-time-prediction.git
   cd podcast-listening-time-prediction
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use 'myenv\Scripts\activate'
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset:
   Place the `train.csv` and `test.csv` files in the `data/` directory.

2. Train the model:
   ```bash
   python train.py
   ```

3. Predict listening times on the test data:
   ```bash
   python predict.py
   ```

4. Evaluate model performance on the validation set:
   After training, the model's RMSE on the validation set will be printed.

5. Submit the predictions:
   The predictions will be saved in a CSV file in the required format for Kaggle submission.

## Results

The Random Forest model achieves an RMSE of **10.08** on the validation set, indicating that, on average, the predictions are off by about 10 minutes from the true listening times.

### Next Steps
- **Hyperparameter Tuning**: The model can be fine-tuned using techniques like grid search or random search to improve its performance.
- **Feature Engineering**: Additional features or transformations could be added to further enhance model accuracy.
- **Alternative Models**: Exploring other regression models such as **XGBoost** or **Gradient Boosting** may yield better results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
