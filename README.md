# BBCA Stock Price Prediction using LSTM


This project implements an LSTM (Long Short-Term Memory) model to predict BBCA (Bank Central Asia Tbk) stock prices on the Indonesia Stock Exchange. The project combines technical analysis indicators with deep learning to forecast stock price movements based on historical data.

## 📌 Project Description

This project is designed to predict BBCA stock closing prices using an LSTM deep learning model. The project incorporates various technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands as features for training the model. The notebook includes the complete workflow from data acquisition, preprocessing, model training, to evaluation of results.

## ✨ Key Features

- Stock data acquisition from Yahoo Finance
- Technical indicator extraction:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- LSTM architecture with multiple layers and dropout to prevent overfitting
- Visualization of prediction results using Plotly
- Model and scaler storage for future predictions
- Model evaluation using MAE, MSE, and RMSE metrics

## 🛠 Technologies Used

- **Programming Language**: Python 3.7+
- **Deep Learning Framework**: TensorFlow 2.10+, Keras
- **Data Analysis Libraries**: Pandas, NumPy
- **Visualization Libraries**: Plotly, Matplotlib
- **Technical Analysis Library**: TA-Lib
- **Others**: scikit-learn, joblib

## 📂 Directory Structure

```
BBCAPrediction/
├── BBCA_Prediksi.ipynb          # Main notebook for model training
├── README.md                     # Project documentation
├── requirements.txt              # Dependency list
└── artifacts/                    # Folder for saving models and scalers
    ├── best_bbca_lstm_model.keras  # Best model after training
    ├── scaler.pkl                # Scaler for data normalization
    └── prediction_results/       # Future prediction results
```

## ⚙️ Installation and Requirements

### System Requirements
- Python 3.7 or newer
- Google Colab (optional, but recommended for training)

### Installing Dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:
```
yfinance==0.2.28
pandas==2.0.3
numpy==1.24.4
scikit-learn==1.3.0
tensorflow==2.13.0
ta==0.10.2
plotly==5.15.0
joblib==1.3.0
matplotlib==3.7.2
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/username/BBCAPrediction.git
   cd BBCAPrediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download data and train the model:
   - Open `BBCA_Prediksi.ipynb` in Jupyter Notebook or Google Colab
   - Run all cells in the notebook

4. For future predictions:
   - Adjust the prediction dates in the configuration section
   - Run the prediction section in the notebook

## 📈 Results and Evaluation

The trained LSTM model achieved the following performance on test data:

- **MAE (Mean Absolute Error)**: 0.0584
- **MSE (Mean Squared Error)**: 0.00539
- **RMSE (Root Mean Squared Error)**: 0.0734

The prediction graph shows that the model is able to capture general patterns of BBCA stock price movements, with some deviations at extreme points.

## 📋 How to Use for Future Predictions

1. Ensure the model is trained and saved in the `artifacts` folder
2. Adjust prediction dates in the configuration section:
   ```python
   CONFIG = {
       'ticker': 'BBCA.JK',
       'start_date': '2021-01-01',
       'end_date': '2025-12-31',  # Adjust end date
       'sequence_length': 60,
       'learning_rate': 0.001,
       'epochs': 100,
       'batch_size': 32
   }
   ```
3. Run the prediction section in the notebook
---

*Note: Stock predictions do not guarantee future accuracy. Using this model for investment decisions should be done with proper risk considerations.*
