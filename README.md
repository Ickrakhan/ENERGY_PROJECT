## Energy Consumption Prediction using LSTM

Project Overview

This project applies Long Short-Term Memory (LSTM) neural networks to forecast household energy consumption using time-series data. The dataset (household_power_consumption_small.txt) contains features like Global Active Power, Voltage, and Sub-metering values. The main goal is to analyze historical patterns and predict future consumption trends for better energy management.

Project Structure

ENERGY_PROJECT/
│── household_power_consumption_small.txt  # Smaller dataset used for training & testing
│── lstm_energy.py                         # Main Python script (LSTM model training & prediction)
│── prediction_plot.png                    # Output graph of actual vs predicted values
│── results.csv                            # CSV file containing Actual vs Predicted values
│── requirements.txt                       # List of dependencies
│── README.md                              # Project documentation

Requirements

Install all dependencies with:

pip install -r requirements.txt


Main libraries used:

TensorFlow / Keras
NumPy
Pandas
Matplotlib
Scikit-learn

How to Run

Clone this repository:

git clone https://github.com/Ickrakhan/ENERGY_PROJECT.git


Navigate to the project folder:

cd ENERGY_PROJECT


Run the script:

python lstm_energy.py

## Results

The LSTM model successfully captured the trend of household energy consumption.

## Outputs generated:

prediction_plot.png → Visualization of Actual vs Predicted energy consumption.

results.csv → Contains tabular data of Actual vs Predicted values for further analysis.

Sample (from results.csv):

Actual	Predicted
4.216	4.101
5.360	5.275
3.458	3.602
4.123	4.008

## Conclusion

This project demonstrates the capability of LSTM neural networks in predicting household energy consumption from historical data. The model successfully captured patterns in electricity usage and provided reliable future forecasts, proving the effectiveness of deep learning in energy management.
