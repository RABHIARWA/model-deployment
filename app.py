from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)

# Load the LSTM model
model = tf.keras.models.load_model('./models/lstm_model.h5')

# Load the MinMaxScaler and fit it to the training data
scaler = MinMaxScaler()
scaler.fit(pd.read_csv('merged_data.csv')[['Month', 'Day', 'Hour', 'load', 'Frequency', 'Bandwidth', 'Antennas', 'TXpower', 'Energy']])

def preprocess_input(bs, cell_name, load, frequency, bandwidth, antennas, txpower):
    numerical_data = scaler.transform([[load, frequency, bandwidth, antennas, txpower]])
    categorical_data = pd.get_dummies(pd.DataFrame({'BS': [bs], 'CellName': [cell_name]}))
    input_data = np.concatenate((numerical_data, categorical_data.values), axis=1)
    input_data_reshaped = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
    return input_data_reshaped

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data from request
    bs = float(request.form.get('BS'))
    cell_name = float(request.form.get('CellName'))
    date = float(request.form.get('date'))
    
    # Preprocess input data
    input_data = preprocess_input(bs, cell_name, date)
    
    # Perform prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Pass prediction to the client
    return jsonify({'prediction': prediction.tolist()})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
