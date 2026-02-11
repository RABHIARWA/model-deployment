# EnergiX Web Deployment

EnergiX is an AI-powered solution designed to predict energy consumption in 5G NR base stations. This repository contains the web deployment of EnergiX, providing a user-friendly interface to interact with pre-trained LSTM and XGBoost models.

The models were trained on three merged datasets provided by Huawei, and interactive dashboards were created using PowerBI to analyze energy patterns and support data-driven decision-making.

## Features

- **Interactive Form:** Input base station details (ID, date, hour, etc.) to get energy predictions.
- **Dual Model Prediction:** Uses pre-trained LSTM and XGBoost models for accurate forecasting.
- **Instant Insights:** Quickly displays predicted energy values for informed decision-making.

## Deployment Stack

- **Front-end:** HTML, CSS, JavaScript (Mobirise framework)
- **Back-end:** Python Flask server
- **Machine Learning Models:** Pre-trained LSTM and XGBoost models
- **Data Processing:** StandardScaler for numerical features, one-hot encoding for categorical variables

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/RABHIARWA/EnergiX-Web-Deployment.git
   cd EnergiX-Web-Deployment
2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Run the Flask app: python app.py
5. Ooen the browser and go to http://127.0.0.1:5000 to access the web interface
   
   
