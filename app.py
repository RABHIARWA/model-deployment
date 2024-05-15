from datetime import datetime 
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


app = Flask(__name__)

# Load the LSTM model
model = tf.keras.models.load_model('./models/lstm_model (2).h5')



# Load the MinMaxScaler and fit it to the training data
scaler = StandardScaler()

def preprocess_input(bs, cell_name, month,day,hour,load, frequency, bandwidth, antennas, txpower):
    numerical_data = scaler.transform([[month,day,hour,load, frequency, bandwidth, antennas, txpower]])
    categorical_data = pd.get_dummies(pd.DataFrame({'BS': [bs], 'CellName': [cell_name]}))
    input_data = np.concatenate((numerical_data, categorical_data.values), axis=1)
    input_data_reshaped = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
    return input_data_reshaped

@app.route('/predict', methods=['POST'])
def predict():
    bs = request.form.get('BS')
    hour = float(request.form.get('hour'))
    date = request.form.get('date')
    print(bs)
    print(hour)
    print(date)
    print(type(date))
    date_obj = datetime.strptime(date, '%Y-%m-%d')  # Convert string to datetime object
    
    month = date_obj.month  # Extract month
    day = date_obj.day     # Extract day
    # Extract form data from request
    df=pd.read_csv('final_dataset_with_preds.csv')
    #Since most machine learning models only accept numerical variables, preprocessing the categorical variables becomes a necessary step. 
    #We need to convert these categorical variables to numbers such that the model is able to understand and extract valuable information.
    df_encoded = pd.get_dummies(df,columns=['BS','RUType','Mode'])
    target=df['Energy']
    print("Shape of input data before scaling: ",df.drop(['Energy'],axis=1).shape)
    #features=df_encoded.drop(['Energy'],axis=1)
    features=df_encoded
    model_features=['Month', 'Cell0_load' ,'Hour', 'Day','Cell0_TXpower','Cell0_Bandwidth', 'Antennas' ,'Cell0_Frequency','Energy']

    features_scaled=scaler.fit_transform(features[model_features])
    print("Shape of features scaled: ",features_scaled.shape)
    X = np.array(features_scaled).reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
    x_train,x_test,y_train,y_test=train_test_split(X,target,test_size=0.2,random_state=42)

    #You can check the architecture of the loaded model using the summary() method.
    #This step is optional but can help you understand the model's structure.
    model.summary()
    #fit the keras model on the dataset
    #If you have labeled data for evaluation, you can use the evaluate() method to assess the model's performance.
    model.evaluate(X, target, verbose=2)
    print(x_test.shape)

    
    single_features=[month, 0 ,hour, day,0,0, 0 ,0,10]
    single_features = np.array([single_features])
    single_features_reshaped = np.array(single_features).reshape((single_features.shape[0], 1, single_features.shape[1]))

   
    prediction = model.predict(single_features_reshaped)
    print("PREDICTION:" ,prediction)
    print(type(prediction))

    
    # Convert the NumPy array to a string
    prediction_str = np.array2string(prediction)

    # Return the prediction string as a JSON response
    return jsonify(prediction=prediction_str)
    
    
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
