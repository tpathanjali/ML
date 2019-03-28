# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import dill as pickle
from sklearn.preprocessing import StandardScaler
#use command python server.py 12345 in anaconda prompt or any python prompt
# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST']) #post method to send data and receive info
def predict():
    if lr:
        try:
            json_ = request.json #json must be double quoted
            #use the below code X_test.to_dict(orient='records')
            #replace single quote(') with double quote('') while sending the request
            #from postman application
            print(json_)
            #transforming json into a dataframe
            query = pd.DataFrame(json_)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    with open('C:/Users/sagar/Desktop/automobile/model.pkl' ,'rb') as f:
        lr = pickle.load(f)
    print ('Model loaded')
    
    app.run(port=port, debug=True)