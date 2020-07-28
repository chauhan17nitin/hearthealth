from flask import Flask , jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
 
app = Flask(__name__)


model=load_model("model1.h5")
 
@app.route('/', methods=['POST'])
def predict():

    if request.is_json:
        hdata=request.get_json()

    age=hdata["age"]
    gender=hdata["gender"]
    height=hdata["height"]
    weight=hdata["weight"]
    ap_hi=hdata["ap_hi"]
    ap_lo=hdata["ap_lo"]
    cholestrol=2.0
    gluc=1.0
    smoke=hdata["smoke"]
    alco=hdata["alco"]
    active=["active"]
    bmi=weight/(height/100)**2

    age= (age-53.32)/6.76

    height= (height-164.36)/8.18
    weight=(weight-74.12)/14.33
    ap_hi=(ap_hi-126.61)/16.76
    ap_lo= (ap_lo-81.35)/9.72

    input_data=[age,gender,height,weight,ap_hi,ap_lo,cholestrol,gluc,smoke,alco,active,bmi]
    result=model.predict(np.array(input_data))

    output = {'results': result}

    return jsonify(results=output)

 
 
if __name__ == "__main__":
   app.run(debug=True)