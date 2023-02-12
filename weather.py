from flask import Flask,render_template,redirect

from flask import request
import  numpy as np
import pickle

from keras.models import load_model

ourmodel = load_model('weather.h5')

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST' , 'GET'])
def predict():
    all_features = [(float(x)) for x in request.form.values()]
    x_test = [np.array(all_features)]

    x_test = np.array(x_test)

    predicted = ourmodel.predict(x_test)
    index = np.argmax(predicted)

    mapping = {
        0 : 'DRIZZLE' ,
        1 : 'FOGGY' , 
        2 : 'RAINY' ,
        3 : 'SNOWY' ,
        4 : 'SUNNY' 
    }

    variable = str(mapping[index])
    
    if index == 0:
        return render_template('drizzle.html', outputpredicted  = variable)
    elif index == 1:
        return render_template('foggy.html', outputpredicted  = variable)        
    elif index == 2:
        return render_template('rainy.html', outputpredicted  = variable)
    elif index == 3:
        return render_template('snowy.html', outputpredicted  = variable)
    if index == 4:
        return render_template('sunny.html', outputpredicted  = variable)        

if __name__  == '__main__':
    app.run(debug=True)