# -*- coding: utf-8 -*-

#importing libraries
import cv2
import numpy as np
import traceback
import pickle
import keras
import time
from flask import Flask, request, jsonify, make_response
import os

app = Flask(__name__)

@app.route('/classify',methods=['POST'])
def test():
    try:
        ima = request.files['image']
        ima.save('img.jpg')
        image='img.jpg'
        Xtest=[]

        model_name = 'KYC_Vgg_transfer_learning.h5'
        model2 = keras.models.load_model(model_name)#loading the saved model
        
        
        pred=(cv2.imread(image,cv2.IMREAD_GRAYSCALE))
        pred=cv2.resize(pred,(224,224))#resizing the input image
        pred=cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        Xtest.append(pred)
        Xtest=np.array(Xtest)
        
        start = time.time()
        preds=model2.predict(Xtest, verbose=1)
        end = time.time()
        
        res=np.argmax(preds)
        
        #loading the LabelEncoder() obj 
        pkl_file = open('encoder.pkl', 'rb')
        le_depart = pickle.load(pkl_file) 
        pkl_file.close()
        
        result = le_depart.inverse_transform([res])[0]
        
        os.remove('img.jpg')
        return make_response(jsonify(
                {
                    'status' : 'success',
                    'desc' : '',
                    'result' : result
                }
            ), 200)

    except:
        os.remove('img.jpg')
        print(traceback.print_exc())
        return make_response(jsonify(
                {
                    'status' : 'fail',
                    'desc':'upload files',
                    'result' : ''
                }
            ), 207)

#image='NTM4MTI5MDYzMTE.jpeg'
#print(test(image))
