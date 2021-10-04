from flask import Flask, request, Response,make_response
#import jsonpickle
import numpy as np
import cv2
import os
import io
from PIL import Image
from FRecon import FRecon


reconhecimento = FRecon("FACENET","../")


# initialize API
app = Flask( __name__ )

@app.route( '/facialrecognition', methods=['POST'] )
def Reconhecimento_Racial():
    if request.files['image'].filename == '':
        return Response( 'No selected image', status=200, mimetype='text/xml' )
    else:
        
        image = request.files["image"]
        image_bytes = Image.open(image)
        
        img = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

        img2 = reconhecimento.PredictFace(img,"LR")

        retval, buffer = cv2.imencode('.jpg', img2)

        response = make_response(buffer.tobytes())
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set('Content-Disposition', 'attachment', filename='output.jpg' )
        return response 
        

if __name__ == '__main__':
    port = 5000
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )

