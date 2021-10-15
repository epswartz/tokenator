import cv2
import numpy as np
import requests
from flask import Flask, request, send_file
from token_creator import create_token
import io
import time
from PIL import Image
from skimage.io import imsave

app = Flask(__name__)

@app.route('/create_token',methods=['POST'])
def creator():
  file = request.files['file']
  data = file.read()
  # convert string of image data to uint8
  arr = np.fromstring(data, np.uint8)

  # decode image
  img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  token = create_token(img)
  now_str = str(round(time.time() * 10000))
  output_filename = "output_" + now_str + ".png"

  # imsave(output_filename, token)
  # return "Image Saved: " + output_filename

  print("Returning token")
  buf = io.BytesIO()
  token_img = Image.fromarray(token)
  token_img.save(buf, format='PNG')
  buf.seek(0)
  return send_file(buf, attachment_filename=output_filename, mimetype='image/png')

if __name__ == '__main__':

    # For App Engine, this MUST be 8080 or you get 502.
    app.run(host='0.0.0.0', port=8080, debug=True)