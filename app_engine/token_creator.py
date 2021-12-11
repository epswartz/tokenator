from predict import container_predict
from skimage.io import imread
from glob import glob
import numpy as np
from PIL import Image
import uuid
import os

TOKEN_TEMPLATE = imread("./token_template.png")
TEMPLATE_COLOR = np.empty((280,280,4))
TEMPLATE_COLOR[:,:,0] = 254
TEMPLATE_COLOR[:,:,1] = 0
TEMPLATE_COLOR[:,:,2] = 254
TEMPLATE_COLOR[:,:,3] = 255
mask = np.all(TEMPLATE_COLOR == TOKEN_TEMPLATE, axis=2)


def create_token(img):
    img_id = str(uuid.uuid4())

    # Read URL for backend GCP vision container.
    model_endpoint = os.getenv('VISION_CONTAINER_ENDPOINT')
    if not model_endpoint:
        resp = container_predict(img, img_id)
    resp = container_predict(img, img_id, model_endpoint)
    box = resp['predictions'][0]['detection_boxes'][0] # TODO multiple faces
    token = crop_token(img, box)
    return token

def create_token_batch(imgs):
    img_ids = [str(uuid.uuid4()) for img in imgs]

    # Read URL for backend GCP vision container.
    model_endpoint = os.getenv('VISION_CONTAINER_ENDPOINT')
    if not model_endpoint:
        resp = container_predict(imgs, img_ids)
    resp = container_predict(imgs, img_ids, model_endpoint)
    tokens = []
    for i in range(len(imgs)):
        box = resp['predictions'][i]['detection_boxes'][0] # TODO multiple faces
        tokens.append(crop_token(imgs[i], box))
    return tokens


def crop_token(img, box):

    h = img.shape[0]
    w = img.shape[1]
    y1, x1, y2, x2 = box

    y1 -= .03
    y2 += .03
    x1 -= .03
    x2 += .03

    rectx = round(x1*w)
    recty = round(y1*h)
    rectw = round((x2*w) - rectx)
    recth = round((y2*h) - recty)

    if recth < rectw:
        diff = rectw - recth
        recty -= diff//2
        recth = rectw
    elif rectw < recth:
        diff = recth - rectw
        rectx -= diff//2
        rectw = recth

    rectx1 = max(0, rectx)
    recty1 = max(0, recty)
    rectx2 = min(img.shape[1]-1, rectx+rectw)
    recty2 = min(img.shape[0]-1, recty+recth)

    cropped = Image.fromarray(img[recty1:recty2, rectx1:rectx2,[2,1,0]])

    cropped = np.array(cropped.resize((280,280)))
    if cropped.shape[-1] == 3:
        cropped = np.dstack((cropped, np.empty((280,280))))
    cropped[:,:,3] = 255

    token = TOKEN_TEMPLATE.copy()
    token[mask] = cropped[mask]
    return token