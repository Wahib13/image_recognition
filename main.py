import os
import pathlib

import aiofiles
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

import torchvision.models as models
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from consts import class_names

app = FastAPI()

# model_transfer = models.resnet50(pretrained=True)
# for param in model_transfer.parameters():
#     param.require_grad = False

# model_transfer.load_state_dict(torch.load('nn_models/model_transfer.pt', map_location=torch.device('cpu')))


@app.post('/detect_face/')
async def detect_face(file: UploadFile = File(...)):
    cwd = pathlib.Path().resolve()
    print(cwd)
    working_image_path = os.path.join(cwd, f'uploads/{file.filename}')
    output_image_path = os.path.join(cwd, f'processed/{file.filename}')
    # save as file to path
    async with aiofiles.open(working_image_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    image_path = working_image_path
    face_cascade = cv2.CascadeClassifier('/usr/src/app/haarcascades/haarcascade_frontalface_alt.xml')

    # load color (BGR) image
    img = cv2.imread(image_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # print number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(output_image_path, img)
    return FileResponse(output_image_path)


# @app.post('/detect_dog_breed/')
# async def detect_dog_breed(file: UploadFile = File(...)):
#     cwd = pathlib.Path().resolve()
#     working_image_path = os.path.join(cwd, f'uploads/{file.filename}')
#     # save as file to path
#     async with aiofiles.open(working_image_path, 'wb') as out_file:
#         content = await file.read()
#         await out_file.write(content)
#
#     predicted_breed = predict_breed(working_image_path, model_transfer)
#
#     return {'breed': predicted_breed}


def predict_breed(img_path, model):
    # load the image and return the predicted breed
    img = Image.open(img_path)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transformed_image = transformations(img)[:3, :, :].unsqueeze(0)
    output = model(transformed_image)
    _, preds = torch.max(output, 1)
    prediction = np.squeeze(preds.cpu().numpy())
    return class_names[prediction]
