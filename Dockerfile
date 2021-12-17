# pull official base image
FROM python:3.9.5-slim-buster

# image editing libraries
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# cache resnet50 in the image
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN cp resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/