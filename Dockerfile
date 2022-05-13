
FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu18.04

MAINTAINER alessandro lambertini <lambertini.1938390@studenti.uniroma1.it>

# RUN executed when build
# 	docker build -t DeepVO:1.0 .
RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN mkdir -p /home/app

COPY . /home/app

RUN cd /home/app/ && pip3 install -r requirements.txt

WORKDIR /home/app/

# CMD execute when run
#	docker run dockerID
#	docker run --gpus 1 -ti dockerID python3 main.py
#	docker run --gpus 1 --ipc=host -ti dockerID python3 main.py

CMD mkdir -p ./Dataset

CMD wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip -O ./
CMD unzip ./data_odometry_color.zip -d ./Dataset

CMD wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip -O ./
CMD mkdir ./data_odometry_poses.zip -d ./Dataset

