
# FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:11.5.1-base-ubuntu20.04

MAINTAINER lambertini and landini <lambertini.1938390@studenti.uniroma1.it>

### RUN executed when build
# RUN apt-get update && apt-get install -y python3 python3-pip sudo

# create dir and copy all the code
RUN mkdir -p /home/app
COPY . /home/app

# installing all the requred libraries
RUN cd /home/app/ && pip3 install -r requirements.txt

#setting workdir
WORKDIR /home/app/

### CMD execute when run

CMD mkdir -p ./Dataset
CMD mkdir -p ./Dataset/sequences
CMD mkdir -p ./Dataset/poses
CMD ls -la

# clonong dataset inside ./Dataset
# CMD wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip -O ./
# CMD unzip ./data_odometry_color.zip -d ./Dataset
# CMD rm -rf ./data_odometry_color.zip

# CMD wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip -O ./
# CMD mkdir ./data_odometry_poses.zip -d ./Dataset
# CMD rm -rf ./data_odometry_poses.zip


# 	docker build -t deep_vo:1.0 .

#	docker run dockerID
#	docker run --gpus 1 -ti dockerID python3 main.py
#	docker run --gpus 1 --ipc=host -ti dockerID python3 main.py


# docker 
