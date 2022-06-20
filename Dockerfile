
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

MAINTAINER lambertini and landini <lambertini.1938390@studenti.uniroma1.it>

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN mkdir -p /home/app

WORKDIR /home/app/

RUN pip3 install -r requirements.txt

RUN mkdir -p ./Dataset
RUN mkdir -p ./Dataset/sequences
RUN mkdir -p ./Dataset/poses

# clonong dataset inside ./Dataset
RUN wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip -O ./
RUN unzip ./data_odometry_color.zip -d ./Dataset
RUN rm -rf ./data_odometry_color.zip

RUN wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip -O ./
RUN mkdir ./data_odometry_poses.zip -d ./Dataset
RUN rm -rf ./data_odometry_poses.zip
