
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo wget unzip git libsm6 libxext6

RUN mkdir -p /home/app
WORKDIR /home/app/

RUN mkdir -p ./Dataset
RUN mkdir -p ./Dataset/sequences
RUN mkdir -p ./Dataset/poses

RUN mkdir -p ./History
RUN mkdir -p ./Model

# COPY ./requirements.txt /home/app/
# RUN pip3 install -r requirements.txt


RUN git clone "https://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@github.com/lambertinialessandro/EAI-FinalProject.git"
RUN mv ./EAI-FinalProject/* ./
RUN rm -rf ./EAI-FinalProject/

RUN pip3 install -r requirements.txt
