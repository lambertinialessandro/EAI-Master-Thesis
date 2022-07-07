
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo wget unzip

RUN mkdir -p /home/app

WORKDIR /home/app/

RUN pip3 install -r requirements.txt

RUN mkdir -p ./Dataset
RUN mkdir -p ./Dataset/sequences
RUN mkdir -p ./Dataset/poses

RUN mkdir -p ./History
RUN mkdir -p ./Model

RUN git clone "https://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@github.com/lambertinialessandro/EAI-FinalProject.git"
RUN mv ./EAI-FinalProject/* ./
RUN rm -rf ./ EAI-FinalProject/


