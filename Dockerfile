FROM python:latest

RUN mkdir /16831_project
ADD . /16831_project

RUN apt-install sudo -y
RUN apt-get update && sudo apt install ffmpeg -y

RUN pip install -r /16831_project/requirements.txt
