FROM python:3.8-slim

WORKDIR /app/
ADD ./requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip3 install -r requirements.txt

COPY ./ /app/
WORKDIR /app/

ENTRYPOINT ["python", "run.py"]
