# Dockerfile
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
#ENV APP_HOME /app

COPY . /app
WORKDIR /app

#WORKDIR $APP_HOME
#COPY . ./


RUN pip install -r requirements.txt

ENV PORT 8080

ENTRYPOINT ["python"]
CMD ["src/app.py"]


#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app