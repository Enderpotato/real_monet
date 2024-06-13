FROM ubuntu:20.04

WORKDIR /app
COPY . /app
RUN apt-get -y update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt


EXPOSE 5000
CMD ["python3", "app.py"]