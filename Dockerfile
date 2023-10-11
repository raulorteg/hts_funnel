FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y openbabel && \
    apt-get -y install software-properties-common && \
    apt-get -y install python3-pip && \
    apt-get -y install openbabel

WORKDIR /funnel
COPY . .

RUN python3 -m pip install -r requirements.txt

WORKDIR /funnel/scripts
ENTRYPOINT ["python3", "main.py"]

