FROM python:3.7
MAINTAINER Dev721 "dev721.lkc@gmail.com"

WORKDIR /cnnFlaskApp
#EXPOSE 5000

COPY /requirement.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirement.txt

COPY . /cnnFlaskApp

EXPOSE 5000:5000
ENTRYPOINT python3 app.py 5000
#CMD python app.py