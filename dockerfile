FROM python:3.7

MAINTAINER promisejeremiah

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /src/iris

WORKDIR /src/iris

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py", "--host", "0.0.0.0", "--port", "5000"]
