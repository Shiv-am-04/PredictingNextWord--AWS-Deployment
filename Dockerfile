FROM python:3.11.4-slim-buster
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip

RUN apt update -y && apt install awscli -y

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","app.py"]