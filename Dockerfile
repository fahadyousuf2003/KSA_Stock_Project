FROM python:3.12-slim

RUN apt update -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python3", "app.py"]

