FROM python:3.10.14-slim-buster

ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONPATH="/app" 
 
WORKDIR /app 
COPY requirements.txt /app/

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app
ENTRYPOINT ["python", "src/main.py"]