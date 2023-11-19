FROM python:3.11-slim-buster

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /usr/src/app
COPY . /usr/src/app
RUN ls


ENTRYPOINT ["uvicorn","main:app", "--port", "8000"]