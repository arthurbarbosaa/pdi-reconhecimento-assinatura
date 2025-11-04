FROM python

WORKDIR /app

RUN apt update && apt install -y git vim libgl1

COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip  && python3 -m pip install -r requirements.txt