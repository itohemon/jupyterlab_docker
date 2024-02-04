FROM python:3.10 
USER root

ENV LANG=ja_JP.UTF-8 \
    LAUGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    TZ=JST-9 \
    TERM=xterm

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 && \
    apt-get -y install ffmpeg && \
    apt-get install -y vim less && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src
COPY requirements.txt /src
WORKDIR /src

#JupyterLabで拡張機能インストールするために必要
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash && \ 
    apt-get install nodejs

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r requirements.txt
