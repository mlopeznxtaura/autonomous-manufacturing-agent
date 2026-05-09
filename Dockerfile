FROM nvidia/cuda:12.3.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

EXPOSE 8000 9090
ENTRYPOINT ["python3", "main.py"]
CMD ["--mode", "simulate", "--duration", "300", "--line", "LINE_A"]
