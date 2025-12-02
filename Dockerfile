# 1단계: NVIDIA CUDA 런타임 (Ubuntu 22.04)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# 2단계: PPA 추가 및 Python 3.12 설치
# software-properties-common: add-apt-repository 명령어를 위해 필요
# curl: pip 설치를 위해 필요
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3단계: pip 수동 설치 (PPA 버전은 pip가 기본 포함 안 됨)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 4단계: python, pip 명령어 심볼릭 링크 연결
# (이제 'python'이라고 치면 'python3.12'가 실행됨)
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# --- 빌드 단계 (Builder) ---
FROM base AS builder

WORKDIR /install

COPY requirements.txt .

# pip 업그레이드 및 패키지 설치
# (pip가 3.12용으로 설치되었는지 확인)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/local -r requirements.txt

# 추가 라이브러리 설치
RUN pip install --no-cache-dir --prefix=/install/local --no-deps \
    nano-graphrag==0.0.8.2 \
    graspologic==3.4.4 \
    hnswlib==0.8.0

# --- 최종 실행 단계 (Final) ---
FROM base

WORKDIR /app

# 빌드된 패키지 복사
COPY --from=builder /install/local /usr/local

COPY . .

EXPOSE 8011

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]