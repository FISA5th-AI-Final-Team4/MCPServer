# ==========================================
# 1단계: Base Image (공통 환경)
# ==========================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 시스템 패키지 및 Python 3.12 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 파이썬 심볼릭 링크 설정
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# ==========================================
# 2단계: Builder (가상 환경에 설치)
# ==========================================
FROM base AS builder

WORKDIR /install

# 가상 환경 생성 (/opt/venv)
RUN python -m venv /opt/venv

# 가상 환경 활성화 (PATH에 추가하면 자동으로 venv 내부의 pip/python 사용)
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# 패키지 설치
# [중요] venv 내부로 설치되므로 권한 문제나 경로 꼬임이 없음
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn

# 추가 라이브러리 설치
RUN pip install --no-cache-dir --no-deps \
    nano-graphrag==0.0.8.2 \
    graspologic==3.4.4 \
    hnswlib==0.8.0

# ==========================================
# 3단계: Final (최종 실행)
# ==========================================
FROM base

WORKDIR /app

# [핵심] Builder에서 만든 가상 환경 폴더를 통째로 복사
COPY --from=builder /opt/venv /opt/venv

# [핵심] 가상 환경을 시스템 PATH의 맨 앞에 등록
# 이제 'python'이나 'uvicorn'을 치면 /opt/venv/bin/ 내부의 것을 사용함
ENV PATH="/opt/venv/bin:$PATH"

# 소스 코드 복사
COPY . .

EXPOSE 8011

# venv가 PATH에 잡혀있으므로 명령어가 아주 깔끔해짐
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]