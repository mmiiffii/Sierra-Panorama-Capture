FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git libgl1 &&     rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt &&     python -m playwright install --with-deps chromium

WORKDIR /app
COPY feratel_pano_grabber.py /app/feratel_pano_grabber.py

ENTRYPOINT ["python", "/app/feratel_pano_grabber.py"]
