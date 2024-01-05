FROM python:3.11-slim
WORKDIR /usr/src/app
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models api
COPY models models
COPY api api

RUN chown -R appuser:appuser /usr/src/app
USER appuser
WORKDIR /usr/src/app/api

EXPOSE 8080
CMD ["gunicorn", "-w 4", "-b :8080", "server:app"]
