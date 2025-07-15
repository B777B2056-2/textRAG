FROM python:3.13-slim

WORKDIR /app

COPY ../common ./common
COPY . ./knowledge_base

ENV PYTHONPATH="${PYTHONPATH}:/app/common"

RUN pip install -r knowledge_base/requirements.txt

EXPOSE 8001
CMD ["python", "knowledge_base/app.py"]