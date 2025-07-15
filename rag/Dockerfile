FROM python:3.13-slim

WORKDIR /app

COPY ../common ./common
COPY . ./rag

ENV PYTHONPATH="${PYTHONPATH}:/app/common"

RUN pip install -r rag/requirements.txt

EXPOSE 8000
CMD ["python", "rag/app.py"]