FROM python:3.13-slim

WORKDIR /app

COPY ../common ./common
COPY ./rag ./rag

RUN pip install -r rag/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app/rag

EXPOSE 8000
CMD ["python", "app.py"]