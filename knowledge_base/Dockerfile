FROM python:3.13-slim

WORKDIR /app

COPY ../common ./common
COPY ./knowledge_base ./knowledge_base

RUN pip install -r knowledge_base/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app/knowledge_base

EXPOSE 8001
CMD ["python", "app.py"]