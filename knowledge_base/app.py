#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import yaml
import logging
from flask import Flask, request
import sys
sys.path.append("..")
from common.common import *
app = Flask(__name__)
from knowledge_base import KnowledgeBase


@app.route("/knowledgebase/create", methods=["POST"])
def create_knowledgebase():
    with open(os.path.join(os.path.dirname(__file__), "knowledgebase.yaml"), "r", encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    kb_name = request.args.get('name')
    if kb_name is None or len(kb_name) == 0:
        return build_error_response(1001)
    
    chunk_size = request.args.get('chunk_size')
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = configs.get('embeddings').get('default_chunk_size')
    
    doc = request.files['doc']
    tmp_dir = configs.get('storage').get('tmp_doc_base_dir')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    storage_path = os.path.join(tmp_dir, doc.filename)
    doc.save(storage_path)
    
    try:
        kb = KnowledgeBase(name=kb_name)
        kb.create(file_path=storage_path, chunk_size=int(chunk_size))
    except Exception as e:
        logging.error(
            f"create knowledge base exception occurred: {e}",
            exc_info=True)
        return build_error_response(1002, e)
    return build_normal_error_response(data=None)


@app.route("/knowledgebase/query", methods=["POST"])
def query_knowledgebase():
    data = request.get_json()
    if not all([k in data for k in ["name", "question", "n_results"]]):
        return build_error_response(1001)
        
    try:
        kb = KnowledgeBase(name=data["name"])
        answers = kb.query(question=data["question"], top_k=data["n_results"])
    except Exception as e:
        logging.error(
            f"query knowledge base exception occurred: {e}",
            exc_info=True)
        return build_error_response(1003, e)
    return build_normal_error_response(data=answers)


def init_logger(conf) -> None:
    from logging.handlers import RotatingFileHandler
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map[conf.get('web').get('logger').get('level')]
    filename = conf.get('web').get('logger').get('filename')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    max_bytes = conf.get('web').get('logger').get('max_bytes')
    backup_count = conf.get('web').get('logger').get('backup_count')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=level)
    
    formatter = '%(asctime)s -<>- %(filename)s -<>- [line]:%(lineno)d -<>- %(levelname)s -<>- %(message)s'
    size_rotate_file = RotatingFileHandler(filename=filename, maxBytes=max_bytes, backupCount=backup_count)
    size_rotate_file.setFormatter(logging.Formatter(formatter))
    size_rotate_file.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=level)
    console_handler.setFormatter(logging.Formatter(formatter))
    
    logger.addHandler(size_rotate_file)
    logger.addHandler(console_handler)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "knowledgebase.yaml"), "r", encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    init_logger(configs)
    app.run(host="0.0.0.0", port=configs.get("web").get("port"))