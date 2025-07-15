#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import yaml
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
        return build_error_response(1003, e)
    return build_normal_error_response(data=answers)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "knowledgebase.yaml"), "r", encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    app.run(host="0.0.0.0", port=configs.get("web").get("port"))