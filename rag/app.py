#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import yaml
from flask import Flask, request
import sys
sys.path.append("..")
from common.common import *
app = Flask(__name__)
from rag import RetrievalAugmentedGeneration


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not all([k in data for k in ["kb_name", "prompt"]]):
        return build_error_response(1001)
    
    kb_name = data["kb_name"]
    prompt = data["prompt"]
    
    if "n_results" not in data:
        with open(os.path.join(os.path.dirname(__file__), "rag.yaml"), "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        default_n_results = configs.get("rag").get("default_n_results")
        n_results = default_n_results
    else:
        n_results = data["n_results"]
    
    try:
        rag = RetrievalAugmentedGeneration()
        answer = rag.generate(kb_name=kb_name, prompt=prompt, n_results=n_results)
    except Exception as e:
        return build_error_response(1004, e)
    return build_normal_error_response(data=answer)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "rag.yaml"), "r", encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    app.run(host="0.0.0.0", port=configs.get("web").get("port"))