#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import yaml
import logging
from flask import Flask, request, Response
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
        
    if "stream" not in data:
        stream = False
    else:
        stream = data["stream"]
    
    rag = RetrievalAugmentedGeneration()
    
    if not stream:
        try:
            answer = rag.generate(kb_name=kb_name, prompt=prompt, n_results=n_results, stream=stream)
        except Exception as e:
            return build_error_response(1004, e)
        return build_normal_error_response(data=answer)
    else:
        def event_stream():
            llm_stream = rag.generate(kb_name=kb_name, prompt=prompt, n_results=n_results, stream=stream)
            for event in llm_stream:
                if not event:
                    continue
                yield f"data: {event.choices[0].delta.content}\n\n"
        return Response(event_stream(), content_type="text/event-stream")


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
    with open(os.path.join(os.path.dirname(__file__), "rag.yaml"), "r", encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    init_logger(configs)
    app.run(host="0.0.0.0", port=configs.get("web").get("port"))