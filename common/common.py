#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from typing import Any
from flask import jsonify, Response


ERRORS = [
    {"code": 1000, "message": "Unknown Error"},
    {"code": 1001, "message": "Missing required fields"},
    {"code": 1002, "message": "Create Knowledge Base Failed"},
    {"code": 1003, "message": "Query Knowledge Base Failed"},
    {"code": 1004, "message": "Call RAG failed"}
]


def build_error_response(err_code: int, e: Exception = None) -> tuple[Response, int]:
    for error in ERRORS:
        if error.get("code") == err_code:
            if e is not None:
                msg = f"{error.get('message')}, Error: {e}"
            else:
                msg = error.get("message")
            return jsonify({"code": err_code, "message": msg}), 200
    
    if e is not None:
        msg = f"Error: {e}"
    else:
        msg = "Unknown Error"
    return jsonify({"code": 1000, "message": msg}), 200


def build_normal_error_response(data: Any) -> tuple[Response, int]:
    return jsonify({"code": 200, "message": "success", "data": data}), 200