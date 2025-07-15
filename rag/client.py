#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from typing import List, Any, Dict
import requests


class BaseClient:
    def __init__(self, clt_name: str):
        import os
        import yaml
        with open(os.path.join(os.path.dirname(__file__),"rag.yaml"), "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self.endpoint = configs.get("server").get(clt_name).get("endpoint")
        
        
    def post(self, uri: str, params: Dict) -> Any:
        url = self.endpoint + "/" + uri
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception(f"Error while calling the {url}: "
                            f"status code {response.status_code}")
        resp = response.json()
        if resp["code"] != 200:
            msg = resp["message"]
            raise Exception(f"Error while calling the {url}: {msg}")
        return resp["data"]


class KnowledgeBaseClient(BaseClient):
    def __init__(self):
        super(KnowledgeBaseClient, self).__init__("knowledgebase")
    
    def query(self, name: str, question: str, n_results: int) -> List[str]:
        params = {
            "name": name,
            "question": question,
            "n_results": n_results,
        }
        data = self.post(uri="knowledgebase/query", params=params)
        return data