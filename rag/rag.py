#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from openai import OpenAI
from client import KnowledgeBaseClient


class RetrievalAugmentedGeneration:
    def __init__(self):
        import os
        import yaml
        with open(os.path.join(os.path.dirname(__file__), "rag.yaml"), "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self.model_name = configs.get("rag").get("model_name")
        self.openai_base_url = configs.get("rag").get("openai_base_url")
        self.openai_api_key = configs.get("rag").get("openai_api_key")
        
    def generate(self, kb_name: str, prompt: str, n_results: int) -> str:
        # 1. 检索知识库
        kb_clt = KnowledgeBaseClient()
        context = kb_clt.query(name=kb_name, question=prompt, n_results=n_results)
        # 2. 使用检索结果作为context，拼接完整prompt
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        # 3. 调用OpenAI API，使用LLM生成回答
        client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Please use Context as a reference to answer questions"},
                {"role": "user", "content": full_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content