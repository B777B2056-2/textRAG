#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import uuid
import yaml
from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
import dashscope
from http import HTTPStatus


class KnowledgeBase:
    def __init__(self, name: str):
        self.name = name
        
        with open(os.path.join(os.path.dirname(__file__), "knowledgebase.yaml"), "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            
        self._embedding_api_key = configs.get('embeddings').get('openai_api_key')
        self._embedding_dimensions = configs.get('embeddings').get('dimensions')
        self._embedding_model_name = configs.get('embeddings').get('model_name')
        self._embedding_base_url = configs.get('embeddings').get('openai_base_url')
        
        self._embedding_client = OpenAI(
            api_key=self._embedding_api_key,
            base_url=self._embedding_base_url,
        )
        
        chromadb_persistent_path = configs.get('chromadb').get('local_persist_directory')
        self._db = chromadb.PersistentClient(chromadb_persistent_path)
        
    
    def _load(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        return loader.load()
    
        
    def _chunk(self, docs: List[Document], chunk_size: int, overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=overlap)
        return splitter.split_documents(docs)
    

    def _embed(self, texts: List[str]) -> List[List[float]]:
        completion = self._embedding_client.embeddings.create(
            model=self._embedding_model_name,
            input=texts,
            dimensions=self._embedding_dimensions,
            encoding_format="float"
        )
        
        result = [[] for _ in range(len(completion.data))]
        for item in completion.data:
            if item.object != 'embedding':
                continue
            result[item.index] = item.embedding
        return result

    
    def _store(self, text: str, embeddings: List[float]) -> None:
        try:
            collections = self._db.get_or_create_collection(f"{self.name}_collection")
        except Exception as e:
            print(e)
            raise e
        
        try:
            collections.upsert(ids=str(uuid.uuid4()), documents=text, embeddings=embeddings)
        except Exception as e:
            print(e)
            raise e
        
        
    def create(self, file_path: str, chunk_size: int = 1024, overlap: int = 0):
        docs = self._load(file_path)
        all_splits = self._chunk(docs, chunk_size, overlap)
        embeddings = self._embed(texts=[s.page_content for s in all_splits])
        for i in range(len(all_splits)):
            self._store(text=all_splits[i].page_content, embeddings=embeddings[i])
        
    
    def _rerank(self, question: str, docs: List[str], top_k: int) -> List[str]:
        with open(os.path.join(os.path.dirname(__file__), "knowledgebase.yaml"), "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        
        resp = dashscope.TextReRank.call(
            model=configs.get('rerank').get('model_name'),
            query=question,
            documents=docs,
            top_n=top_k,
            return_documents=True,
            api_key=configs.get('rerank').get('openai_api_key')
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Rerank API call failed with status code {resp.status_code}")
        
        resp.output.results.sort(key=lambda x: x.relevance_score, reverse=True)
        return [result["document"]["text"] for result in resp.output.results]
        
        
    def query(self, question: str, top_k: int = 5) -> List[str]:
        # 1. 用户问题向量化
        question_embedding = self._embed([question])[0]
        # 2. 获取向量数据库collections
        try:
            collections = self._db.get_collection(f"{self.name}_collection")
        except Exception as e:
            print(e)
            raise e
        # 3. 查询（召回）
        try:
            results = collections.query(question_embedding, n_results=top_k)
        except Exception as e:
            print(e)
            raise e
        docs = results["documents"][0]
        # 4. 重排序
        return self._rerank(question=question, docs=docs, top_k=top_k)
        
        
if __name__ == "__main__":
    kb = KnowledgeBase(name="test")
    # kb.create("/Users/jiangrui07/Downloads/通义千问3-Embedding.md", chunk_size=64)
    print(kb.query("Qwen3 Embedding 模型系列是什么？"))