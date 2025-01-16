import os
import json
import torch
import openai
import structllm as sllm
from torch import Tensor
from openai import OpenAI
from tqdm.autonotebook import trange
from typing import List, Union, TypeVar, Dict
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from chromadbx import NanoIDGenerator

class user_qa:
    def __init__(self, args):
       self.args = args
    
    def ask_question(self):
        question = input("please input your question: ")
        #
        if question.lower() in ["exit"]:
            print("bye!")
            return False
        else:
            result = sllm.CoT(self.args)
            pass     
        return True
    
    def start(self):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system, you can ask all the questions about the previous text")
            while True:
               if not self.ask_question():
                break
            return True
    
