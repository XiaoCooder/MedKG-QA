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
    
    def ask_question(self, args):
        question = input("please input your question: ")
        if question.lower() in ["exit"]:
            print("bye!")
            return False
        else:
            with open(args.qa_output_path, 'a') as fout:
                fout.write(f"Qustion : {question}\n")
            #Cot based on multi-modal Rerank
            rerank_result, context_rerank, summary_rerank, qas_rerank = sllm.rerank.rerank(self.args, question)
            #with open(args.qa_output_path, 'a') as fout:
            #    fout.write(f"Answer : {rerank_result}\n")   
            
        return True
    
    def start(self, args):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system, you can ask all the questions about the interview text")
            while True:
               if not self.ask_question(args):
                break
            return True
    
