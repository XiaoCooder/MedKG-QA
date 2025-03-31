import structllm as sllm
import json
import asyncio

class query_prompt():
    def __init__(self, args, data, character = None, names=None, descriptions=None):
        self.data = data
        self.args = args
        self.character = character
        self.names = names
        self.descriptions = descriptions
        self.naive_prompt = None
        
    def create_prompt(self, task = None, question = None):
        #载入基础prompt
        assert task is not None
        args = self.args  
        names = self.names  #name列表
        descriptions = self.descriptions #人物介绍

        if (self.naive_prompt is not None):
           self.naive_prompt = None

        if task == "clean":
         try:
           with open(args.clean_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
            
        elif task == "extract_q":
         try:
           with open(args.extract_q_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
         
        elif task == "extract_a":
         try:
           with open(args.extract_a_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data, question = question)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
         
        elif task == "summary":
         try:
           with open(args.summary_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
         
        elif task == "summary_rerank":
         try:
           with open(args.reranker_prompt,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        elif task == "context_rerank":
         try:
           with open(args.reranker_prompt,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        elif task == "qas_rerank":
         try:
           with open(args.reranker_prompt,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
         
        elif task == "get_answer":
         try:
           with open(args.qa_prompt,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        
        elif task == "extract_triple":
         try:
           with open(args.extractKG_prompt,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        
        elif task == "extract_qa":
         try:
           with open(args.extractQA,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass

        elif task == "extract_keywords":
         try:
           with open(args.extract_keywords,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_ask_Prompt(self.data, question, task)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass   

    def add_query_Prompt(self, data, character=None , names = None, question=None):
        if isinstance(data, list):
            Prompt = "\n".join(data)
        else:
            Prompt = data
        if (question):
            Prompt = Prompt + "Find the question in the context and answer the question based on its next sentence , just output the final answer:" + question
        return Prompt
    
    def add_ask_Prompt(self, data, question, task):

        if task == "extract_triple": 
            data_prompt = ''  
            for i in range(len(data)):
               task_prompt = f"I need you to extract triples from the following:{data[i]}\n"
               data_prompt = data_prompt + task_prompt
            Prompt = data_prompt
        
        if task == "extract_qa": 
            data_prompt = ''  
            for i in range(len(data)):
               task_prompt = f"I need you to extract Q&A pairs from the following:{data[i]}\n"
               data_prompt = data_prompt + task_prompt
            Prompt = data_prompt
        
        if task == "extract_keywords": 
            data_prompt = ''  
            for i in range(len(data)):
               task_prompt = f"I need you to extract keywords from the following:{data[i]}\n"
               data_prompt = data_prompt + task_prompt
            Prompt = data_prompt

        return Prompt
    
   
