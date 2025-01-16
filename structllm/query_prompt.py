import structllm as sllm
import json

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
                             "content": self.add_query_Prompt(self.data,self.character,self.names)
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
         assert question is not None
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
                             "content": self.add_query_Prompt(self.data,self.character,self.names)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
         
        elif task == "ask1":
         try:
           with open(args.summary_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data,self.character,self.names)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        elif task == "ask2":
         try:
           with open(args.summary_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data,self.character,self.names)
                         }
                )
         except json.JSONDecodeError as e:
             #print(f"JSON 解码错误: {e}")
             pass
        elif task == "ask3":
         try:
           with open(args.summary_prompt_path,'r',encoding = 'utf-8') as json_file:
                self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                         {
                             "role": "user",
                             "content": self.add_query_Prompt(self.data,self.character,self.names)
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
        if (question is not None):
            Prompt = Prompt + "\n Answer the question based on context:" + question
        return Prompt
    
    def add_ask_Prompt(self, data, character=None , names = None, question=None):
        if isinstance(data, list):
            Prompt = "\n".join(data)
        else:
            Prompt = data
        if (question is not None):
            Prompt = Prompt + "\n Answer the question based on context:" + question
        return Prompt
    
   
