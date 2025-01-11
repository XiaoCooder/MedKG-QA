import structllm as sllm
import json

class query_prompt():
    def __init__(self, args, data, character, names, descriptions):
        self.data = data
        self.args = args
        self.character = character
        self.names = names
        self.descriptions = descriptions
        self.naive_prompt = None
        
    def create_prompt(self, task = None):
        #载入基础prompt
        assert task is not None
        args =self.args  
        names =self.names  #name列表
        descriptions = self.descriptions #人物介绍
        
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
         
        elif task == "qa_extract":
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
        
        elif task == "clean":
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
        
        elif task == "extract_qa":
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
         
        elif task == "summary":
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
    
    def add_query_Prompt(self, data, character, names):
        Prompt = data
        return Prompt
    
   
