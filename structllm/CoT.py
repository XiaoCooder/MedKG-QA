import re
import structllm as sllm
import json
import openai

def CoT(self, args, question):
    llm = sllm.llm.gpt(args)
    self.question = question
    
    
    
