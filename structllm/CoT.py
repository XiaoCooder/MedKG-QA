import re
import structllm as sllm
import json
import openai

def CoT(self, args, question):
    llm = sllm.llm.gpt(args)
    self.question = question
    results_prompt = sllm.retrieve.get_context_collection_and_query(args.encoder_model, query_texts = question , recall_num = 3)
    #data = [candidate_question for candidate_question in results_prompt['documents'][0]]
    #contents = [candidate_content.get('content', '') for candidate_content in data['metadatas'][0]]
    print(results_prompt)
    
