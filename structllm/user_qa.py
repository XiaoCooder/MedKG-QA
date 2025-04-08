import structllm as sllm
import json
import os
import tqdm
class user_qa:
    def __init__(self, args, corpus, path):
        self.args = args
        self.corpus = corpus
        self.path = path

    def ask_question(self, question = ""):

        if question.__eq__(""): 
            question = input("please input your question: ")
            if question.lower() in ["exit"]:
                print("bye!")
                return False
            answers, used_triples = sllm.cot.cot(self.args, question, self.corpus, self.path)
            #import pdb;pdb.set_trace()
            used_triples_text = ", ".join([f"[{h},{r},{t}]" for h, r, t in used_triples])
            qa_item = {
                "Q": question,
                "A": answers[0],
                "used_triples": used_triples_text
            }
            # 1. 如果文件存在，先读取已有内容
            if os.path.exists(self.args.qa_output_path):
                with open(self.args.qa_output_path, 'r', encoding='utf-8') as fin:
                    try:
                        qa_history = json.load(fin)
                    except json.JSONDecodeError:
                        qa_history = []  # 文件内容为空或损坏
            else:
                qa_history = []
            # 2. 添加新记录
            qa_history.append(qa_item)
            
            # 3. 写回文件（覆盖写入）
            with open(self.args.qa_output_path, 'w', encoding='utf-8') as fout:
                json.dump(qa_history,fout,ensure_ascii=False,indent=2)
            
            #处理三元组变为一个子图
            
            print(answers[0])
        else:
            answers, used_triples = sllm.cot.cot(self.args, question, self.corpus, self.path)
            used_triples_text = ", ".join([f"[{h},{r},{t}]" for h, r, t in used_triples])
            return  {
                "Q": question,
                "A": answers[0],
                "used_triples": used_triples_text
            }
            

        return True
    
    def start(self, test=True):
        if test:
            print("test")
            with open("/home/wcy/code/KG-MedQA-v1.0/output/ceshi/llm-deepseek-chat__SentenceBERT__bs-10__20250408_184255/questions.txt", 'r', encoding='utf-8') as f:
                qs = f.readlines()
                qa_items = []
                count = 0
                for q in tqdm.tqdm(qs):
                    item = self.ask_question(q)
                    qa_items.append(item)
                    
                    count += 1
                    if count >= 20:
                        break
                # 1. 如果文件存在，先读取已有内容
                if os.path.exists(self.args.qa_output_path):
                    with open(self.args.qa_output_path, 'r', encoding='utf-8') as fin:
                        try:
                            qa_history = json.load(fin)
                        except json.JSONDecodeError:
                            qa_history = []  # 文件内容为空或损坏
                else:
                    qa_history = []
                # 2. 添加新记录
                qa_history.append(qa_items)
                
                # 3. 写回文件（覆盖写入）
                with open(self.args.qa_output_path, 'w', encoding='utf-8') as fout:
                    json.dump(qa_history,fout,ensure_ascii=False,indent=2)
                
                
        # 启动问答，循环等待用户输入
        else:
            print("Welcome to the Q&A system")
            while True:
               if not self.ask_question():
                break
            return True
            


    
