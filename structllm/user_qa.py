import structllm as sllm
import json
import os
from tqdm import tqdm
import asyncio
from aiomultiprocess import Pool

class user_qa:
    def __init__(self, args, corpus, path, qs_path):
        self.args = args
        self.corpus = corpus
        self.path = path
        self.qs_path = qs_path

    async def ask_question(self, question = ""):
        if question.__eq__(""): 
            question = input("please input your question: ")
            if question.lower() in ["exit"]:
                print("bye!")
                return False
            answers, used_triples = sllm.cot.cot(self.args, question, self.corpus)
            used_triples_text = ", ".join([f"[{h},{r},{t}]" for h, r, t in used_triples])
            if not answers:
                answers = ["未找到答案"]  # 如果是空列表，设置一个默认值
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
            answers, used_triples = await sllm.cot.cot(self.args, question, self.corpus)
            used_triples_text = ", ".join([f"[{h},{r},{t}]" for h, r, t in used_triples])
            if not answers:
                answers = ["未找到答案"]  # 如果是空列表，设置一个默认值
            return  {
                "Q": question,
                "A": answers[0],
                "used_triples": used_triples_text
            }
        return True
    
    async def QAProcess(self, args, data, idx, key):
        
        print(idx)
        if idx == -1:
            describe = "process"
        else:
            idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
            describe = "p-" + idx
            
        qa_items = []
        args.key = key
        
        for q in tqdm(data, desc = describe):
            item = await self.ask_question(q)
            qa_items.append(item)
        return qa_items
        
        
    async def start(self, test=True, all_keys = []):
        if test:
            with open(self.qs_path, 'r', encoding='utf-8') as f:
                data = f.readlines()
                
            if self.args.num_process == 1:
                results = await self.QAProcess(self.args, data, -1, all_keys[0])
            else:
                num_each_split = int(len(data) / self.args.num_process)
                split_data = []
                for idx in range(self.args.num_process):
                        start = idx * num_each_split
                        if idx == self.args.num_process - 1:
                            end = max((idx + 1) * num_each_split, len(data))
                            split_data.append(data[start:end])
                        else:
                            end = (idx + 1) * num_each_split
                            split_data.append(data[start:end])
                async with Pool() as pool:
                        tasks = [pool.apply(self.QAProcess, args=(self.args, split_data[idx], idx, all_keys[idx])) for idx in range(self.args.num_process)]
                        results = await asyncio.gather(*tasks)
            #写入
            qa_items = []
            for r in results:
                qa_items.extend(r)
                
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
                json.dump(qa_history, fout, ensure_ascii=False, indent=2)
                
                
        # 启动问答，循环等待用户输入
        else:
            print("Welcome to the Q&A system")
            while True:
               if not self.ask_question():
                break
            return True
            


    
