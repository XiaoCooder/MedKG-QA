import structllm as sllm
import json
import os

class user_qa:
    def __init__(self, args, corpus, path):
        self.args = args
        self.corpus = corpus
        self.path = path

    def ask_question(self):

        question = input("please input your question: ")
        if question.lower() in ["exit"]:
            print("bye!")
            return False
        else: 
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

        return True
    
    def process_web_question(self, question):
        """
        处理从 Web 接口来的问题并返回答案
    
        参数:
            question (str): 用户的问题
        
        返回:
            str: 回答
        """
    # 记录问题
        with open(self.args.qa_output_path, 'a') as fout:
            fout.write(f"Question : {question}\n")
        
    # 使用 rerank 和 cot 获取答案
        rerank_result, context_rerank, summary_rerank, qas_rerank = sllm.rerank.rerank(self.args, question)
        answer = sllm.cot.cot(self.args, question, rerank_result)
    
    # 记录答案
        with open(self.args.qa_output_path, 'a') as fout:
            fout.write(f"Answer : {answer}\n")
        
        return answer


    def start(self):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system")
            while True:
               if not self.ask_question():
                break
            return True


    
