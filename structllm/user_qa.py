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
    
    def process_web_question(self, question):
        """
        处理从 Web 接口来的问题并返回答案
        
        参数:
            question (str): 用户的问题
        
        返回:
            str: 回答
        """
        try:
            print("开始处理问题...")
            
            # 使用asyncio来运行异步函数
            # import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # 与ask_question相同的方式调用异步函数
            result = loop.run_until_complete(sllm.cot.cot(self.args, question, self.corpus))
            loop.close()
            
            answers, used_triples = result
            
            # 格式化三元组为文本
            used_triples_text = ", ".join([f"[{h},{r},{t}]" for h, r, t in used_triples])
            
            print("问答处理完成")
            # 创建QA项
            qa_item = {
                "Q": question,
                "A": answers[0] if answers else "无法回答这个问题",
                "used_triples": used_triples_text
            }
            
            # 读取现有QA历史
            if os.path.exists(self.args.qa_output_path):
                with open(self.args.qa_output_path, 'r', encoding='utf-8') as fin:
                    try:
                        qa_history = json.load(fin)
                    except json.JSONDecodeError:
                        qa_history = []  # 文件内容为空或损坏
            else:
                qa_history = []
            
            # 添加新记录
            qa_history.append(qa_item)
            
            # 写回文件
            with open(self.args.qa_output_path, 'w', encoding='utf-8') as fout:
                json.dump(qa_history, fout, ensure_ascii=False, indent=2)
            
            # 返回答案
            return answers[0] if answers else "无法回答这个问题"
        
        except Exception as e:
            # 错误处理
            error_message = f"处理问题时出错: {str(e)}"
            print(error_message)
            return error_message

    def start(self):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system")
            while True:
               if not self.ask_question():
                break
            return True
            


    
