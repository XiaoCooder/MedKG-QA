import structllm as sllm

class user_qa:
    def __init__(self, args):
       self.args = args
    
    def ask_question(self):
        question = input("please input your question: ")
        if question.lower() in ["exit"]:
            print("bye!")
            return False
        else:
            with open(self.args.qa_output_path, 'a') as fout:
                fout.write(f"Qustion : {question}\n")
            #Cot based on multi-modal Rerank
            rerank_result, context_rerank, summary_rerank, qas_rerank = sllm.rerank.rerank(self.args, question)
            #with open(args.qa_output_path, 'a') as fout:
            #    fout.write(f"Rerank result: {rerank_result}\n")   
            answer = sllm.cot.cot(self.args, question, rerank_result)
            with open(self.args.qa_output_path, 'a') as fout:
                fout.write(f"Answer : {answer}\n")
            print(answer)
        return True
    
    def start(self):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system, you can ask all the questions about the interview text")
            while True:
               if not self.ask_question():
                break
            return True
    
