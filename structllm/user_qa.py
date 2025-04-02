import structllm as sllm

class user_qa:
    def __init__(self, args, corpus):
       self.args = args
       self.corpus = corpus

    def ask_question(self):

        question = input("please input your question: ")
        if question.lower() in ["exit"]:
            print("bye!")
            return False
        else:
            #with open(self.args.qa_output_path, 'a') as fout:
            #    fout.write(f"Qustion : {question}\n")  
            answer, triple_list = sllm.cot.cot(self.args, question, self.corpus)
            
            with open(self.args.qa_output_path, 'a') as fout:
                fout.write(f"Answer : {answer}\n")

            #处理三元组变为一个子图
            
            print(answer)

        return True
    
    def start(self):
        # 启动问答，循环等待用户输入
            print("Welcome to the Q&A system")
            while True:
               if not self.ask_question():
                break
            return True
    
