import structllm as sllm
import argparse
import openai
import re

def getparameter(text):
    data = "{Chris Anderson:How does Twitter operate?},{Chris Anderson:Tim Evans once said that we are like a monkey with a computer attached to it. What does this mean?},{Chris Anderson:How can we turn the tiny straw of communication bandwidth with the tertiary layer of intelligence into a large highway?},{Chris Anderson:In the best - case scenario, what new human possibilities might we discover?},{Chris Anderson:If AI were to threaten Earth, what would we need?},{Chris Anderson:Last summer, we discussed reusability, and Elon Musk had just demonstrated it spectacularly for the first time. Since then, what has been developed?},{Chris Anderson:What is the holy grail of rocketry or space transport?},{Chris Anderson:With Starship, what is the goal?},{Chris Anderson:What is the main design of Starship?},{Chris Anderson:When will a Starship go to Mars for the first time, presumably without people but with equipment?},{Chris Anderson:When will Starship carry people?},{Chris Anderson:When will Starship transport around 100 people at a time?},{Chris Anderson:What is the expected cost of Starship putting 100 tons into orbit?},{Chris Anderson:What propellants does Starship use?},{Chris Anderson:One of the first tasks on Mars will be to create what?}"

    names = []
    questions = []

    # 使用正则表达式分割数据项
    pattern = r"\{(.*?):(.*?)\}"

    matches = re.findall(pattern, text)

    # 提取姓名和问题
    for match in matches:
       names.append(match[0].strip())  # 姓名
       questions.append(match[1].strip())  # 问题

# 输出结果
    return names, questions

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    # setting for openai
    parser.add_argument('--openai_url', default="https://api.deepseek.com/v1", type=str, help='The url of openai')
    parser.add_argument('--key', default="sk-bd9f56da3a114296b3de026382a4827c", type=str, help='The key of openai or path of keys')
    
    # input data path
    parser.add_argument('--folder_path', default="dataset/WikiSQL_TB_csv/test", type=str, help='The CSV data pth.')
    parser.add_argument('--data_path', default="dataset/WikiSQL_CG", type=str, help='The CG data pth.')
    parser.add_argument('--character_path', default="input/character.txt", type=str, help='')
    parser.add_argument('--clean_prompt_path', default="structllm/prompt_/clean_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_q_prompt_path', default="structllm/prompt_/query_quest_P.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_a_prompt_path', default="structllm/prompt_/extract_a_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--summary_prompt_path', default="structllm/prompt_/summary_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--batch_size', default="10", type=int, help='The prompt pth.')
    
    # setting model
    parser.add_argument('--model', default="deepseek-chat", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--encoder_model', default="SentenceBERT", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    
    # output
    parser.add_argument('--store_error', action="store_true", default=True)
    parser.add_argument('--error_file_path', default="timeout_file.txt", type=str)
    parser.add_argument('--output_result_path', default="output/output_result.txt", type=str)
    parser.add_argument('--output_path', default="output/output_result.txt", type=str)
    parser.add_argument('--debug', default=0, type=int)
    
    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    args = parse_args()
    llm = sllm.llm.gpt(args)
    max_retries = 3         # 最大重试次数
    retry_count = 0         # 当前重试次数
    total_num = 0           # 防止卡死
    result = None
    data = []
    with open('/home/wcy/code/InterviewSystem-v0.1/output/test_data.txt', 'r', encoding='utf-8') as file:
      for line in file:
          data.append(line.strip()) 
   
    for i in range(len(data) // args.batch_size + (1 if len(data) % args.batch_size != 0 else 0)):
            start_index = i * args.batch_size
            end_index = min(start_index + args.batch_size, len(data))  # 确保不超过总长度
            # 提取一个批次
            context_data = data[start_index:end_index]
            print(f"chunk {i}")
            
            questions, answers = [], []
            flag = True
            max_retries = 3 

            while ( flag ) :
                    retry_count += 1
                    ########2.提取信息#######
                    #提取qa缓存
                    query_prompt = sllm.query_prompt.query_prompt(args, context_data)
                    query_prompt.create_prompt(task = "extract_q" )
                    responses_q = llm.get_response(query_prompt.naive_prompt)
                    
                    #提取summary缓存
                    #query_prompt.create_prompt(task = "summary_context")
                    #responses_sum = llm.get_response(query_prompt.naive_prompt)
                    
                    for response_q in responses_q:
                        try:
                            #解析response_qa
                            result = response_q.choices[0].message.content
                            #print(result)
                            with open("/home/wcy/code/InterviewSystem-v0.1/output/test_output.txt", "a", encoding="utf-8") as file:
                                      file.write(result+"\n")
                            names, questions = getparameter(result)
                            for i in range(len(names)):
                                 print(names[i]+":"+questions[i])
                                 
                            #questions = sllm.align.get_q_parameter(result)
                            #for i in range(len(questions)):
                            #    q.append(questions[i])
                            
                            #解析response_sum
                            #result = responses_sum[0].choices[0].message.content 
                            #summary_data = result 
                            #print(result)
                            
                        except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等
                            print(e)
                            total_num += 1
                            continue

                        except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
                            print(e)
                            total_num += 1 # 防止卡死
                            continue

                        except openai.APITimeoutError as e: # 超时
                            print(e)
                            total_num += 1 # 防止卡死
                            continue

                        except ValueError as e: # maximum context length
                            print(e)
                            continue

                        except Exception as e: # 其他错误
                            print(e)
                            total_num += 1 # 防止卡死
                            continue
                    flag = False
                    
            """
            while (retry_count < max_retries):
                   query_prompt = sllm.query_prompt.query_prompt(args, context_data)
                   for i in range(len(questions)):
                       query_prompt.create_prompt(task = "extract_a",question = question)
                       responses_a = llm.get_response(query_prompt.naive_prompt) 
                   for response_a in responses_a:
                        try:
                            #解析response_a
                            result = response_a.choices[0].message.content
                            with open("/home/wcy/code/InterviewSystem-v0.1/output/test_output.txt", "a", encoding="utf-8") as file:
                                      file.write(result+"\n")
                            answers.append(response_a)
                            
                        except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等
                            print(e)
                            total_num += 1
                            continue

                        except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
                            print(e)
                            total_num += 1 # 防止卡死
                            continue

                        except openai.APITimeoutError as e: # 超时
                            print(e)
                            total_num += 1 # 防止卡死
                            continue

                        except ValueError as e: # maximum context length
                            print(e)
                            continue

                        except Exception as e: # 其他错误
                            print(e)
                            total_num += 1 # 防止卡死
                            continue
                    flag = False
            """
                
                