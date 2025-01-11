import re
import structllm as sllm
import openai
from openai import OpenAI
import json

def Interview(args, data, character, names, descriptions):

    #args 系统设置
    #data：text ：访谈内容
    #character：访谈者编号
    #names :访谈者身份
    #description :访谈者介绍
   llm = sllm.llm.gpt(args)
   
   result_data = []
   result_data, qa_data ,summary_data, context_data = [], [], []
   def extarct_qa(input_text):
       sub_data = []
       
       
       
       
       
       
       
       return sub_data
   for i in range(len(data)):  
       mini_data = data[i]
       mini_character = character[i]
       max_retries = 3         # 最大重试次数
       retry_count = 0         # 当前重试次数
       total_num = 0           # 防止卡死
       result = None
       flag = True
       while (retry_count < max_retries and flag) :
        retry_count += 1
        #读取
        query_prompt = sllm.query_prompt.query_prompt(args, mini_data, mini_character, names, descriptions) #[配置文件 ,谈话内容 ，]
        ########1.清洗文本#######
        query_prompt.create_prompt(task = "clean")
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                result = response.choices[0].message.content
                result = result.strip('[]')
                result_data.append(result)
                print(names[character[i]-1]+" : "+result)
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
            total_num += 1 # 可得到结果
            
   flag = True
   while (retry_count < max_retries and flag ) :
        retry_count += 1
        ########2.提取信息#######
        query_prompt = sllm.query_prompt.query_prompt(args, data, character, names, descriptions)
        #
        query_prompt.create_prompt(task = "qa_extract")
        responses_qa = llm.get_response(query_prompt.naive_prompt)
        
        query_prompt.create_prompt(task = "summary_context")
        responses_sum = llm.get_response(query_prompt.naive_prompt)
        for response_qa in responses_qa:
            try:
                #解析response_qa
                result = response_qa.choices[0].message.content
                qa_sub_data = sllm.align.get_qa_parameter(result)
                for i in range(len(qa_sub_data)):
                    qa_data.append(qa_sub_data[i])
                    
                #解析response_sum
                result = responses_sum[0].choices[0].message.content 
                summary_data.append(result)
                
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
            total_num += 1 # 可得到结果
            
            
    ########3.注入数据库#######
   sllm.retrieve.get_qa_collection_and_write(args.encoder_model , qa_data = qa_data)
   sllm.retrieve.get_summary_collection_and_write(args.encoder_model , summarydata = summary_data)
   sllm.retrieve.get_context_collection_and_write(args.encoder_model , context = context_data)

   return result_data, result_data, qa_data, summary_data
