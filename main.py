import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
from collections import defaultdict
import re

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def InterviewProcess(args,Data,Cha,Names,Descriptions):
                
                if not args.debug:
                    try:
                        for i in range(len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)):
                          start_index = i * args.batch_size
                          end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          subCha = Cha[start_index:end_index]
                          print(f"*************chunk {i}*************\n")
                          cleaned_data, qa_data, summary_data = sllm.Interview.Interview(args,subData,subCha,Names,Descriptions)
                    except Exception as e:    
                        if args.store_error:
                            pass

                else:
                    cleaned_path = os.path.join(args.output_path, 'cleaned_text.txt')
                    qa_path = os.path.join(args.output_path, 'qa_pairs.txt')
                    summary_path = os.path.join(args.output_path, 'summary.txt')
                    check_path(cleaned_path)
                    num_batches = len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)
                    for i in tqdm(range(num_batches), desc="Processing batches"):
                          start_index = i * args.batch_size
                          end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          subCha = Cha[start_index:end_index]
                          #print(f"chunk {i}")
                          cleaned_data, qa_data, summary_data = sllm.Interview.Interview(args,subData,subCha,Names,Descriptions,i)
                          #save chunk
                          with open(cleaned_path, 'a') as fout:
                                 fout.write(f"*************chunk {i}*************\n")
                          with open(qa_path, 'a') as fout:
                                 fout.write(f"*************chunk {i}*************\n")
                          with open(summary_path, 'a') as fout:
                                 fout.write(f"*************chunk {i}*************\n")

                          for i in range(len(cleaned_data)):
                            with open(cleaned_path, 'a') as fout:
                                 fout.write(cleaned_data[i])
                          for idx,element in enumerate(qa_data):
                            n,q,a = element
                            with open(qa_path, 'a') as fout:
                                 fout.write(f"Question {idx}"+" "+n+":"+q+" Answer:"+a+"\n")
                          with open(summary_path, 'a') as fout:
                                 fout.write(summary_data+"\n")

                        
                          
                          
                          
                          
def InterviewRead(args):
    print('load Inteview data...')
    with open(args.data_path, "r", encoding="utf8") as fin:
        lines = fin.readlines()
    data = []  # 用来存储说话人的内容
    character = []  # 用来存储说话人的序号
    speaker = None  # 当前说话人
    content = ""  # 当前发言内容

    # 正则表达式：提取说话人编号和时间戳
    speaker_pattern = re.compile(r"说话人(\d)")

    # 逐行分析文件内容
    for line in lines:
        line = line.strip()  # 去掉行首尾的空格和换行符
        # 如果是一个新的说话人，记录当前发言并准备处理新的发言
        match = speaker_pattern.match(line)
        if match:
            if speaker is not None:  # 如果之前有说话人，保存之前的内容
                data.append(content.strip())
                character.append(speaker)
            speaker = int(match.group(1))  # 更新当前说话人
            content = ""  # 重置内容
            #print(f"Matched Speaker: {speaker}")
        else:
            content += line + " "

    # 最后一条内容处理（文件结束时）
    if speaker is not None:
        data.append(content.strip())
        character.append(speaker)
    print(f"length of Interview : {len(data)}")
    return data,character


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # setting for openai
    parser.add_argument('--openai_url', default="", type=str, help='The url of openai')
    parser.add_argument('--key', default="", type=str, help='The key of openai or path of keys')
    parser.add_argument('--embedding_key', default="", type=str, help='The key of openai or path of keys')
    parser.add_argument('--dynamic_open', default=True, type=bool, help='The key of openai or path of keys')

    # input data path
    parser.add_argument('--folder_path', default="dataset/WikiSQL_TB_csv/test", type=str, help='The CSV data pth.')
    parser.add_argument('--data_path', default="dataset/WikiSQL_CG", type=str, help='The CG data pth.')
    parser.add_argument('--character_path', default="input/character.txt", type=str, help='')
    
    #prompt path
    parser.add_argument('--clean_prompt_path', default="structllm/prompt_/clean_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_q_prompt_path', default="structllm/prompt_/extract_q_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_a_prompt_path', default="structllm/prompt_/extract_a_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--summary_prompt_path', default="structllm/prompt_/summary_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--reranker_prompt', default="structllm/prompt_/reranker_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--qa_prompt', default="structllm/prompt_/qa_prompt.json", type=str, help='The prompt pth.')
    
    # setting model
    parser.add_argument('--model', default="gpt-3.5-turbo", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--encoder_model', default="SentenceBERT", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--retriever_align', default="SentenceBERT", type=str, help='')
    parser.add_argument('--batch_size', default="10", type=int, help='The prompt pth.')

    # output
    parser.add_argument('--store_error', action="store_true", default=True)
    parser.add_argument('--error_file_path', default="timeout_file.txt", type=str)
    parser.add_argument('--output_path', default="output/output_result.txt", type=str)
    parser.add_argument('--qa_output_path', default="output/qa_history.txt", type=str)
    parser.add_argument('--chroma_dir', default="chroma", type=str, help='The chroma dir.')
    #others
    parser.add_argument('--debug', default=0, type=int)
    
    args = parser.parse_args()
    return args

def CharacterRead(args):
    # 用于存储名字和描述
    names = []
    descriptions = []
    
    # 正则表达式：提取名字和描述（假设格式为 '名字: 描述'）
    pattern = re.compile(r'([^:]+):\s*(.+)')

    # 逐行读取文件
    with open(args.character_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()  # 去掉首尾空格和换行符

            # 使用正则表达式匹配
            match = pattern.match(line)
            if match:
                name = match.group(1)  # 提取名字
                description = match.group(2)  # 提取描述
                names.append(name)
                descriptions.append(description)
    print(f"Number of Speakers : {len(names)-1}")
    return names, descriptions


  
      
if __name__=="__main__":
    args = parse_args()
    #load api-key
    if not args.key.startswith("sk-"):
        with open(args.key, "r",encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
    args.key = all_keys[0]
    
    flag =True
    while (True):
        if (flag):
            user_input = input("I noticed that there is already data in your database. \nDo you need to open the Q&A directly? \nIf not, I will process the new input data\n").strip().lower()
            flag = False
        else:
            user_input = input("Invalid input. Please enter 'yes' or 'no'.\n")
        if user_input == "no":
            #create DB （如果第一次安装可能没有数据）
            sllm.retrieve.get_collection(args.encoder_model,name="qas" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.get_collection(args.encoder_model,name="context" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.get_collection(args.encoder_model,name="summary" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.get_collection(args.encoder_model, name="path" ,chroma_dir= args.chroma_dir)
            #reset DB （避免遗留数据）
            sllm.retrieve.rebuild_collection(args.encoder_model,name="qas" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.rebuild_collection(args.encoder_model,name="context" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.rebuild_collection(args.encoder_model,name="summary" ,chroma_dir= args.chroma_dir)
            sllm.retrieve.rebuild_collection(args.encoder_model,name="path" ,chroma_dir= args.chroma_dir)

            #loda interview data
            InterviewData,InterviewCha = InterviewRead(args)
            Names, Descriptions = CharacterRead(args)
            #process
            InterviewProcess(args,InterviewData,InterviewCha,Names,Descriptions)
            #Q&A system
            sllm.retrieve.get_path_collection_and_write(args.encoder_model, path = args.output_path)
            args.qa_output_path = os.path.join(args.output_path, 'qa_history.txt')
            qa_bot = sllm.user_qa.user_qa(args)
            qa_bot.start()  
            
        elif user_input == "yes":
            #Q&A system
            path = [candidate_content.get('path') for candidate_content in sllm.retrieve.get_output_path(args.encoder_model)['metadatas'][0]][0]
            args.qa_output_path = os.path.join(path, 'qa_history.txt')
            print(args.qa_output_path)
            qa_bot = sllm.user_qa.user_qa(args)
            qa_bot.start()  
        
        else: continue

    