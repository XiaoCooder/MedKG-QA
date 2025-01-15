import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
from collections import defaultdict
import re
def InterviewProcess(args,Data,Cha,Names,Descriptions):
    output_result_path = args.output_result_path
    print("Start PID %d and save to %s" % (os.getpid(), output_result_path))

    with open(output_result_path+".txt", "w",encoding='utf-8') as fresult:
                # fdetail.write(f"=== Answer:{answer}\n")
                if not args.debug:
                    try:
                        for i in range(len(Data) // args.batch_size):
                          start_index = i * args.batch_size
                          end_index = start_index + args.batch_size
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          subCha = Cha[start_index:end_index]
                          result = sllm.Interview.Interview(args,subData,subCha,Names,Descriptions)
                          #sys.stdout = sys.__stdout__  # 恢复标准输出流
                          #result_dict = dict()
                          #fresult.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                          #fresult.flush()

                    except Exception as e:    
                        if args.store_error:
                            #with open(error_path,"a",encoding='utf-8') as f:
                               #f.write(json.dumps(tmp_dict, ensure_ascii=False) + "\n")
                            pass

                else:
                    # sys.stdout = fdetail
                    for i in range(len(Data) // args.batch_size):
                          start_index = i * args.batch_size
                          end_index = start_index + args.batch_size
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          subCha = Cha[start_index:end_index]
                          result_data = sllm.Interview.Interview(args,subData,subCha,Names,Descriptions)
                          
                    # result = [ list(sample) if type(sample)==set else sample for sample in result ]
                    #print(f"result:{result}, output_result_path:{output_result_path}")
                    result_dict = dict()
                    fresult.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                    fresult.flush()

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
    parser.add_argument('--prompt_path', default="structllm/prompt_/wikisql.json", type=str, help='The prompt pth.')
    parser.add_argument('--clean_prompt_path', default="structllm/prompt_/wikisql.json", type=str, help='The prompt pth.')
    parser.add_argument('--batch_size', default="10", type=int, help='The prompt pth.')
    
    # setting model
    parser.add_argument('--model', default="gpt-3.5-turbo", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--encoder_model', default="SentenceBERT", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    
    # output
    parser.add_argument('--store_error', action="store_true", default=True)
    parser.add_argument('--error_file_path', default="timeout_file.txt", type=str)
    parser.add_argument('--output_result_path', default="output/output_result.txt", type=str)
    parser.add_argument('--output_path', default="output/output_result.txt", type=str)
    
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
    #创建数据库 （如果第一次安装可能没有数据）
    sllm.retrieve.get_collection(args.encoder_model,name="qas" ,chroma_dir= args.chroma_dir)
    sllm.retrieve.get_collection(args.encoder_model,name="context" ,chroma_dir= args.chroma_dir)
    sllm.retrieve.get_collection(args.encoder_model,name="summary" ,chroma_dir= args.chroma_dir)
    #重置数据库 （避免遗留数据）
    sllm.retrieve.rebuild_collection(args.encoder_model,name="qas" ,chroma_dir= args.chroma_dir)
    sllm.retrieve.rebuild_collection(args.encoder_model,name="context" ,chroma_dir= args.chroma_dir)
    sllm.retrieve.rebuild_collection(args.encoder_model,name="summary" ,chroma_dir= args.chroma_dir)
  #载入语言模型的api-key
    if not args.key.startswith("sk-"):
        with open(args.key, "r",encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
    args.key = all_keys[0]

  #把数据读入到程序中
    InterviewData,InterviewCha = InterviewRead(args)
    Names, Descriptions = CharacterRead(args)
  
  #执行
    InterviewProcess(args,InterviewData,InterviewCha,Names,Descriptions)
  
  #问答系统
    qa_bot = sllm.user_qa(args)
    qa_bot.start()  
