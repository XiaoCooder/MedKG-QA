
import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import re
import asyncio
from aiomultiprocess import Pool
from sentence_transformers import SentenceTransformer
import torch
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 检查路径是否存在，如果不存在则创建目录
def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

# 批量处理数据以提取知识图谱
async def KGProcess(args, Data, idx, api_key , encoder):
    
                args.key = api_key
                
                if idx == -1:
                        output_path= args.output_path
                        describe = "process"
                else:
                        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
                        check_path(args.output_path)
                        output_path = args.output_path + "/p-" + idx
                        describe = "p-" + idx
                        check_path(output_path)
                        
                print("Start PID %d and save to %s" % (os.getpid(), output_path))
                
                if not args.debug:
                    try:
                        for i in range(len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)):
                          start_index = i * args.batch_size
                          end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          print(f"*************chunk {i}*************\n")
                          triple_data, qa_data= await sllm.ExtractKG.ExtractKGQA(args,subData,encoder)
                    except Exception as e:    
                        if args.store_error:
                            pass

                else:
                    triple_data_path = os.path.join(output_path, 'triples.txt')
                    qa_data_path = os.path.join(output_path, 'qa_pairs.txt')
                    check_path(triple_data_path)
                    check_path(qa_data_path)
                    num_batches = len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)
                    
                    for i in tqdm(range(num_batches), desc = describe):
                        start_index = i * args.batch_size
                        end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                        subData = Data[start_index:end_index]
                        
                        triple_data, qa_data = await sllm.ExtractKG.ExtractKGQA(args, subData, encoder)
                        #save chunk
                        with open(triple_data_path, 'a', encoding='utf-8') as fout:
                                fout.write(f"*************chunk {i}*************\n")
                                for triple in triple_data:
                                    fout.write(f"[{triple['head']}, {triple['relation']}, {triple['tail']}]\n") 
                        with open(qa_data_path, 'a', encoding='utf-8') as fout:
                                fout.write(f"*************chunk {i}*************\n")
                                for qa in qa_data:
                                    fout.write(f"Q: {qa['question']}\nA: {qa['answer']}\n\n")  # 问答对格式化                          

# 读取并解析文本数据为完整句子
def TxtRead(args):
    print('load txt data...')
    with open(args.data_path, "r", encoding="utf8") as fin:
        raw_lines = fin.readlines()

    # 先去除空行和两端空白
    lines = [line.strip() for line in raw_lines]

    sentences = []  # 存储合并后的完整句子
    current_sentence = ""  # 用来暂存当前正在组合的句子

    # 定义句子结束的标点符号，可根据需要进行扩展
    sentence_endings = ("。", "!", "?")

    for line in lines:
        if current_sentence:
            # 句子拼接时在行间添加空格
            current_sentence += " " + line
        else:
            current_sentence = line

        # 如果当前行以句子结束标点结尾，则认为是完整句子
        if line.endswith(sentence_endings):
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # 如果文件最后一部分没有以结束标点结束，也将其作为一句
    if current_sentence:
        sentences.append(current_sentence.strip())

    print(f"文本共 {len(sentences)} 个完整句子")
    return sentences

# 解析应用程序的命令行参数
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
    parser.add_argument('--extractKG_prompt', default="structllm/prompt_/extractKG.json", type=str, help='The prompt pth.')
    parser.add_argument('--extractQA', default="structllm/prompt_/extractKGQA.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_keywords', default="structllm/prompt_/extract_keywords.json", type=str, help='The prompt pth.')

    # setting model
    parser.add_argument('--model', default="gpt-3.5-turbo", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--encoder_model', default="SentenceBERT", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--retriever_align', default="SentenceBERT", type=str, help='')
    parser.add_argument('--batch_size', default="10", type=int, help='The prompt pth.')
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    
    # output
    parser.add_argument('--output_path', default="output/output_result.txt", type=str)
    parser.add_argument('--qa_output_path', default="output/qa_history.txt", type=str)

    #others
    parser.add_argument('--debug', default=0, type=int)
    
    args = parser.parse_args()
    return args

# 合并多线程产生的
def merge_chunks(args):
    """
    合并分块文件并恢复原始顺序
    
    :param input_dir: 分块文件所在目录
    :param output_file: 合并后的输出文件
    :param split_sizes: 每个分块的行数列表
    """
    # 确保split_sizes是整数列表
    data = []
    # 读取所有分块文件内容
    for i in range(chunk_num):
        chunks = []
        idx = "0" + str(idx) if i < 10 else str(i)  # 00 01 02 ... 29
        input_dir = args.output_path + "/p-" + idx
        chunk_file = os.path.join(input_dir, f"chunk_{i}.txt")
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks.append(f.readlines())
        data.append(chunks)
    
    # 合并并恢复原始顺序
    merged_lines = []
    for chunk in chunks:
        # 过滤空行和分块标记行
        filtered = [line for line in chunk 
                   if line.strip() and not line.startswith('***')]
        merged_lines.extend(filtered)
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(merged_lines)
    
    merge_chunks(
        input_dir="path/to/chunk/files",
        output_file="merged_output.txt",
        split_sizes=split_sizes
    )


      
# 应用程序的主入口
async def main():
    args = parse_args()
    #load api-key
    if not args.key.startswith("sk-"):
        with open(args.key, "r",encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) >= args.num_process, (len(all_keys), args.num_process)
    
    encoder = sllm.retrieve.Encoder(args.encoder_model)
    flag =True

    while (True):
        if (flag):
            user_input = input("I noticed that there is already data in your database. \nDo you need to open the Q&A directly? \nIf not, I will process the new input data\n").strip().lower()
            flag = False
        else:
            user_input = input("Invalid input. Please enter 'yes' or 'no'.\n")
        if user_input == "no":
            #create DB （如果第一次安装可能没有数据）
            await sllm.retrieve.get_collection(name="path" , encoder = encoder)
            await sllm.retrieve.get_collection(name="triples" , encoder = encoder)
            await sllm.retrieve.get_collection(name="triple_head" , encoder = encoder)
            await sllm.retrieve.get_collection(name="triple_relation" , encoder = encoder)
            await sllm.retrieve.get_collection(name="triple_tail" , encoder = encoder)
            await sllm.retrieve.get_collection(name="qa_pairs" , encoder = encoder)
            #reset DB （避免遗留数据）
            await sllm.retrieve.rebuild_collection(name="path" ,encoder = encoder)
            await sllm.retrieve.rebuild_collection(name="triples" ,encoder = encoder)
            await sllm.retrieve.rebuild_collection(name="triple_head" ,encoder = encoder)
            await sllm.retrieve.rebuild_collection(name="triple_relation" ,encoder = encoder)
            await sllm.retrieve.rebuild_collection(name="triple_tail" ,encoder = encoder)
            await sllm.retrieve.rebuild_collection(name="qa_pairs" ,encoder = encoder)
            
            #loda interview data
            data = TxtRead(args)
            
            #process
            if args.num_process == 1:
                await KGProcess(args, data, -1, all_keys[0], encoder)
            else:
                num_each_split = int(len(data) / args.num_process)
                split_data = []
                for idx in range(args.num_process):
                        start = idx * num_each_split
                        if idx == args.num_process - 1:
                            end = max((idx + 1) * num_each_split, len(data))
                            split_data.append(data[start:end])
                        else:
                            end = (idx + 1) * num_each_split
                            split_data.append(data[start:end])
                async with Pool() as pool:
                        tasks = [pool.apply(KGProcess, args=(args, split_data[idx], idx, all_keys[idx], encoder)) for idx in range(args.num_process)]
                        await asyncio.gather(*tasks)
                #merge txt
                #merge_chunks(args)
            
            #Q&A system
            await sllm.retrieve.get_path_collection_and_write(path = args.output_path ,encoder=encoder)
            args.qa_output_path = os.path.join(args.output_path, 'qa_history.txt')
            qa_bot = sllm.user_qa.user_qa(args)
            qa_bot.start()  
            
        elif user_input == "yes":

            #Q&A system
            response = await sllm.retrieve.get_output_path(encoder=encoder)  # 先 await 获取返回值
            path = [candidate_content.get('path') for candidate_content in response['metadatas'][0]][0]
            qa_pairs_path = '/home/wcy/code/KG-MedQA-v1.0/output/ceshi/llm-deepseek-chat__SentenceBERT__bs-10__20250331_201025/qa_pairs.txt'
            sllm.graph.triplesProcess(args, path, qa_pairs_path)
            import pdb;pdb.set_trace()
            
            args.qa_output_path = os.path.join(path, 'qa_history.txt')
            print(args.qa_output_path)
            qa_bot = sllm.user_qa.user_qa(args)
            qa_bot.start()  
        
        else: continue

if __name__=="__main__":
    asyncio.run(main())