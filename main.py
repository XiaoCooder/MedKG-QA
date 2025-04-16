
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
from sklearn.metrics import f1_score, recall_score, accuracy_score

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
                                for triple in triple_data:
                                    fout.write(f"[{triple['head']}, {triple['relation']}, {triple['tail']}]\n") 
                        with open(qa_data_path, 'a', encoding='utf-8') as fout:
                                for qa in qa_data:
                                    fout.write(f"Q: {qa['question']}\nA: {qa['answer']}\n\n")  # 问答对格式化                          

# 批量处理数据以提取知识图谱
async def EvalProcess(args, data, idx, api_key):
    args.key = api_key              
    if not args.debug:
        try:
            for i in range(len(data) // args.batch_size + (1 if len(data) % args.batch_size != 0 else 0)):
                start_index = i * args.batch_size
                end_index = min(start_index + args.batch_size, len(data))  # 确保不超过总长度
                # 提取一个批次
                subData = data[start_index:end_index]
                correct, matched, pred_labels, true_labels = await sllm.acc.evaluate_answer_quality(args,subData)
        except Exception as e:    
            if args.store_error:
                pass

    else:
        correct, matched, pred_labels, true_labels = await sllm.acc.evaluate_answer_quality(args, data)
    
    return correct, matched, pred_labels, true_labels
                    
                                    
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

def TxtRead1(args):
    print('load txt data...')
    sentences = []
    with open(args.data_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 去除前缀编号（如 1. 2.）
            if line[0].isdigit():
                dot_index = line.find('.')
                if 0 < dot_index < 4:
                    line = line[dot_index + 1:].strip()
            sentences.append(line)
    return sentences

# 解析应用程序的命令行参数
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # setting for openai
    parser.add_argument('--openai_url', default="", type=str, help='The url of openai')
    parser.add_argument('--key', default="api_key.txt", type=str, help='The key of openai or path of keys')
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
    parser.add_argument('--get_answer', default="structllm/prompt_/get_answer_and_triple.json", type=str, help='The prompt pth.')
    parser.add_argument('--get_answer1', default="structllm/prompt_/get_answer.json", type=str, help='The prompt pth.')
    parser.add_argument('--acc_prompt', default="structllm/prompt_/judge_acc.json", type=str, help='The prompt pth.')

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
    parser.add_argument('--test', default=True, type=bool)

    
    
    args = parser.parse_args()
    return args

# 合并多线程产生的
def merge_output_files(base_dir, num_process):
    merged_dir = os.path.join(base_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    merged_qa_path = os.path.join(merged_dir, "qa_pairs.txt")
    merged_triples_path = os.path.join(merged_dir, "triples.txt")

    with open(merged_qa_path, 'w', encoding='utf-8') as qa_out, \
         open(merged_triples_path, 'w', encoding='utf-8') as triple_out:

        for idx in range(num_process):
            sub_dir = os.path.join(base_dir, f"p-{idx:02d}")
            qa_file = os.path.join(sub_dir, "qa_pairs.txt")
            triple_file = os.path.join(sub_dir, "triples.txt")

            # 合并 qa_pairs
            if os.path.exists(qa_file):
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_out.writelines(f.readlines())

            # 合并 triples
            if os.path.exists(triple_file):
                with open(triple_file, 'r', encoding='utf-8') as f:
                    triple_out.writelines(f.readlines())

    print(f"合并完成，输出路径：{merged_qa_path}, {merged_triples_path}")



      
# 应用程序的主入口
async def main():
    args = parse_args()
    
    
    # if args.test:
    #     sllm.acc.evaluate_answer_quality(args)
    #     return
    
    
    #load api-key
    if not args.key.startswith("sk-"):
        with open(args.key, "r",encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) >= args.num_process, (len(all_keys), args.num_process)
    else:
        all_keys = []
        all_keys.append(args.key)
        
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
            data = TxtRead1(args)
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
                        merge_output_files(base_dir=args.output_path, num_process=args.num_process)
            
            #Q&A system
            await sllm.retrieve.get_path_collection_and_write(path = args.output_path ,encoder=encoder)
            args.qa_output_path = os.path.join(args.output_path, 'qa_history.txt')
            qa_bot = sllm.user_qa.user_qa(args)
            qa_bot.start( test=True, all_keys = all_keys )  
            
        elif user_input == "yes":

            #Q&A system
            response = await sllm.retrieve.get_output_path(encoder=encoder)  # 先 await 获取返回值
            path = [candidate_content.get('path') for candidate_content in response['metadatas'][0]][0]
            current_path = os.path.join(path,'merged')
            qa_pairs_path = os.path.join(current_path,'qa_pairs.txt')
            qs_path = sllm.graph.split_qa_pairs(qa_pairs_path)
            args.qa_output_path = os.path.join(current_path, 'qa_history.json')
            #读取数据库内容
            
            corpus_ = sllm.graph.graph(args,current_path)
            corpus = corpus_.load_triples()
            qa_bot = sllm.user_qa.user_qa(args, corpus, current_path, qs_path)
            await qa_bot.start(test = True, all_keys = all_keys)

            
            #评测模块
            with open(args.qa_output_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
                
            if args.num_process == 1:
                corrects, matcheds, pred_labels, true_labels = await EvalProcess(args, qa_data, -1, all_keys[0])
            else:
                num_each_split = int(len(qa_data) / args.num_process)
                split_data = []
                for idx in range(args.num_process):
                        start = idx * num_each_split
                        if idx == args.num_process - 1:
                            end = max((idx + 1) * num_each_split, len(qa_data))
                            split_data.append(qa_data[start:end])
                        else:
                            end = (idx + 1) * num_each_split
                            split_data.append(qa_data[start:end])
                async with Pool() as pool:
                        tasks = [pool.apply(EvalProcess, args=(args, split_data[idx], idx, all_keys[idx])) for idx in range(args.num_process)]
                        results = await asyncio.gather(*tasks)
                      
                            # 合并所有进程的结果
                corrects = sum([r[0] for r in results])
                matcheds = sum([r[1] for r in results])
                pred_labels = []
                true_labels = []
                for r in results:
                    pred_labels.extend(r[2])
                    true_labels.extend(r[3])
                    # 计算指标
                total = len(true_labels)  # 总样本数
                # 计算准确率、召回率和F1分数
                accuracy = corrects / total if total > 0 else 0
                recall = matcheds / total if total > 0 else 0
                # 确保两个列表长度一致
                min_len = min(len(true_labels), len(pred_labels))
                
                if min_len > 0:
                    f1 = f1_score(true_labels[:min_len], pred_labels[:min_len])
                else:
                    f1 = 0
                    
                # 打印结果
                print(f"Total samples: {total}")
                print(f"Accuracy (correct/total): {accuracy:.4f}")
                print(f"Recall (matched/total): {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                #sllm.acc.evaluate_answer_quality(args,args.qa_output_path)
            
        else: continue

if __name__=="__main__":
    asyncio.run(main())