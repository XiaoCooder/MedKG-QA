import re
import structllm as sllm
import ast
from sentence_transformers import SentenceTransformer,util
import torch
import re
import json

class align:
    def __init__(self,args):
        self.args = args
        context_data, cleaned_data, summary_data = [], [], []
           
class SentenceBertRetriever:
    def __init__(self, corpus) -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        # self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        print(f"retrieve device:{self.device}")
        self.retrieve_model = SentenceTransformer(
            'xiaobu-embedding-v2',
            device=self.device
        )
        self.corpus = corpus

    def get_embedding(self, text):
        result = self.retrieve_model.encode(text)
        flat_array = result.tolist()
        return flat_array

    def get_topk_candidates(self, topk, query):
        # Corpus of documents and their embeddings
        api_corpus_embeddings = self.get_embedding(self.corpus)
        api_corpus_embeddings_list = []
        for i in range(len(api_corpus_embeddings)):
            api_corpus_embeddings_list.append(api_corpus_embeddings[i])
        # Queries and their embeddings
        queries_embeddings = self.get_embedding(query)
        queries_embeddings_list = []
        for i in range(len(queries_embeddings)):
            queries_embeddings_list.append(queries_embeddings[i])
        # Find the top-k corpus documents matching each query
        cos_scores = util.cos_sim(queries_embeddings_list, api_corpus_embeddings_list)
        # hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=topk)
        all_query_candidate_api_index = []
        for i in range(len(cos_scores)):
            hits = torch.argsort(cos_scores[i], descending=True)[:topk]
            all_query_candidate_api_index.append(hits.tolist())
        return all_query_candidate_api_index


    def count_accuracy(self, label, candidate):
        assert len(label) == len(candidate)

        topk_count = 0
        # hit = [0]*30
        # count = [0]*30
        for i in range(len(label)):
            # count[label[i]] += 1
            if label[i] in candidate[i]:
                topk_count += 1
                # hit[label[i]] += 1
        accuracy = topk_count / len(label)
        return accuracy

def GetRetriever(args, corpus):
    if args.retriever_align == "SentenceBERT":
        retriever = SentenceBertRetriever(corpus)
    else:
        pass
    return retriever

def get_parameters(text):
    names = []
    questions = []

    # 使用正则表达式分割数据项
    pattern = r"\{(.*?):(.*?)\}"

    matches = re.findall(pattern, text)

    # 提取姓名和问题
    for match in matches:
       names.append(match[0].strip())  # 姓名
       questions.append(match[1].strip())  # 问题
    return names, questions

def get_triples(text):
    """
    使用正则表达式从大模型生成的回答中提取三元组。
    
    参数：
        text (str): 大模型的回答文本，其中包含形如 [a, b, c] 的三元组
    
    返回：
        list: 三元组列表，每个三元组为一个包含三个字符串的列表
    """
    # 定义正则表达式：
    # \[ 和 \] 匹配中括号
    # (.*?) 使用非贪婪匹配提取每个元素，逗号为分隔符
    pattern = r'\[\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*\]'
    # 使用 re.findall 提取所有匹配的三元组，返回的结果是一个元组列表
    matches = re.findall(pattern, text)
    # 将元组转换为列表形式，并去除每个元素两边的空白字符
    triples = [[item.strip() for item in match] for match in matches]
    return triples

def get_chunk_id(result, chunk_ids):
    #transfer rerank string into chunk_id list
    matches = re.findall(r'\[.*?\]', result)
    #import pdb; pdb.set_trace()
    #result_array = ast.literal_eval(matches)
    chunk_id_list = json.loads(matches[0])
    return chunk_id_list