import re
import structllm as sllm
import ast
from sentence_transformers import SentenceTransformer,util
import torch
import re

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


def get_target_type(args, response, cgdata):
    pattern = r'\((.*?)\)'  # 正则表达式模式，匹配括号中的内容
    #pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, response)  # 查找所有匹配的内容
    result = ''
    list_rel = []
    if matches:
        result = matches[0]  # 获取第一个匹配的内容
        result = match = re.search(r"[\u4e00-\u9fff]+", result)[0]
        print(result)
        _, CG_relations = get_entitise_relations(cgdata)
        retriever = GetRetriever(args, CG_relations)
        top1_api = retriever.get_topk_candidates(1,result)
        label_rel = [CG_relations[char[0]] for char in top1_api]
    else:
        top1_api = None
        label_rel = None
    return top1_api, label_rel

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