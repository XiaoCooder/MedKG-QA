import os
import chromadb
import torch
from torch import Tensor

from tqdm.autonotebook import trange
from typing import List, Union, TypeVar, Dict
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from chromadbx import NanoIDGenerator

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)



def get_embedding_BERT(text, model="stella_en_400M_v5") -> List[float]:
    """ 用于对文本进行编码，编码器是stella_en_400M_v5的,返回值是float list """
    #print(text)
    retrieve_model = SentenceTransformer(
        model_name_or_path=model,
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    if isinstance(text, str):
        result = retrieve_model.encode([text])
        flat_array = result.tolist()
        return flat_array
    elif isinstance(text, list):
        result = retrieve_model.encode(text)
        flat_array = result.tolist()
        return flat_array

class NewEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def __call__(self, input: D) -> Embeddings:
        embeddings = self.encoder.encode(input)
        return embeddings

class EncoderBERT:
    def encode(
        self,
        text: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[Tensor]:
        text_embeddings = []
        for batch_start in trange(
            0, len(text), batch_size, disable=show_progress_bar
        ):
            batch_end = batch_start + batch_size
            batch_text = text[batch_start:batch_end]
            #print(f"Batch {batch_start} to {batch_end-1}")
            assert "" not in batch_text
            resp = get_embedding_BERT(batch_text)
            """for i, be in enumerate(resp):
                assert (
                        i == be.index
                )  # double check embeddings are in same order as input
                """
            batch_text_embeddings = [e for e in resp]
            text_embeddings.extend(batch_text_embeddings)
        return text_embeddings

class BERT:
    def __init__(self) -> None:
        self.q_model = EncoderBERT()
        self.doc_model = self.q_model

    def encode(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> List[Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

class Encoder:

    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if encoder_name == "SentenceBERT":
            self.encoder = BERT()
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                "stella_en_400M_v5",
                "cuda:0" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
        else :pass

def _get_embedding_and_save_to_chroma(
    data: List[Dict[str, str]],
    collection: Collection,
    encoder: Encoder,
    batch_size: int = 64,
):
    """ 将编码存入chroma向量数据库 """
    encoder_ = encoder.encoder

    docs = [item["question"] for item in data] #question list
    meta_keys = list(data[0].keys())           #data中dict的keys
    del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=batch_size, show_progress_bar=True)
    # else:
    #     embeddings = encoder_.doc_model.encode(
    #         docs, batch_size=batch_size, show_progress_bar=True
    #     )
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()

    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in range(len(docs))],
        )
    return collection

#prompt存储
def put_embedding_prompt(retriever: str, prompt_data: list = None, collection = None):

    encoder = Encoder(retriever)
    retrieve_data = []

    for idx , elements in enumerate(prompt_data):
        k,v = elements[0],elements[1]
        data_prompt = {
                    "question": k,
                    "content": v
        }
        if(data_prompt in retrieve_data): continue
        retrieve_data.append(data_prompt)
    _get_embedding_and_save_to_chroma(retrieve_data, collection, encoder)
    return collection

#获取collection
def get_collection(retriever: str, name: str = None ,chroma_dir: str = None):

    encoder = Encoder(retriever)
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    Collection =  chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    return Collection

#获取qas_collection并存储prompt
def get_qas_collection_and_write(retriever: str , qa_data: list = None, name: str = None ,chroma_dir: str = None, chunk_id = None):
#1 起服务
    encoder = Encoder(retriever)
    if name == None:
        name = "qas"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []

    for idx , elements in enumerate(qa_data):
        n,q,ans = elements
        print(n,q,ans)
        data_prompt = {
                    "name": n,
                    "question": q,
                    "answer": ans,
                    "chunk_id":chunk_id
        }
        if(data_prompt in retrieve_data): continue
        retrieve_data.append(data_prompt)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["question"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

#获取qas_collection并查询top K
def get_qas_collection_and_query(retriever: str , name: str = None ,chroma_dir: str = None ,query_texts: str = None ,recall_num : int = None):
    encoder = Encoder(retriever)
    if name == None:
        name = "qas"
    
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_qas = collection.query(query_texts=query_texts , n_results = recall_num)
    return results_qas

#数据库记录数量
def chroma_count(retriever: str , name: str = None ,chroma_dir: str = None):
    encoder = Encoder(retriever)
    if name == None:
        name = "main"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    num = collection.count()
    return num

#重置数据库
def rebuild_collection(retriever: str , name: str = None ,chroma_dir: str = None):
    
    encoder = Encoder(retriever)
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    chroma_client.delete_collection(name)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    return collection

#获取context_collection并查询top K
def get_context_collection_and_query(retriever: str, name: str = None, chroma_dir: str = None, query_texts: str = None, recall_num : int = None):
    
    encoder = Encoder(retriever)
    if name == None:
        name = "context"
    
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_context = collection.query(query_texts = query_texts, n_results = recall_num)
    return results_context

#获取summary_collection并查询top K
def get_summary_collection_and_query(retriever: str, name: str = None, chroma_dir: str = None, query_texts: str = None, recall_num : int = None):
    
    encoder = Encoder(retriever)
    if name == None:
        name = "summary"
    
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_summary = collection.query(query_texts = query_texts, n_results = recall_num)
    return results_summary

#获取qa_collection并存储prompt
def get_context_collection_and_write(retriever: str, context_data: list = None, name: str = None, chroma_dir: str = None, chunk_id = None):
#1 起服务
    encoder = Encoder(retriever)
    if name == None:
        name = "context"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []

    for i in range(len(context_data)):
        data = {
                    "context": context_data[i],
                    "chunk_id": chunk_id
        }
        if(data in retrieve_data): continue
        retrieve_data.append(data)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["context"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

#获取qa_collection并存储prompt
def get_summary_collection_and_write(retriever: str, summary_data, name: str = None, chroma_dir: str = None, chunk_id = None):
#1 起服务
    encoder = Encoder(retriever)
    if name == None:
        name = "summary"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []
    data_prompt = {
        "summary": summary_data,
        "chunk_id": chunk_id
    }
    retrieve_data.append(data_prompt)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["summary"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None


