from sentence_transformers import SentenceTransformer,util
model="stella_en_400M_v5"
embedding_model = SentenceTransformer(
        model_name_or_path=model,
        trust_remote_code=True,
    )
text1 =["Hello World!","heLLO"]
result1 = embedding_model.encode(text1)
print(type(result1))
print(result1)
flat_list = result1.tolist()
print(len(flat_list))
print(type(flat_list))
print(flat_list)
list =[]
text2 ="How are you?What is your favorite"
result2 = embedding_model.encode(text2)
