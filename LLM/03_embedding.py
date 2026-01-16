from sentence_transformers import SentenceTransformer


# 1. 加载一个预训练的Embedding模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 准备句子
sentences = ["The weather is lovely today.","It's so sunny outside!"]

# 2. 调用模型计算嵌入向量
embeddings = model.encode(sentences)
print(embeddings.shape)     # 句子数量和维度