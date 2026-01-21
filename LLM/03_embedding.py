from sentence_transformers import SentenceTransformer


# 1. 加载一个预训练的Embedding模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 准备句子      将需要输入的内容以列表的形式存放
sentences = ["The weather is lovely today.","It's so sunny outside!","he drove to the stadium"]

# 2. 调用模型计算嵌入向量
embeddings = model.encode(sentences)
print(embeddings.shape)     # 句子数量和维度
# 输出结果。  (2, 384)   2:句子的数量   384:每个句子的向量维度


# 向量空间中的向量值。 脱离模型无实际含义
# for i in embeddings:
#     print(i)        # 向量内容
#     print("-----")

# 相似度。
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# 输出结果如下，可以理解为是一个表格，其中横轴分别表示 A、B、C  纵轴也分别表示A、B、C  相交的点就是对应的横轴与纵轴的相似度
# tensor([[1.0000, 0.6660, 0.1058],
#         [0.6660, 1.0000, 0.1471],
#         [0.1058, 0.1471, 1.0000]])


#  上述情况输出的矩阵，是因为输入的是embeddings，embeddings，是拿embeddings自己和自己比较了。
#  相似度的另外一种情况，可以拿一个句子去和embeddings去比较，例如：
new_embeddings = model.encode("sun")            # 用户问题
similarities = model.similarity(new_embeddings, embeddings)     # 知识库
print(similarities)
# 这种情况下，输出的就是。tensor([[0.4095, 0.6020, 0.1625]])
# 这样的话，我们就可以把1和2两个相关性较高的句子拿出来，第3个相关性较低，就不拿出来了

#==========重要结论如下=============
#-----正常情况下，我们就可以把问题和知识库的内容传入，做相似度的比较，得出相似的向量-------
