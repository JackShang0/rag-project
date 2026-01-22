from operator import itemgetter
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, \
    CharacterTextSplitter  # 推荐使用RecursiveCharacterTextSplitter 递归分割：按优先级尝试多个分隔符，递归进行。
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

api_key = "sk-3990a9f3869a4007ba32ce390e34e8f0"
# 1. 设置模型
# 1.1 大语言模型
llm = ChatOpenAI(model="deepseek-chat",
                 base_url="https://api.deepseek.com",
                 api_key=api_key)  # 此处为什么不需要写apikey和api_base_url     此处都放到了环境变量中
# 1.2 向量嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name=r"E:\shangdj\python\rag\model_dir\BAAI\bge-large-zh-v1___5")

# 2. 设置数据处理（加载、分块、存储、检索）
file_dir = Path("knowledge")  # 设置文件目录
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # 滑动窗口分块  为防止切断，允许100字的重叠，
vector_store = Chroma(embedding_function=embedding_model, persist_directory="./chroma_v2")  # 向量数据库的准备
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # 检索  使用默认相似匹配方式即可，匹配数量为5个

# 提示词
format_prompt = PromptTemplate.from_template("""
            请严格扮演一个信息提取助手的角色。你的任务是根据提供的【参考文档】来回答问题。

            【参考文档开始】
            {context}
            【参考文档结束】
            
            规则：
            1.  你的回答必须完全基于上述参考文档。如果答案未在文档中明确提及，你必须说“文档中未提及相关信息”。
            2.  不要引入文档以外的知识或假设。
            3.  如果文档中的信息相互矛盾，请指出这一点。
            4.  在回答的最后，用括号注明答案所依据的文档句子编号（例如：基于[1][3]）。
            
            问题：{question}
""")

# 3. 编排“链”  按照任务的调用去编排任务  检索 -> 将上下文向量、问题，提供给提示词 -> 在将提示词输入到大模型 -> 大模型返回结果
chain = ({"question": RunnablePassthrough()}  # 拿到检索向量时候的问题
         | RunnablePassthrough.assign(context=itemgetter("question") | retriever)  # 根据问题去匹配向量库的内容，找到相关的向量retriever
         | format_prompt  # 把问题和上下文整合成prompt
         | llm  # 交给大模型处理
         | StrOutputParser()  # 把结果转换成最终简洁答复
         )



def init_doc():
    # 4. 初始化知识库，即文档的加载和切分   注意：只需执行一次即可
    docs = DirectoryLoader(str(file_dir), loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"}).load()  # 加载文档
    docs = text_splitter.split_documents(docs)  # 切分文档
    vector_store.add_documents(docs)  # 存储文档


if __name__ == '__main__':
    init_doc()

    query = "能飞多久"
    print(chain.invoke(query))