from openai import OpenAI

from _04_file_db import model, collection  # 可以从之前的文件中导入

api_key = "sk-3990a9f3869a4007ba32ce390e34e8f0"

# 默认使用的是 openai   如果要使用deepseek需要改动参数
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key
)


# 步骤1 : 语义检索
def retrieval(query):
    context = ""
    query = [query]
    query_embedding = model.encode(query)

    # 从向量数据库中获取相似度较高的几条数据
    data = collection.query(query_embedding.tolist(), n_results=5)
    # print(data)

    # 这才是实际需要的内容
    text_list = data['documents'][0]
    # print(text_list)

    for t in text_list:
        context += t
        context += "\r\n-------------------------\r\n"

    return context


# 步骤2：增强Query         prompt提示词增强
def augmented(query, context=""):
    if not context:
        return f"请简要回答以下问题：{query}"
    else:
        # prompt = f"""你是一个严谨的RAG助手。
        #         请根据以下提供的上下文信息来回答问题。
        #         如果上下文信息不足以回答问题，请直接说“根据提供的信息无法回答”
        #         如果回答使用了上下文中的信息，在回答后输出使用了哪些上下文。
        #         上下文信息：{context}
        #         ---------------------
        #         问题：{query}
        #         """


        prompt= f"""
            请严格扮演一个信息提取助手的角色。你的任务是根据提供的【参考文档】来回答问题。

            【参考文档开始】
            {context}
            【参考文档结束】
            
            规则：
            1.  你的回答必须完全基于上述参考文档。如果答案未在文档中明确提及，你必须说“文档中未提及相关信息”。
            2.  不要引入文档以外的知识或假设。
            3.  如果文档中的信息相互矛盾，请指出这一点。
            4.  在回答的最后，用括号注明答案所依据的文档句子编号（例如：基于[1][3]）。
            
            问题：{query}
                   
        """

        # 正常模型对上下文的长度是有限制的，可以使用分词器来判断长度是否满足    （deepseek的限制是64k）
        return prompt


# 步骤3：生成回答
def generation(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",  # 选择模型
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    # print(generation(augmented("无人机简称",retrieval("无人机简称"))))
    # print(generation(augmented("飞机是什么",retrieval("飞机是什么"))))

    query = "请帮我介绍埃及"
    query = "能飞多久"

    context = retrieval(query)
    prompt = augmented(query, context)
    print(generation(prompt))
