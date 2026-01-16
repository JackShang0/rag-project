from openai import OpenAI

from pathlib import Path

api_key = "sk-3990a9f3869a4007ba32ce390e34e8f0"

# 默认使用的是 openai   如果要使用deepseek需要改动参数
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key
)


# 步骤1：实现一个简单的“关键词匹配”检索器
def retrieval(query):
    context = ""

    # 1. 遍历所有文件
    path_list = list(Path("knowledge").glob("*.txt"))

    # 2. 找到和问题相关的文件
    for path in path_list:
        if path.stem in query:
            # 如果文件名和查询条件相关  就把对应文件的内容读取出来 添加到context上下文中
            context += path.read_text(encoding="UTF-8")

            # 如果多个文件，则换行
            context += "\n\n"

    return context


# 步骤2：增强Query         prompt提示词增强
def augmented(query,context=""):
    if not context:
        return f"请简要回答以下问题：{query}"
    else:
        prompt = f"""请根据上下文来回答问题，如果上下文不足以回答问题，请直接说：“根据上下文信息，无法回答问题”
        上下文：{context}
        问题：{query}
"""

        # 正常模型对上下文的长度是有限制的，可以使用分词器来判断长度是否满足    （deepseek的限制是64k）
        return prompt



# 步骤3：生成回答
def generation(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",      # 选择模型
        messages=[
            {"role": "user", "content": prompt},
      ]
    )

    return response.choices[0].message.content





print("=======知识库检索======")
# print(retrieval("无人机"))


print("=======prompt的增强======")
# print(augmented("无人机简称",retrieval("无人机简称")))



if __name__ == '__main__':
    # print(generation(augmented("无人机简称",retrieval("无人机简称"))))
    # print(generation(augmented("飞机是什么",retrieval("飞机是什么"))))

    query = "请帮我介绍埃及"
    query = "能飞多久"


    # 不使用RAG
    print(generation(query))


    print("\n\n\n\n\n")

    # 使用RAG
    context = retrieval(query)
    prompt = augmented(query,context)
    print(generation(prompt))
