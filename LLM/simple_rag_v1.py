from openai import OpenAI

from openai import OpenAI
from pathlib import Path

client = OpenAI(
    api_key="sk-65d66b02152e4dffa55089488f63acaa",
    base_url="https://api.deepseek.com")



# 步骤1:实现一个简单的“关键词匹配”检索器
def retrieval(query):
    context = ""

    # 1. 遍历所有文件
    path_list = list(Path("konwledge").glob("*.txt"))

    # 2. 找到所有的文件
    for path in path_list:
        if path.stem in query:
            # 如果和query相关
            # 3. 相关文件的内容读取出来，添加到 context 中
            context += path.read_text(encoding="utf-8")
            context += "\n\n\n"

    return context

print(retrieval(input("query: ")))

# 步骤2:增强Query



# 步骤3:生成回答
def generation(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content