from langchain_core.prompts import PromptTemplate

format_prompt = PromptTemplate.from_template("请告诉我一个关于{object}的笑话")

prompt_a = format_prompt.format(object="程序员")
prompt_b = format_prompt.format(object="汽车")
prompt_c = format_prompt.format(object="印度")

print(prompt_a)
print(prompt_b)
print(prompt_c)