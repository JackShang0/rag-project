from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
import chromadb

# 添加基础信息打印
print("Python脚本启动...")
print(f"当前工作目录: {Path.cwd()}")

try:
    model = SentenceTransformer(r"E:\shangdj\python\rag\model_dir\BAAI\bge-large-zh-v1___5"
                                , device='cpu' ) # 强制使用 CPU
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

try:
    client = chromadb.PersistentClient("./chroma_v1")
    print("向量数据库客户端创建成功")
except Exception as e:
    print(f"向量数据库客户端创建失败: {e}")
    exit(1)

try:
    collection = client.get_or_create_collection(
        name="shangdj",
        metadata={
            "介绍": "文本文件的向量数据库",
            "创建时间": str(datetime.now()),
            "hnsw:space": "cosine"
        }
    )
    print("集合获取/创建成功")
except Exception as e:
    print(f"集合创建失败: {e}")
    exit(1)


def txt_2db():
    print("\n=== 开始处理文本数据 ===")

    # 1. 检查 knowledge 目录
    knowledge_path = Path(r"E:\shangdj\python\rag\rag-project\LLM\knowledge")
    print(f"检查目录: {knowledge_path.absolute()}")

    if not knowledge_path.exists():
        print(f"错误: {knowledge_path} 目录不存在")
        print("请确保在当前目录下创建 knowledge 文件夹")
        return

    if not knowledge_path.is_dir():
        print(f"错误: {knowledge_path} 不是目录")
        return

    # 2. 查找文本文件
    path_list = list(knowledge_path.glob("*.txt"))
    print(f"找到 {len(path_list)} 个 .txt 文件")

    if len(path_list) == 0:
        print("错误: knowledge 目录中没有找到 .txt 文件")
        print("请确保 knowledge 目录中有文本文件")
        return

    # 3. 读取文件内容
    text_list = []
    for i, path in enumerate(path_list):
        print(f"  [{i + 1}/{len(path_list)}] 读取文件: {path.name}")
        try:
            text = path.read_text(encoding="utf-8")
            text_list.append(text)
            print(f"     读取成功，字符数: {len(text)}")
        except Exception as e:
            print(f"     读取失败: {e}")
            continue

    if len(text_list) == 0:
        print("错误: 所有文件读取失败")
        return

    print(f"成功读取 {len(text_list)} 个文件的内容")

    # 4. 进行向量嵌入
    print("开始向量嵌入...")
    try:
        embeddings = model.encode(text_list)
        print(f"向量嵌入完成，嵌入维度: {embeddings.shape}")
    except Exception as e:
        print(f"向量嵌入失败: {e}")
        return

    # 5. 存入向量数据库
    print("存入向量数据库...")
    try:
        # 生成 ID 列表
        ids = [f"doc_{i}" for i in range(len(text_list))]

        collection.add(
            embeddings=embeddings.tolist(),
            documents=text_list,
            metadatas=[{"id": i, "filename": path_list[i].name} for i in range(len(text_list))],
            ids=ids
        )

        print(f"数据库中的数据量: {collection.count()}")
        print("数据处理完成！")

    except Exception as e:
        print(f"数据存入失败: {e}")


if __name__ == '__main__':
    print("\n=== 主程序开始 ===")
    txt_2db()
    print("=== 主程序结束 ===")