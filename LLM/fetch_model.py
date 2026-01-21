from modelscope import snapshot_download

# 用于下载模型的代码
model_dir = snapshot_download(
    model_id='BAAI/bge-large-zh-v1.5',  # 约1.2GB   模型名称
    cache_dir=r'E:\shangdj\python\rag\model_dir'  # 模型下载的地址
)
