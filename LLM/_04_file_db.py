from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
import chromadb
import torch

# 在代码开头添加
torch.cuda.empty_cache()  # 清空 GPU 缓存

model = SentenceTransformer(r"E:\shangdj\python\rag\model_dir\BAAI\bge-large-zh-v1___5"
                            , device='cpu')

# 实例化向量数据库
# client = chromadb.Client()  # 这种方式是放到内存中的
client = chromadb.PersistentClient("./chroma_v1")  # 这种就是文件数据库，会保存在本地文件中，不会受断电影响
collection = client.get_or_create_collection(
    name="shangdj",
    metadata={
        "介绍": "文本文件的向量数据库",
        "创建时间": str(datetime.now()),
        # 其中带：的都是比较特殊的
        "hnsw:space": "cosine"  # 使用那种相似度搜索方法    cosine：余弦相似度
    }
)


def txt_2db():
    # 1 加载所有的文件
    path_list = list(Path(r"/LLM/knowledge").glob("*.txt"))
    text_list = []  # 文本内容

    for path in path_list:
        text = path.read_text(encoding="utf-8")
        text_list.append(text)

    # 2 进行向量嵌入
    embeddings = model.encode(text_list)

    # 3 存入向量数据库
    collection.add(
        embeddings=embeddings.tolist(),  # 将所有向量放进去
        documents=text_list,  # 文本     根据向量找到文本，把文本给到大模型
        metadatas=[{"id": i} for i, _ in enumerate(text_list)],  # 元数据
        ids=[f"doc_{i}" for i, _ in enumerate(text_list)],  # id
    )
    print(f'数据库中的数据量：{collection.count()}')


if __name__ == '__main__':
    # try:
    #     txt_2db()
    # except Exception as e:
    #     print(f"发生异常：{e}")
    #     import traceback
    #     traceback.print_exc()

    query = ["能飞多久"]
    query = ["现在要做什么"]
    query_embedding = model.encode(query)

    data = collection.query(query_embedding.tolist(), n_results=5)
    print(data)

    # 这才是实际需要的内容
    text_list = data['documents'][0]
    print(text_list)

    for t in text_list:
        print(t)

    # 输出的内容格式如下：其中在chromadb中，使用余选计算相似度，distances字段的值=1-相似度

    # {'ids': [['doc_0', 'doc_1']],
    #  'embeddings': None,
    #  'documents': [
    #      ['埃及旅游以其古老的金字塔、神庙、尼罗河风光和红海度假胜地闻名，必游景点包括吉萨金字塔、帝王谷、卡纳克神庙、大埃及博物馆（GEM）、阿布辛贝神殿，并体验尼罗河游船和红海潜水，最佳季节是10月至次年5月，避开酷热的夏季，中国游客需提前办理签证或可办理落地签。\n核心体验\n古迹奇观: 吉萨金字塔群、狮身人面像、帝王谷（法老墓）、卡納克神廟、阿布辛貝神殿。\n尼羅河: 乘坐遊輪，欣賞兩岸風光，參觀沿途神廟（艾德夫神廟、康翁波神廟）。\n博物館: 開羅的大埃及博物館（GEM）和埃及博物館，收藏豐富。\n紅海度假: 赫爾格達（Hurghada）或沙姆沙伊赫（Sharm El Sheikh）是潛水和海灘勝地。\n文化市集: 開羅的哈利利大市集，體驗本地生活。\n最佳季节\n10月至5月: 天氣最宜人，氣溫舒適，是旅遊旺季。\n避開6月至9月: 夏季酷熱，溫度常超過40°C。\n行程建议\n經典路線: 開羅 (金字塔) → 飛機/夜臥火車 → 盧克索 (帝王谷、神廟) → 亞斯旺 (尼羅河段) → 阿布辛貝 → 尼羅河遊船 → 紅海度假（可選）。\n交通: 城市間可搭乘國內航班或夜臥火車；尼羅河段多利用遊輪；開羅可利用出租車或網約車。\n签证与准备\n签证: 持中國普通護照需提前辦理埃及簽證，也可在機場申請落地簽（需備好往返機票、酒店訂單、2000美元現金等）。\n必備: 防曬霜、太陽眼鏡、帽子、舒適的鞋子，以及應對日夜溫差的衣物。\n貨幣: 阿拉伯埃及鎊（EGP），美元/歐元在主要旅遊區也通用。',
    #       '无人机（UAV/Drone）是靠自身动力飞行、可遥控或自主飞行的飞行器，分为军用、工业级和消费级，参数包括飞行性能（续航、速度、高度）、相机规格（像素、光圈、焦距）、图传距离、避障系统和机身重量等。常见消费级无人机如大疆 Mavic 系列提供高画质航拍，具备全向避障和长距离图传，满足娱乐、摄影等需求。\n无人机介绍\n定义：无需驾驶员在座舱内操控的飞行器，通过遥控或预设程序自主飞行。\n分类：\n按用途：军事（侦查、攻击）、工业（测绘、巡检）、消费（航拍、娱乐）。\n按结构：\n旋翼式：多轴（四轴、六轴），垂直起降，悬停稳定，适合复杂环境。\n固定翼：类似飞机，续航长，适合大面积、长距离任务。\n混合动力：结合两者优点，可垂直起降和高速巡航。\n主要组成：机身、动力系统（电机、螺旋桨）、飞控系统、传感器（GPS、IMU）、电池、相机、图传系统。\n核心参数详解\n飞行时间/续航 (Flight Time)：单块电池最长飞行时长，通常30-46分钟（大疆机型）。\n最大飞行速度 (Max Speed)：不同模式下（如运动模式）的最高速度，以米/秒 (m/s) 或公里/小时 (km/h) 为单位。\n图传距离 (Transmission Distance)：遥控器与无人机之间的最远通信距离，影响操控范围。\n相机参数 (Camera Specs)：\n传感器：CMOS尺寸 (如1/1.3英寸, 4/3英寸)。\n像素：照片或视频的有效像素数。\n视频分辨率：如5.1K/50fps, 4K/30fps。\n视场角 (FOV)：广角相机通常84°，长焦镜头视角更小。\n光圈 (f/1.8, f/2.8)。\n避障系统 (Obstacle Sensing)：多向视觉或红外传感器，感知并避免碰撞。\n云台 (Gimbal)：三轴机械云台，实现画面稳定。\n起飞重量 (Takeoff Weight)：影响便携性和法规要求 (如<250g可免于注册)。\n消费级无人机代表参数示例 (大疆 Mavic 3)\n续航：46分钟。\n相机：主摄4/3英寸CMOS，长焦1/2英寸。\n图传：OcuSync 3.0+，15公里。\n避障：全向双目视觉系统。\n重量：895/899克。']],
    #  'uris': None,
    #  'included': ['metadatas', 'documents', 'distances'],
    #  'data': None,
    #  'metadatas': [[{'id': 0}, {'id': 1}]],
    #  'distances': [[0.6723231673240662, 0.6820040941238403]]}

# -----------------------------------------------------查看向量数据库信息-----------------------------------------
    # 查看数据库基本信息
    print(f"数据库名称: {collection.name}")
    print(f"文档数量: {collection.count()}")
    # 查看数据库中的所有数据
    all_data = collection.get()
    print("\n=== 数据库中的所有数据 ===")

    # 按格式打印每个文档
    for i, (doc_id, document, metadata) in enumerate(zip(
            all_data['ids'],
            all_data['documents'],
            all_data['metadatas']
    )):
        print(f"\n文档 {i + 1} (ID: {doc_id}):")
        print(f"元数据: {metadata}")
        print(f"内容摘要: {document[:200]}...")  # 只显示前200个字符
        print("-" * 50)