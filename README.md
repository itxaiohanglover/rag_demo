## RAG
### 介绍
- 背景：在实际业务场景中，通用的基础大模型可能存在无法满足我们需求的情况。
	- 知识局限性：大模型的知识来源于训练数据，而这些数据主要来自于互联网上已经公开的资源，对于一些实时性的或者非公开的，由于大模型没有获取到相关数据，这部分知识也就无法被掌握。​
	- 数据安全性：为了使得大模型能够具备相应的知识，就需要将数据纳入到训练集进行训练。然而，对于企业来说，数据的安全性至关重要，任何形式的数据泄露都可能对企业构成致命的威胁。
	- 大模型幻觉：由于大模型是基于概率统计进行构建的，其输出本质上是一系列数值运算。因此，有时会出现模型“一本正经地胡说八道”的情况，尤其是在大模型不具备的知识或不擅长的场景中。 
- 介绍：RAG 是`检索增强生成`（Retrieval Augmented Generation）的简称，它为大语言模型 (LLMs) 提供了从数据源检索信息的能力，并以此为基础生成回答。
	- 步骤1：问题理解，准确把握用户的意图。
	- 步骤2：`知识检索`，从知识库中相关的知识检索。【难点，`用户提问可能以多种方式表达，而知识库的信息来源可能是多样的，包括PDF、PPT、Neo4j等格式。`】
	- 步骤3：答案生成，将检索结果与问题。
- 优点：
	- 提高准确性和相关性。
	- 改善时效性，使模型适应当前事件和知识。
	- 降低生成错误风险，依赖检索系统提供的准确信息。

以下是RAG输出到大型语言模型的典型模板：
```
你是一个{task}方面的专家，请结合给定的资料，并回答最终的问题。请如实回答，如果问题在资料中找不到答案，请回答不知道。

问题：{question}

资料：
- {information1}
- {information2}
- {information3}
```
其中，{task}代表任务的领域或主题，{question}是最终要回答的问题，而{information1}、{information2}等则是提供给模型的外部知识库中的具体信息。

### 分类
参考论文：Retrieval-Augmented Generation for Large Language Models: A Survey

RAG可以根据技术复杂度，分为三种：
- Naive RAG：Naive RAG是RAG技术的最基本形式，也被称为经典RAG。包括`索引、检索、生成三个基本步骤`。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aee4bad0d42f4fa581ec5610355bef0e.png#pic_center)
	- 索引（Indexing）：将文档库分割成较短的 Chunk，即文本块或文档片段，然后构建成向量索引。【`离线计算，存储到向量数据库中，例如Milvus。`】
	- 检索（Retrieval）：计算问题和 Chunks 的相似度，检索出若干个相关的 Chunk。【采用`在线计算`，但是随着`知识库的增大`，导致`检索速度变慢、检索效果出现退化`。】![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97bb35c928614f15a4c1e8495a188d43.png#pic_center)
		- 解决检索速度变慢。
			- 当数据库非常大的时候，`向量相似度计算会非常耗时间`，可以在检索之前进行 `召回`（Recall），即从 数据库 中快速获得大量大概率相关的 Chunk，然后只有这些 Chunk 会参与计算向量相似度。
			- 召回 步骤`不要求非常高的准确性`，因此通常`采用简单的基于字符串的匹配算法`。由于这些算法`不需要任何模型，速度会非常快`，常用的算法有 `TF-IDF，BM25` 等。
			- 另外，也有很多工作致力于实现更快的 向量检索 ，例如 `faiss`，`annoy`。
		- 解决检索效果出现退化。
			- 增加一个`二阶段检索——重排 (Rerank)`，即利用 `重排模型（Reranker）`，使得越相似的结果排名更靠前。这样就能实现`准确率稳定增长，即数据越多，效果越好`（如上图中`紫线`所示）。​
			- 通常，为了与 重排 进行区分，`一阶段检索有时也被称为 精排` 。而在一些更复杂的系统中，`在 召回 和 精排 之间还会添加一个 粗排 步骤`。
			- 在整个 检索 过程中，`计算量的顺序是 召回 > 精排 > 重排`，而`检索效果的顺序则是 召回 < 精排 < 重排` 。
			- 检索过程完成后，从得到排好序的一系列 检索文档（Retrieval Documents）`挑选最相似的 k 个结果`，将它们`和用户查询拼接成prompt的形式，输入到大模型`。
	- 生成（Generation）：将检索到的Chunks作为背景信息，生成问题的回答。​
- Advanced RAG：Advanced RAG在Naive RAG的基础上进行优化和增强。`包含额外处理步骤，分别在数据索引、检索前和检索后进行`。包括更精细的数据清洗、设计文档结构和添加元数据，以提升文本一致性、准确性和检索效率。`在检索前使用问题的重写、路由和扩充等方式对齐问题和文档块之间的语义差异`。`在检索后通过重排序避免“Lost in the Middle”现象，或通过上下文筛选与压缩缩短窗口长度`。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70825efdc4254217bc6c50aa472e3110.png#pic_center)
- Modular RAG：Modular RAG`引入更多具体功能模块`，例如查询搜索引擎、融合多个回答等。技术上融合了检索与微调、强化学习等。流程上对RAG模块进行设计和编排，出现多种不同RAG模式。提供更大灵活性，系统可以根据应用需求选择合适的功能模块组合。`模块化RAG`的引入使得系统更自由、灵活，适应不同场景和需求。

**RAG和SFT对比：**
| 特性       | RAG技术                                    | SFT模型微调                              |
| :--------- | :----------------------------------------- | :--------------------------------------- |
| 知识更新   | 实时更新检索库，适合动态数据，无需频繁重训 | 存储静态信息，更新知识需要重新训练       |
| 外部知识   | 高效利用外部资源，适合各类数据库           | 可对齐外部知识，但对动态数据源不够灵活   |
| 数据处理   | 数据处理需求低                             | 需构建高质量数据集，数据限制可能影响性能 |
| 模型定制化 | 专注于信息检索和整合，定制化程度低         | 可定制行为，风格及领域知识               |
| 可解释性   | 答案可追溯，解释性高                       | 解释性相对低                             |
| 计算资源   | 需要支持检索的计算资源，维护外部数据源     | 需要训练数据集和微调资源                 |
| 延迟要求   | 数据检索可能增加延迟                       | 微调后的模型反应更快                     |
| 减少幻觉   | 基于实际数据，幻觉减少                     | 通过特定域训练可减少幻觉，但仍然有限     |
| 道德和隐私 | 处理外部文本数据时需要考虑隐私和道德问题   | 训练数据的敏感内容可能引发隐私问题       |

**开源RAG框架**
- TinyRAG：DataWhale成员宋志学精心打造的纯手工搭建RAG框架。​
- LlamaIndex：一个用于构建大语言模型应用程序的数据框架，包括数据摄取、数据索引和查询引擎等功能。​
- LangChain：一个专为开发大语言模型应用程序而设计的框架，提供了构建所需的模块和工具。​
- QAnything：网易有道开发的本地知识库问答系统，支持任意格式文件或数据库。​
- RAGFlow：InfiniFlow开发的基于深度文档理解的RAG引擎。​
- ···
### 实战 - 搭建 RAG Demo
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/139c75addae84c58a82ac54357688f71.png#pic_center)
#### 预备
1.开通免费试用阿里云PAI—DSW

链接：https://free.aliyun.com/?searchKey=PAI

>开通PAI-DSW 试用 ，可获得 5000算力时！有效期3个月！

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b18266b7dc54c6ab4ed879ecc4fb1f4.png#pic_center)
2.在魔搭社区进行授权

链接：https://www.modelscope.cn/my/mynotebook/authorization

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/924f90fae1584b32a930a2ce862a4b57.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/768f022dd7e147c08a3e932e248f2e4c.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a406e6947b14301924a0338bf62d11a.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9d1a1a6f84e431696063ea3353a0a01.png#pic_center)
创建实例：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/92178bb4b9d74546ab5453d19ced91cd.png#pic_center)
这里一定要选择支持资源包抵扣的服务器
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c4402978d96145829887be40a46747ef.png#pic_cecnter)
创建实例后打开，能看到这个界面就成功啦！

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aa4b4128f80e4d48b9e7dc982b0260fe.png#pic_center)
#### 索引
新建`bge-small-zh-v1.5-download.py`，用于`向量模型下载`：
```python
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
```
```
/usr/local/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Downloading: 100%|██████████| 190/190 [00:00<00:00, 328B/s]
Downloading: 100%|██████████| 776/776 [00:00<00:00, 1.48kB/s]
Downloading: 100%|██████████| 124/124 [00:00<00:00, 266B/s]
Downloading: 100%|██████████| 47.0/47.0 [00:00<00:00, 92.2B/s]
Downloading: 100%|██████████| 91.4M/91.4M [00:00<00:00, 120MB/s] 
Downloading: 100%|██████████| 349/349 [00:00<00:00, 715B/s]
Downloading: 100%|██████████| 91.4M/91.4M [00:00<00:00, 123MB/s] 
Downloading: 100%|██████████| 27.5k/27.5k [00:00<00:00, 54.9kB/s]
Downloading: 100%|██████████| 52.0/52.0 [00:00<00:00, 65.0B/s]
Downloading: 100%|██████████| 125/125 [00:00<00:00, 253B/s]
Downloading: 100%|██████████| 429k/429k [00:00<00:00, 703kB/s]
Downloading: 100%|██████████| 367/367 [00:00<00:00, 759B/s]
Downloading: 100%|██████████| 107k/107k [00:00<00:00, 220kB/s]
```
封装向量模型类 EmbeddingModel：
```python
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

# 定义向量模型类
class EmbeddingModel:
    """
    用于加载预训练的模型并计算文本的嵌入向量的类。
    """

    def __init__(self, path: str) -> None:
        """
        初始化方法，加载预训练的分词器和模型。

        :param path: 预训练模型的路径。
        """
        # 加载预训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # 加载预训练的模型，并将其移动到GPU上
        self.model = AutoModel.from_pretrained(path).cuda()
        print(f'Loading EmbeddingModel from {path}.')

	# 为了充分发挥GPU矩阵计算的优势，输入和输出都是一个 List，即多条文本和他们的向量表示。
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        计算文本列表的嵌入向量。

        :param texts: 要计算嵌入向量的文本列表。
        :return: 嵌入向量的列表。
        """
        # 使用分词器对文本进行编码
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # 将编码后的输入数据移动到GPU上
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}

        # 不计算梯度，以提高计算效率
        with torch.no_grad():
            # 将编码后的输入传递给模型，获取模型输出
            model_output = self.model(**encoded_input)
            
            # 获取句子的嵌入向量，这里假设模型输出的第一个元素包含嵌入信息
            sentence_embeddings = model_output[0][:, 0]

        # 归一化嵌入向量
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # 将嵌入向量转换为列表格式并返回
        return sentence_embeddings.tolist()
```
测试：
```python
print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
embed_model = EmbeddingModel(embed_model_path)

"""
> Create embedding model...
Loading EmbeddingModel from ./AI-ModelScope/bge-small-zh-v1___5.
"""
```
#### 检索
编写一个知识库文档`knowledge.txt`：
```txt
广州大学（Guangzhou University），简称广大（GU），是由广东省广州市人民政府举办的全日制普通高等学校，实行省市共建、以市为主的办学体制，是国家“111计划”建设高校、广东省和广州市高水平大学重点建设高校。广州大学的办学历史可以追溯到1927年创办的私立广州大学；1951年并入华南联合大学；1983年筹备复办，1984年定名为广州大学；2000年7月，经教育部批准，与广州教育学院（1953年创办）、广州师范学院（1958年创办）、华南建设学院西院（1984年创办）、广州高等师范专科学校（1985年创办）合并组建成立新的广州大学。
郑州机械研究所有限公司（以下简称郑机所）的前身机械科学研究院1956年始建于北京，是原机械工业部直属一类综合研究院所，现隶属于国资委中国机械科学研究总院集团有限公司。郑机所伴随着共和国的成长一路走来，应运而生于首都，碧玉年华献中原。多次搬迁，驻地从北京经漯河再到郑州；数易其名，由机械科学研究院到漯河机械研究所再到郑州机械研究所，现为郑州机械研究所有限公司。1956～1958年应运而生：依据全国人大一届二次会议的提议和第一机械工业部的决策，1956年3月6日，第一机械工业部发文《（56）机技研究第66号》，通知“机械科学实验研究院”（后改名为“机械科学研究院”）在北京成立。1959～1968年首次创业：承担国家重大科研项目与开发任务，以及行业发展规划以及标准制定等工作，如“九大设备”的若干关键技术等。1969～1972年搬迁河南：1969年按照“战备疏散”的要求，机械科学研究院主体迁建河南漯河，成立“漯河机械研究所”；1972年因发展需要，改迁河南郑州，成立郑州机械研究所。1973～1998年二次创业：先后隶属于国家机械工业委员会、机械电子工业部、机械工业部；1981年4月罗干由铸造室主任升任副所长，同年经国务院批准具备硕士学位授予权；1985年“葛洲坝二、三江工程及其水电机组项目”荣获国家科技进步特等奖。1999～2016年发展壮大：1999年转企改制，隶属于国资委中国机械科学研究总院；2008年被河南省首批认定为“高新技术企业”；2011年获批组建新型钎焊材料与技术国家重点实验室；2014年被工信部认定为“国家技术创新示范企业”；历经十多年开发出填补国内外空白的大型齿轮齿条试验装备，完成了对三峡升船机齿条42.2万次应力循环次数的疲劳寿命试验测试；营业收入从几千万发展到近6亿；2017年至今协同发展：2017年经公司制改制，更名为郑州机械研究所有限公司，一以贯之地坚持党对国有企业的领导，充分发挥党委把方向、管大局、保落实的领导作用；一以贯之地建立现代企业制度，持续推进改革改制，努力实现以高质量党建引领郑机所高质量发展。 
非洲野犬，属于食肉目犬科非洲野犬属哺乳动物。 又称四趾猎狗或非洲猎犬； 其腿长身短、体形细长；身上有鲜艳的黑棕色、黄色和白色斑块；吻通常黑色，头部中间有一黑带，颈背有一块浅黄色斑；尾基呈浅黄色，中段呈黑色，末端为白色，因此又有“杂色狼”之称。 非洲野犬分布于非洲东部、中部、南部和西南部一带。 栖息于开阔的热带疏林草原或稠密的森林附近，有时也到高山地区活动。其结群生活，没有固定的地盘，一般在一个较大的范围内逗留时间较长。非洲野犬性情凶猛，以各种羚羊、斑马、啮齿类等为食。奔跑速度仅次于猎； 雌犬妊娠期为69-73天，一窝十只仔，哺乳期持续6-12个星期。 其寿命11年。 非洲野犬正处在灭绝边缘，自然界中仅存两三千只。 非洲野犬被列入《世界自然保护联盟濒危物种红色名录》中，为濒危（EN）保护等级。 ",非洲野犬共有42颗牙齿（具体分布为：i=3/3；c=1/1；p=4/4；m=2/3x2），前臼齿比相对比其他犬科动物要大，因此可以磨碎大量的骨头，这一点很像鬣狗。 主要生活在非洲的干燥草原和半荒漠地带，活跃于草原、稀树草原和开阔的干燥灌木丛，甚至包括撒哈拉沙漠南部一些多山的地带。非洲野犬从来不到密林中活动。 
```

定义一个向量库索引类 `VectorStoreIndex`
```python
# 定义向量库索引类
class VectorStoreIndex:
    """
    用于创建向量库索引，计算文本之间的相似度并进行查询的类。
    """

    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        """
        初始化方法，从文件中加载文档并计算它们的嵌入向量。

        :param document_path: 包含文档的文件路径。
        :param embed_model: 用于生成文档嵌入向量的模型实例。
        """
        self.documents = []  # 存储文档文本的列表
        # 从文件中读取文档并添加到文档列表中
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        # 存储嵌入模型的引用
        self.embed_model = embed_model

        # 为所有文档计算嵌入向量
        self.vectors = self.embed_model.get_embeddings(self.documents)

        # 打印加载文档的数量和文件路径
        print(f'Loading {len(self.documents)} documents for {document_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。

        :param vector1: 第一个向量。
        :param vector2: 第二个向量。
        :return: 两个向量之间的余弦相似度。
        """
        # 计算两个向量的点积
        dot_product = np.dot(vector1, vector2)
        # 计算两个向量的模
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        # 如果模为0，返回0以避免除以0的错误
        if not magnitude:
            return 0
        # 返回归一化的点积作为余弦相似度
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        根据问题查询最相似的文档。

        :param question: 查询的问题文本。
        :param k: 返回最相似文档的数量，默认为1。
        :return: 最相似的文档列表。
        """
        # 为问题文本计算嵌入向量
        question_vector = self.embed_model.get_embeddings([question])[0]
        # 计算问题向量与所有文档向量的相似度
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        # 获取最相似的k个文档的索引
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()
```
测试：
```python
print("> Create index...")​
doecment_path = './knowledge.txt'​
index = VectorStoreIndex(doecment_path, embed_model)

question = '介绍一下广州大学'​
print('> Question:', question)
context = index.query(question)​
print('> Context:', context)
```
>如果知识库很大，需要将知识库切分成多个batch，然后分批次送入向量模型。这里，因为我们的知识库比较小，所以就直接传到了get_embeddings() 函数。

返回结果：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/00ce9b868ac742cc95889a47fcfca5ef.png#pic_center)
我们传入用户问题 介绍一下广州大学，可以看到，准确地返回了知识库中的第一条知识。

#### 生成
编写`Yuan2-2B-Mars-hf-download.py`，下载大模型Yuan2-2B-Mars-hf：
```python
# 源大模型下载​
from modelscope import snapshot_download​
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
```
定义一个大语言模型类 LLM：
```python
# 导入必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 定义大语言模型类
class LLM:
    """
    用于加载和使用Yuan2.0大型语言模型的类。
    """

    def __init__(self, model_path: str) -> None:
        """
        初始化方法，加载预训练的分词器和模型。

        :param model_path: 预训练模型的路径。
        """
        print("Creat tokenizer...")  # 打印创建分词器的信息
        # 加载预训练的分词器，设置不自动添加结束和开始标记
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')

        # 向分词器添加特殊标记
        self.tokenizer.add_tokens([
            '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
            '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
            '<jupyter_code>', '<jupyter_output>', '<empty_output>'
        ], special_tokens=True)

        print("Creat model...")  # 打印创建模型的信息
        # 加载预训练的因果语言模型，并将其设置为半精度浮点数以提高计算效率
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        # 打印加载模型的信息
        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List[str]):
        """
        根据问题和上下文生成回答。

        :param question:  用户提问，是一个str。
        :param context: 检索到的上下文信息，是一个List，默认是[]，代表没有使用RAG。
        """
        # 如果提供了上下文，构建提示信息
        if context:
            prompt = f'背景：{" ".join(context)}\n问题：{question}\n请基于背景，回答问题。'
        else:
            prompt = question

        # 在提示信息后添加分隔符
        prompt += "<sep>"
        # 使用分词器将提示信息转换为模型输入的格式
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        # 使用模型生成文本
        outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        # 解码生成的文本
        output = self.tokenizer.decode(outputs[0])

        # 打印生成的文本，只显示分隔符之后的部分
        print(output.split("<sep>")[-1])
```
测试：
```python
print("> Create Yuan2.0 LLM...")​
model_path = './IEITYuan/Yuan2-2B-Mars-hf'​
llm = LLM(model_path)

print('> Without RAG:')​
llm.generate(question, [])​
print('> With RAG:')​
llm.generate(question, context)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b7f522af38be4c9f9ab0e23febb72b35.png#pic_center)
#### 打包
编写`requirements.txt`：
```txt
transformers
torch
numpy
```
编写安装脚本：`pip_install.sh`
```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
新建`main.py`
```python
from common import constants
from generation.llm import LLM
from indexing.embedding import EmbeddingModel
from retrieval.vector import VectorStoreIndex


def main():
    print("> Create embedding model...")
    embed_model_path = constants.EMBED_MODEL_PATH
    embed_model = EmbeddingModel(embed_model_path)

    print("> Create index...")
    document_path = constants.DOCUMENT_PATH
    index = VectorStoreIndex(document_path, embed_model)

    question = '介绍一下广州大学'
    print('> Question:', question)
    context = index.query(question)
    print('> Context:', context)

    print("> Create Yuan2.0 LLM...")
    model_path = constants.MODEL_PATH
    llm = LLM(model_path)
    print('> Without RAG:')
    llm.generate(question, [])
    print('> With RAG:')
    llm.generate(question, context)


if __name__ == '__main__':
    main()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d9a3794baf749b58d9aeca28110f18c.png#pic_center)
#### 部署
将代码部署到Github：https://github.com/itxaiohanglover/rag_demo

然后进入终端，导入写好的代码：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/184f006960c64630a08e9582b5cd76ef.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e8a1160a15354d5ca4509aede1c72fd3.png#pic_center)

下载模型：
```shell
python setup.py
```

启动代码：
```shell
python main.py
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/80332d89e31d4d5ca839d905499f2d41.png#pic_center)