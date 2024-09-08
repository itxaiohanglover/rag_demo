# 导入必要的库
from typing import List
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