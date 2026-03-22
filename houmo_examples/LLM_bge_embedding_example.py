from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

import os
from PIL import Image

#增加终端恢复功能
import termios
import sys
from contextlib import contextmanager
import  torch
import numpy as np


@contextmanager
def raw_terminal():
    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    try:
        new_attr = old_attr.copy()
        #new_attr[3] &= ~(termios.ECHO | termios.ICANON) # 关闭回显和行缓冲
        #termios.tcsetattr(fd, termios.TCSANOW, new_attr)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attr) # 无论是否异常都恢复


# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    #6个不同的句子
    "北京有很多著名的牛肉面馆。",
    "上海是中国的金融中心。",
    "故宫是北京著名的旅游景点。",
    "兰州牛肉面是非常有名的中国美食。",
    "深圳以科技产业闻名。",
    "北京海淀区有一家西北牛肉面馆非常好吃。",

    #下面对比curl测试
    # "深圳以科技产业闻名。",
    # "深圳以科技产业闻名。",
    # "深圳以科技产业闻名。",
    # "深圳以科技产业闻名。",
    # "深圳以科技产业闻名。",
    # "深圳以科技产业闻名。"
]


torch.set_printoptions(
    precision=8,       # 保留小数位数（可选，按需调整）
    threshold=torch.inf,  # 打印阈值设为无穷大（不截断）
    linewidth=120,     # 每行字符数（避免换行过频繁）
    sci_mode=False     # 关闭科学计数法（可选，更易读）
)

if __name__ == '__main__':
    #generated_text=vllm_qwen_ocr_cn("/home/useradmin/szj_project/xh2a_0.4.0_v2/vllm/frame85.jpg")
    #generated_text=vllm_qwen_ocr_cn("/home/useradmin/szj_project/xh2a_0.4.0_v2/vllm/biaoman.jpg")
    with raw_terminal():
        # Create an LLM.
        #下面路径有相关json配置文件，写死了
        vllm_target_device_name = os.getenv('VLLM_TARGET_DEVICE')
        # 指定模型名称，可以根据需要选择不同版本的Qwen
        if vllm_target_device_name=="cpu": #load cpu model safetensors
            model_path="/home/user/szj_project/models/bge/bge-m3/"
        else:
            model_path="/home/user/szj_project/models/bge/bge-m3-with-jsonin/"
        model = LLM(model=model_path, enforce_eager=True)

        #cpu 跑rerank模型
        #model = LLM(model="/home/user/szj_project/models/bge/bge-reranker-v2-m3/",enforce_eager=True)

        #houmo plugin跑的gguf模型
        #下面几行代码放开后，运行会报错-bert架构不支持
        #model_name="/home/user/szj_project/models/bge/bge-m3/bge-m3-1.1B-Q4_0.gguf"
        # model = LLM(model=model_name,
        #             tokenizer=model_name,
        #             skip_tokenizer_init=True,
        #             enforce_eager=True)
        # Generate embedding. The output is a list of EmbeddingRequestOutputs.
        outputs = model.encode(prompts,pooling_task="embed")
        tensor_list = []
        print(f"##hm outputs type:{type(outputs )} content is:{outputs }")
        # Print the outputs.
        for idx, output in enumerate(outputs):
            #print(output.outputs.embedding)  # list of 4096 floats
            print(f"##hm idx:{idx} requestid:{output.request_id} output.outputs.data shape:{output.outputs.data.shape}: with content:{output.outputs.data} ")  # list of 4096 floats
            tensor_list.append(output.outputs.data)
        merged_tensor = torch.stack(tensor_list, dim=0)
        print("合并后Tensor形状：", merged_tensor.shape)  # 输出：torch.Size([6, 1024])
        merged_np = merged_tensor.cpu().numpy()
        print("转换后NumPy数组形状：", merged_np.shape)  # 输出：(6, 1024)
        np.savetxt(
            "./float_array_merged_np.txt",  # 输出文件路径
            merged_np,          # 要写入的浮点数组
            fmt="%.8f",         # 保留8位小数（控制精度，按需调整）
            delimiter=",",      # 分隔符（逗号/空格/制表符，推荐逗号）
            header="col1,col2", # 表头（可选，方便阅读）
            comments=""         # 取消表头注释符（否则读取时需跳过表头）
            )