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
#from vllm_houmo.models.export_meta_from_gguf import parse_gguf


os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAX_PIXELS"] = "1229312"
os.environ['FPS_MAX_FRAMES'] = "1"

# 多卡设置tensor_parallel_size为卡数
#engine = VllmEngine(model_path,model_type='qwen2_5_vl',gpu_memory_utilization=0.9,limit_mm_per_prompt={"image": 1},tensor_parallel_size=2)

def resize_with_pad(image, target_width, target_height):
    # 原始宽高
    orig_width, orig_height = image.size
    # 计算缩放比例（按目标高度缩放）
    scale = target_height / orig_height
    # 缩放后的宽高（保持比例）
    new_width = int(orig_width * scale)
    new_height = target_height  # 与目标高度一致
    # 缩放图片
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # 创建目标尺寸的空白画布（白色背景）
    padded = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    # 计算左右留白位置（居中放置）
    left = (target_width - new_width) // 2
    padded.paste(resized, (left, 0))  # 顶部对齐
    return padded


from qwen_vl_utils import process_vision_info
def vllm_qwen_ocr_cn(img_path):
    vllm_target_device_name = os.getenv('VLLM_TARGET_DEVICE')
    # 指定模型名称，可以根据需要选择不同版本的Qwen
    if vllm_target_device_name=="cpu": #load cpu model safetensors
        #7B要 360秒
        #model_name = "/home/sunzhenjiang/szj_project/Qwen2.5-7B-Instruct/"
        #model_name = "/home/sunzhenjiang/szj_project/Qwen3-8B/"
        #0.5B的耗时比较快,只有30秒
        #M30 45那台的cpu模型路径
        #model_name = "/home/sunzhenjiang/szj_project/Qwen2-0.5B/"
        #M50 60那台的cpu模型路径
        model_path = "/home/user/szj_project/models/qwen2.5-vl-3b/"
        #公网20.2那台
        #model_name = "/home/sunzhenjiang/szj_project/vllm_cpu_related_src/Qwen2-0.5B"
        print(f"load model{model_path}: on cpu")
    else: #houmo plugin model
        #model_path = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen2.5_vl_1batch/output/xh2/"
        #3b hiden state 不对
        #model_path = "/home/useradmin/szj_project/llm/vllm_cpu_related_src/qwen2.5-vl-3b/"

        #下面的路径，会报safetensor解析错误的日志，header too small 之类的
        #model_path ="/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen2.5_vl_1batch/"
        model_path ="/home/user/szj_project/models/qwen25_vl_8b/"
        #下面路径哦会报：Value error, GGUF model with architecture qwen2vl is not supported yet
        #model_path ="/opt/houmo/models/HiModel-Qwen2.5-vl-7b-p256-8K-b1-2core-1d-xh2-v0.5.0-20251029102452.gguf"


    #parse_gguf(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    model_h=448
    model_w=448
    vllm_qwen = LLM(
            model=model_path,
            #tokenizer=model_path,
            max_num_seqs=1,
            max_model_len=8192*2,
            #enable_memory_profiling=True,
            additional_config={"visual_model_path":"/home/user/szj_project/models/qwen25_vl_8b/mmproj_xh2_qwen2.5-vl_7b_2cores_vit_448x448_v0.7.0_20251228.gguf",
                               "visual_model_w":model_w,
                               "visual_model_h":model_h},
            gpu_memory_utilization=0.9,limit_mm_per_prompt={"image": 1},tensor_parallel_size=1
    )
    #prompt_ocr="你是一个OCR专家，请提取以下图片中的所有文字内容，并原样返回。"
    prompt_ocr="分析一下这张图片有哪些内容"
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_ocr},
                    {
                        "type": "image",
                        "image": img_path,
                    },
                ],
            }
    ]

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)
    #image_data[0]=resize_with_pad(image_data[0],model_w,model_h)
    print(f"##hm IN LLM_qwen25_vl_example.py image_data from process_vision_info is:{image_data}")
    sampling_params = SamplingParams(temperature=0.0,
                                max_tokens=4096,
                                stop_token_ids=stop_token_ids)

    outputs = vllm_qwen.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
        },
            sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
    return generated_text


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


if __name__ == '__main__':
    #generated_text=vllm_qwen_ocr_cn("/home/useradmin/szj_project/xh2a_0.4.0_v2/vllm/frame85.jpg")
    #generated_text=vllm_qwen_ocr_cn("/home/useradmin/szj_project/xh2a_0.4.0_v2/vllm/biaoman.jpg")
    with raw_terminal():
        generated_text=vllm_qwen_ocr_cn("./frame85.jpg")
        print(f"##hm generated_text:{generated_text}")