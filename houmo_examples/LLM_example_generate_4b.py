import sys
import os
import logging
from vllm import LLM, SamplingParams
from vllm.logger import logger
import time
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

from gguf import GGUFReader
import json
from vllm_houmo.models.export_meta_from_gguf import parse_gguf


def convert_numpy_type(value):
    """将numpy类型（如memmap、ndarray）转换为Python原生类型"""
    if isinstance(value, (np.memmap, np.ndarray)):
        # 若为单元素数组，直接取 scalar 值
        if value.size == 1:
            return value.item()  # 转换为 Python 标量（int/float等）
        # 若为多元素数组，转换为 Python 列表
        return value.tolist()
    # 其他类型直接返回
    return value

def gguf_to_config(gguf_path, output_config_path,custom_tokenizer):
    # 读取GGUF文件并提取元数据（处理numpy类型）
    with open(gguf_path, "rb") as f:
        reader = GGUFReader(f)
        metadata = {}
        for key, field in reader.fields.items():
            # 提取值并转换numpy类型
            raw_value = field.parts[field.data[0]]
            converted_value = convert_numpy_type(raw_value)
            metadata[key] = converted_value
    logger.info(f"--------------------------------------")
    logger.info(f"metadata from gguf is:{metadata}")
    logger.info(f"======================================")
    # 映射元数据到Qwen3 config参数
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": metadata.get("qwen3.attention_bias", False),
        "attention_dropout": metadata.get("qwen3.attention_dropout", 0.0),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "head_dim": metadata.get("qwen3.attention.value_length"),
        "hidden_act": metadata.get("qwen3.activation_function", "silu"),
        "hidden_size": metadata.get("qwen3.embedding_length"),
        "initializer_range": metadata.get("qwen3.initializer_range", 0.02),
        "intermediate_size": metadata.get("qwen3.feed_forward_length"),
        "max_position_embeddings": metadata.get("qwen3.context_length"),
        "max_window_layers": metadata.get("qwen3.block_count", 36),
        "num_attention_heads": metadata.get("qwen3.attention.head_count"),
        "num_hidden_layers": metadata.get("qwen3.block_count"),
        "num_key_value_heads": metadata.get("qwen3.attention.head_count_kv"),
        "rms_norm_eps": metadata.get("qwen3.rms_norm_eps", 1e-06),
        "rope_scaling": metadata.get("qwen3.rope.scaling", None),
        "rope_theta": metadata.get("qwen3.rope.freq_base", 1000000),
        "sliding_window": metadata.get("qwen3.sliding_window", None),
        "tie_word_embeddings": metadata.get("qwen3.tie_word_embeddings", False),
        "torch_dtype": metadata.get("qwen3.torch_dtype", "bfloat16"),
        "transformers_version": metadata.get("qwen3.transformers_version", "4.51.0"),
        "use_cache": metadata.get("qwen3.use_cache", True),
        "use_sliding_window": metadata.get("qwen3.use_sliding_window", False),
        "vocab_size":custom_tokenizer.vocab_size
    }

    # 保存为config.json
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Qwen3 config.json已生成：{output_config_path}")


def gguf_to_config_error2(gguf_path, output_config_path):
    # 读取GGUF文件
    with open(gguf_path, "rb") as f:
        reader = GGUFReader(f)

    # 【关键修改】获取元数据（适配新版API）
    # 方法1：通过get_metadata()获取所有元数据（返回字典）
    metadata = reader.get_metadata()  # 适用于gguf-py 0.6.0+

    # 方法2：若get_metadata()不存在，可遍历metadata_items（返回键值对迭代器）
    # metadata = {k: v for k, v in reader.metadata_items}

    # 映射为Hugging Face config字段（以LLaMA为例，其他模型需调整）
    config = {
        "model_type": metadata.get("general.architecture", "llama"),
        "architectures": [f"{metadata.get('general.architecture', 'llama').capitalize()}ForCausalLM"],
        "hidden_size": metadata.get("llama.embedding_length"),
        "num_hidden_layers": metadata.get("llama.block_count"),
        "num_attention_heads": metadata.get("llama.attention_head_count"),
        # 处理num_key_value_heads（部分模型可能没有，默认与attention_head_count一致）
        "num_key_value_heads": metadata.get("llama.attention_head_count_kv",
                                           metadata.get("llama.attention_head_count")),
        "max_position_embeddings": metadata.get("llama.context_length"),
        "rms_norm_eps": metadata.get("llama.attention.layer_norm_rms_epsilon", 1e-6),
        "rope_theta": metadata.get("llama.rope.freq_base", 10000.0),
        "vocab_size": metadata.get("tokenizer.ggml.vocab_size"),
        "tie_word_embeddings": False
    }

    # 保存为config.json
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"config.json已生成：{output_config_path}")


def gguf_to_config_error3(gguf_path, output_config_path):
    # 读取GGUF文件
    with open(gguf_path, "rb") as f:
        reader = GGUFReader(f)

    # 【关键修改】适配0.17.0版本：通过items()迭代获取元数据
    metadata = {}
    for key, value in reader.items():  # 遍历所有元数据项
        metadata[key] = value

    # 映射为Hugging Face config字段（以LLaMA为例）
    config = {
        "model_type": metadata.get("general.architecture", "llama"),
        "architectures": [f"{metadata.get('general.architecture', 'llama').capitalize()}ForCausalLM"],
        "hidden_size": metadata.get("llama.embedding_length"),
        "num_hidden_layers": metadata.get("llama.block_count"),
        "num_attention_heads": metadata.get("llama.attention_head_count"),
        "num_key_value_heads": metadata.get("llama.attention_head_count_kv",
                                           metadata.get("llama.attention_head_count")),
        "max_position_embeddings": metadata.get("llama.context_length"),
        "rms_norm_eps": metadata.get("llama.attention.layer_norm_rms_epsilon", 1e-6),
        "rope_theta": metadata.get("llama.rope.freq_base", 10000.0),
        "vocab_size": metadata.get("tokenizer.ggml.vocab_size"),
        "tie_word_embeddings": False
    }

    # 保存为config.json
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"config.json已生成：{output_config_path}")

def gguf_to_config_error4(gguf_path, output_config_path):
        # 读取GGUF文件并提取元数据（基于reader.fields逻辑）
    with open(gguf_path, "rb") as f:
        reader = GGUFReader(f)
        # 构建元数据字典：从reader.fields中提取键和值
        metadata = {}
        for key, field in reader.fields.items():
            # 从field中获取值（参考你提供的示例逻辑：field.parts[field.data[0]]）
            value = field.parts[field.data[0]]
            metadata[key] = value
    logger.info(f"metadata is:{metadata}")
    # 映射GGUF元数据到Qwen3的config参数
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": metadata.get("qwen3.attention_bias", False),
        "attention_dropout": metadata.get("qwen3.attention_dropout", 0.0),
        "bos_token_id": metadata.get("tokenizer.bos_token_id"),
        "eos_token_id": metadata.get("tokenizer.eos_token_id"),
        "head_dim": metadata.get("qwen3.head_dim"),
        "hidden_act": metadata.get("qwen3.activation_function", "silu"),
        "hidden_size": metadata.get("qwen3.embedding_length"),
        "initializer_range": metadata.get("qwen3.initializer_range", 0.02),
        "intermediate_size": metadata.get("qwen3.feed_forward_length"),
        "max_position_embeddings": metadata.get("qwen3.context_length"),
        "max_window_layers": metadata.get("qwen3.max_window_layers", 36),
        "num_attention_heads": metadata.get("qwen3.attention_head_count"),
        "num_hidden_layers": metadata.get("qwen3.block_count"),
        "num_key_value_heads": metadata.get("qwen3.attention_head_count_kv"),
        "rms_norm_eps": metadata.get("qwen3.rms_norm_eps", 1e-06),
        "rope_scaling": metadata.get("qwen3.rope.scaling", None),
        "rope_theta": metadata.get("qwen3.rope.freq_base", 1000000),
        "sliding_window": metadata.get("qwen3.sliding_window", None),
        "tie_word_embeddings": metadata.get("qwen3.tie_word_embeddings", False),
        "torch_dtype": metadata.get("qwen3.torch_dtype", "bfloat16"),
        "transformers_version": metadata.get("qwen3.transformers_version", "4.51.0"),
        "use_cache": metadata.get("qwen3.use_cache", True),
        "use_sliding_window": metadata.get("qwen3.use_sliding_window", False),
        "vocab_size": metadata.get("tokenizer.ggml.vocab_size")
    }

    logger.info(f"##hm config is:{config}")

    # 保存为config.json
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Qwen3 config.json已生成：{output_config_path}")



def get_files_by_suffix(folder_path, suffix):
    """
    获取指定文件夹下指定后缀的文件名（不含路径）
    """
    # 确保后缀以 '.' 开头
    if not suffix.startswith('.'):
        suffix = '.' + suffix

    # 初始化路径对象
    folder = Path(folder_path)

    # 筛选文件并匹配后缀（glob 方法支持通配符）
    target_files = [
        file.name  # 只取文件名（不含路径）
        for file in folder.glob(f'*{suffix}')  # 匹配所有带指定后缀的文件
        if file.is_file()  # 确保是文件
    ]
    return target_files


logging.getLogger("vllm").setLevel(logging.DEBUG)
M30=0
M50=1

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 函数执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

@timer_decorator
def main():
  # Add vLLM installation path
  custom_tokenizer = None
  vllm_path = "/home/sunzhenjiang/anaconda3/bin/vllm"  # Replace with actual path from `which vllm`
  sys.path.append(os.path.dirname(vllm_path))
  sys.path.append("/home/sunzhenjiang/szj_project/vllm_cpu_related_src/vllm")  # 替换为实际 vLLM 源码目录

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
     model_name = "/home/useradmin/szj_project/llm/vllm_cpu_related_src/Qwen2-0.5B"
     #公网20.2那台
     #model_name = "/home/sunzhenjiang/szj_project/vllm_cpu_related_src/Qwen2-0.5B"
     print(f"load model{model_name}: on cpu")
  else: #houmo plugin model
     if M30:
        model_name = "/opt/hicpp/models/qwen3-8b-1batch/4core_8k/"
     elif M50:
        #model_name = "/home/szj/models/qwen2.5-coder-7b-8k/"
        #model_name = "/home/szj/models/HiModel-Qwen3-14b-p256-16K-2core-1d-xh2-v0.3.1-20250919190044.gguf"
        #qwen3 8B
        #model_name = "/home/szj/models/qwen3_8b_2k_0918/"
        #qwen3 14B 单卡
        if 0: #从文件夹加载gguf，同时save config 等
            model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b/"
            #qwen3 14B双卡
            #model_name = "/home/szj/models/qwen3_14b_2card/"

            gguf_files=get_files_by_suffix(model_name,'gguf')
            if len(gguf_files) != 1:
                logger.info(f"!!!Error:found more than one gguf model file, Please check the folder:{model_name}")
                exit(-1)
            else:
                logger.info(f"find ggufmodelfile:{gguf_files[0]}")
            # 1. 加载GGUF格式的分词器
            print("Loading tokenizer from GGUF file...")
            custom_tokenizer = AutoTokenizer.from_pretrained(model_name,gguf_file=gguf_files[0])
            print("Tokenizer loaded successfully.")
            # 2. 保存分词器到本地目录
            save_dir = model_name
            print(f"Saving tokenizer to {save_dir}...")
            custom_tokenizer.save_pretrained(save_dir)
            print("custom_tokenizer saved successfully.")

            #custom_tokenizer.config.to_json_file(model_name+"./config.json")

            gguf_fullmodel_path=os.path.join(model_name,gguf_files[0])
            gguf_to_config(gguf_fullmodel_path,model_name+"./config.json",custom_tokenizer)
        else:
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_only_gguf/HiModel-Qwen3-8b-p256-8K-2core-1d-xh2-v0.3.1-20250918171140.gguf"
            #szj单batch正常能跑的
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_1batch/HiModel_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v0.6.0_20251203.gguf"

            #szj下面的qwen3 8b 4k模型，据说用开源的训的， 模型会在think阶段循环卡死
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_1batch/HiModel_xh2_qwen3_8b_256_4k_b1_1chip_2cores_v0.6.0_20251203.gguf"
            #4batch 输出有乱码
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_4batch/HiModel-Qwen3-8b-p256-8K-b4-2core-1d-xh2-v0.4.0-20251022171622.gguf"
            #szj下面是4batch 在0.6.0驱动上输出不正常-不是模型问题，是tokenizer的问题
            model_name = "/home/user/szj_project/models/v1.2.0/qwen3_8b_16k_4batch/"

            #4batch 正常输出的模型-v0.6.0版本，用vllm的tokenizer的话，需要add_special_tokens
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_4batch/HiModel_xh2_qwen3_8b_256_8k_b4_1chip_2cores_v0.6.0_20251203.gguf"

            #szj下面是qwen3 30b moe模型
            #model_name = "/home/szj/models/qwen3_moe_30B/HiModel_xh2_qwen3_30b-a3b_256_2k_b1_1chip_2cores_v0.6.0_20251130.gguf"
            #szj下面是qwen3 30b coder moe模型
            #model_name ="/home/szj/models/qwen3_moe_30B/HiModel_xh2_qwen3_30b_a3b_coder_256_16k_b1_1chip_2cores_v0.6.0_20251203.gguf"

     print(f"load model{model_name}: on houmo npu")
  #parse_gguf(model_path)

  # 创建LLM实例，根据你的硬件配置调整参数
  # 对于MAC系统，使用CPU运行，设置gpu_memory_utilization为0.5以避免内存溢出
  llm = LLM(
      model=model_name,
      tokenizer=model_name,
      #tokenizer_mode="slow",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.5,
      dtype="float16", # 根据你的硬件支持选择合适的数据类型
      trust_remote_code=False,
      #tokenizer=custom_tokenizer,  # 传入自定义 Tokenizer
      #skip_tokenizer_init=True,    # 明确跳过自动初始化，避免冲突
  )

  # 定义采样参数
  sampling_params = SamplingParams(
      temperature=0.0,
      #top_k=1,
      #logprobs=5,
      #top_p=0.001,
      seed=42,
      max_tokens=4096
  )

  # 示例输入
  prompts = [
      #"如何在日常生活中提高编程技能",


      #"/no_think 用C++写个冒泡排序",
      "北京市是",
    #下面9个问题用于chat 多batch测试
    "用C++写个冒泡排序,尽量控制200个字左右",
       "介绍一下大语言模型Qwen的特点以及它是哪个公司的产品,尽量控制200个字左右",
       "介绍一下孙中山的生平,尽量控制200个字左右",
       "讲讲三国演义这部小说,尽量控制200个字左右",
    #   "介绍一下老舍,尽量控制200个字左右",
    #   "介绍一下秦始皇,尽量控制200个字左右",
    #   "介绍一下长城,尽量控制200个字左右",
    #   "介绍一下白宫,尽量控制200个字左右",
    #   "介绍一下故宫,尽量控制200个字左右",
  ]

  # 初始化循环计数器和总时间
  loop_count = 0
  total_time = 0.0

  #while 1:
  if 1:
    # 记录循环开始时间
    loop_start_time = time.time()

    # 生成回答
    outputs = llm.generate(prompts, sampling_params)
    out_lenth = []
    # 打印结果
    index=0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt({index}): {prompt}")
        print(f"Generated({index}): text lenth:{len(generated_text)} with content: {generated_text}")
        print("+" * 80)

        #generated_token_num = len(output.outputs[0].token_ids)
        #out_lenth.append(len(generated_token_num))


        # 输入prompt的Token数（属性名prompt_token_ids仍通用）
        #prompt_token_num = len(output.prompt_token_ids)

        # 下面方法可以获取所有的输出tokenids
        #token_ids = [output.outputs[0].token_ids for output in outputs]

        #szj-下面获取text长度的方法不对
        #out_lenth.append(len(generated_text))

        index+=1

    # 记录循环结束时间
    loop_end_time = time.time()

    # 计算当前循环耗时
    current_loop_time = loop_end_time - loop_start_time

    # 更新总耗时和循环计数
    total_time += current_loop_time
    loop_count += 1

    # 计算平均耗时
    average_time = total_time / loop_count

    #total_generated_tokens = sum(len(output.output_token_ids) for output in outputs)
    #print(f"总生成Token数：{total_generated_tokens}")

    total_prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    print(f"总输入Token数：{total_prompt_tokens}")

    for output in outputs:
        generated_token_num = len(output.outputs[0].token_ids)
        out_lenth.append(generated_token_num)

    # 打印统计信息
    print(f"循环总次数: {loop_count}")
    print(f"循环   {loop_count} 次总耗时: {total_time:.4f} 秒")
    print(f"当前第 {loop_count} 次循环开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loop_start_time))}")
    print(f"当前第 {loop_count} 次循环结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loop_end_time))}")
    print(f"当前第 {loop_count} 次循环耗时: {current_loop_time:.4f} 秒")
    print(f"当前第 {loop_count} 次循环所有问题输出token总数: {sum(out_lenth):} tokens")
    print(f"当前第 {loop_count} 次循环输出token速率: {sum(out_lenth)/current_loop_time:.4f} tokens/秒")
    print(f"{loop_count}次循环  平均耗时: {average_time:.4f} 秒")
    print("-" * 80)
    time.sleep(3)


if __name__ == '__main__':
    main()  # 确保所有多进程相关代码在这个条件内执行
