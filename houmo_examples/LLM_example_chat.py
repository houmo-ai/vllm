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
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

logging.getLogger("vllm").setLevel(logging.DEBUG)
M30=0
M50=1

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} 函数执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int,default=2048)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add example params
    parser.add_argument("--chat-template-path", type=str)

    return parser



@timer_decorator
def main(args: dict):
  vllm_target_device_name = os.getenv('VLLM_TARGET_DEVICE')
  # 指定模型名称，可以根据需要选择不同版本的Qwen
  if vllm_target_device_name=="cpu": #load cpu model safetensors
     model_name = "/home/user/szj_project/models/Qwen2-0.5B"
     print(f"load model{model_name}: on cpu")
  else: #houmo plugin model
     if M30:
        model_name = "/opt/hicpp/models/qwen3-8b-1batch/4core_8k/"
     elif M50:
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_only_gguf/HiModel-Qwen3-8b-p256-8K-2core-1d-xh2-v0.3.1-20250918171140.gguf"
            #szj单batch正常能跑的
            model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_1batch/HiModel_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v0.6.0_20251203.gguf"

            #szj下面的qwen3 8b 4k模型，据说用开源的训的， 模型会在think阶段循环卡死
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_1batch/HiModel_xh2_qwen3_8b_256_4k_b1_1chip_2cores_v0.6.0_20251203.gguf"
            #4batch 输出有乱码
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_4batch/HiModel-Qwen3-8b-p256-8K-b4-2core-1d-xh2-v0.4.0-20251022171622.gguf"
            #szj下面是4batch 在0.6.0驱动上输出不正常-不是模型问题，是tokenizer的问题
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_4batch/HiModel-Qwen3-8b-p256-8K-b4-2core-1d-xh2-v0.3.1-20251020095145.gguf"

            #4batch 正常输出的模型-v0.6.0版本，需要add_special_tokens
            #model_name = "/home/useradmin/szj_project/xh2a_0.4.0_v2/models/qwen3_8b_4batch/HiModel_xh2_qwen3_8b_256_8k_b4_1chip_2cores_v0.6.0_20251203.gguf"

            #szj下面是qwen3 30b moe模型
            #model_name = "/home/szj/models/qwen3_moe_30B/HiModel_xh2_qwen3_30b-a3b_256_2k_b1_1chip_2cores_v0.6.0_20251130.gguf"
            #szj下面是qwen3 30b coder moe模型
            #model_name ="/home/szj/models/qwen3_moe_30B/HiModel_xh2_qwen3_30b_a3b_coder_256_16k_b1_1chip_2cores_v0.6.0_20251203.gguf"


            #gpt oss模型
            model_name="/home/useradmin/szj_project/xh2a_0.4.0_v2/models/gpt_oss_20b/"

            #v1.0.0的模型
            #qwen3 8b 32K
            #model_name="/home/user/szj_project/models/v1.0.0/HiModel_xh2_qwen3_8b_256_32k_b1_1chip_2cores_v1.0.0_20260205/"

            #qwen3 8b 16k b4
            model_name="/home/user/szj_project/models/v1.2.0/qwen3_8b_16k_4batch/"

  print(f"load model{model_name}: on houmo npu")
  # 创建LLM实例，根据你的硬件配置调整参数
  # 对于MAC系统，使用CPU运行，设置gpu_memory_utilization为0.5以避免内存溢出

  # Pop arguments not used by LLM
  max_tokens = args.pop("max_tokens")
  temperature = args.pop("temperature")
  top_p = args.pop("top_p")
  top_k = args.pop("top_k")
  chat_template_path = args.pop("chat_template_path")

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
      #下面示例，指定模型运行在指定的哪些device_id上,如果不指定，默认运行在device 0上
      additional_config={
                        "device_ids":[0],
                        },
  )

  # 定义采样参数
  # Create sampling params object
  sampling_params = llm.get_default_sampling_params()
  if max_tokens is not None:
      sampling_params.max_tokens = max_tokens
  if temperature is not None:
      sampling_params.temperature = temperature
  if top_p is not None:
      sampling_params.top_p = top_p
  if top_k is not None:
      sampling_params.top_k = top_k
  #hm 设置成0 代表greedy sample ，decode 速率最快
  sampling_params.temperature = 0
  print(f"##hm in LLM_example**.py sampling_params is:{sampling_params}")
  sampling_params.skip_special_tokens=False
  print(f"##hm in LLM_example**.py after change skip_special_tokens sampling_params is:{sampling_params}")


  def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)

    print("=" * 80)

  # In this script, we demonstrate how to pass input to the chat method:
  conversation = [
      {"role": "system", "content": "You are a helpful assistant"},
      {
          "role": "user",
          #"content": "Write an essay about the importance of higher education.",
          "content": "介绍一下大语言模型Qwen的特点以及它是哪个公司的产品,尽量控制200个字左右",

      },
  ]


  # 初始化循环计数器和总时间
  loop_count = 0
  total_time = 0.0

  #while 1:
  if 1:
    # 记录循环开始时间
    loop_start_time = time.time()

    # 生成回答
    print(f"chat 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loop_start_time))}")
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False,chat_template_kwargs={"enable_thinking": True})

    # 记录循环结束时间
    loop_end_time = time.time()
    print(f"chat 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loop_end_time))}")

    # 计算当前循环耗时
    current_loop_time = loop_end_time - loop_start_time
    #print_outputs(outputs)
    out_lenth = []
    # 打印结果
    index=0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt({index}): {prompt}")
        print(f"Generated({index}): text lenth:{len(generated_text)} with content: {generated_text}")
        print("+" * 80)

        index+=1

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


#参考官方示例代码：https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/chat.py
if __name__ == '__main__':
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
