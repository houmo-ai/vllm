from openai_harmony import (
    load_harmony_encoding, HarmonyEncodingName,
    Role, Message, Conversation, DeveloperContent,ToolDescription,StreamableParser
)
from vllm import LLM, SamplingParams
import os

# 1. 加载编码器
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# 2. 构建 Harmony 格式对话（包含工具定义）
if 0:
    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, "You are a helpful assistant."),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new()
            .with_instructions("回答问题并使用工具")
            .with_tools([{
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }])  # 工具在此声明
        ),
        Message.from_role_and_content(Role.USER, "北京天气？")
    ])
else:
    # 定义工具
    developer_message = DeveloperContent.new().with_instructions("使用友好语气回答").with_function_tools([
            ToolDescription.new(
                "get_current_weather",
                "获取指定地点的当前天气",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市和州，例如：San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                }
            )
        ])

    # 创建对话
    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "旧金山天气如何？")
    ])

    # 渲染对话为模型输入
    #tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)


# 3. 渲染成模型能理解的 token IDs
prefill_ids = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

stop_token_ids = enc.stop_tokens_for_assistant_actions()

MODEL_NAME = "/home/user/szj_project/models/v1.0.0/gpt_oss_20b_128k/"
# --- 2) Run vLLM with prefill ---
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization = 0.95,
    max_num_batched_tokens=4096,
    max_model_len=50000,
    tensor_parallel_size=1
)

sampling = SamplingParams(
    max_tokens=12800,
    temperature=0.0,
    stop_token_ids=stop_token_ids,
)

input_tokens = {"prompt_token_ids":prefill_ids}
print(f"##hm prefill_ids with type:{type(prefill_ids)} content is:{prefill_ids}")
outputs = llm.generate(
    input_tokens,   # batch of size 1
    sampling_params=sampling,
)



# 4. 将 prompt_ids 喂给模型（如 llm.generate），并获得 output_ids
#output_ids = model.generate(prompt_ids, stop_token_ids=enc.stop_tokens_for_assistant_actions())

gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)


# 5. 解析输出
parsed = enc.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
print(f"##hm last parsed:{parsed}")