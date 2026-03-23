# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")


def is_gcs(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("gs://")


def is_azure(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("az://")


def is_cloud_storage(model_or_path: str) -> bool:
    return is_s3(model_or_path) or is_gcs(model_or_path) or is_azure(model_or_path)


def without_trust_remote_code(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs without trust_remote_code without modifying original dict."""
    if "trust_remote_code" not in kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k != "trust_remote_code"}


def modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    # same as huggingface_hub.list_repo_files
    files = [
        file["Path"]
        for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
        if file["Type"] == "blob"
    ]
    return files


def _maybe_json_dict(path: str | PathLike) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: str | PathLike) -> dict[str, str]:
    parsed_dict = dict[str, str]()
    with open(path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.strip().split()
                parsed_dict[model_name] = redirect_name
            except Exception:
                pass
    return parsed_dict


@cache
def maybe_model_redirect(model: str) -> str:
    """
    Use model_redirect to redirect the model name to a local folder.

    :param model: hf model name
    :return: maybe redirect to a local folder
    """

    model_redirect_path = envs.VLLM_MODEL_REDIRECT_PATH

    if not model_redirect_path:
        return model

    if not Path(model_redirect_path).exists():
        return model

    redirect_dict = _maybe_json_dict(model_redirect_path) or _maybe_space_split_dict(
        model_redirect_path
    )
    if redirect_model := redirect_dict.get(model):
        logger.info("model redirect: [ %s ] -> [ %s ]", model, redirect_model)
        return redirect_model

    return model


def parse_safetensors_file_metadata(path: str | PathLike) -> dict[str, Any]:
    with open(path, "rb") as f:
        length_of_metadata = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(length_of_metadata).decode("utf-8"))
        return metadata


def convert_model_repo_to_path(model_repo: str) -> str:
    """When VLLM_USE_MODELSCOPE is True convert a model
    repository string to a Path str."""
    if not envs.VLLM_USE_MODELSCOPE or Path(model_repo).exists():
        return model_repo
    from modelscope.utils.file_utils import get_model_cache_root

    return os.path.join(get_model_cache_root(), model_repo)


import os
import sys
import json
import glob
from pathlib import Path
from gguf.gguf_reader import GGUFReader
import gc
from dataclasses import dataclass


#logger = init_logger(__name__)
globa_gguf_metainfo={}
globa_gguf_vl_metainfo={}

# 核心新增：定义张量元数据结构体类（仅保留需要的4个字段）
@dataclass
class TensorMetadata:
    """张量元数据结构体，存储核心字段，与原tensor对象解耦"""
    shape: tuple          # 对应tensor.shape
    n_elements: int       # 对应tensor.n_elements
    quantization: str     # 对应quantization_str（张量量化类型）
    data_offset: int      # 对应tensor.data_offset
    name: str             # 补充tensor.name，用于文件保存和标识（必要字段）

    def to_dict(self) -> dict:
        """将 TensorMetadata 转换为字典，确保 JSON 可序列化"""
        # 将 shape 转换为元组，并确保所有元素都是 Python int 类型
        shape_list = list(self.shape)
        converted_shape = []
        for item in shape_list:
            if hasattr(item, 'item'):
                # numpy 标量转换为 Python int
                converted_shape.append(int(item.item()))
            elif hasattr(item, 'tolist'):
                # numpy 数组转换为 Python 列表
                converted_shape.append(item.tolist())
            else:
                converted_shape.append(int(item))
        return {
            "shape": tuple(converted_shape),
            "n_elements": int(self.n_elements),
            "quantization": str(self.quantization),
            "data_offset": int(self.data_offset),
            "name": str(self.name),
        }

    def __repr__(self) -> str:
        return f"TensorMetadata(name={self.name!r}, shape={self.shape}, n_elements={self.n_elements}, quantization={self.quantization!r}, data_offset={self.data_offset})"

def get_gguf_metainfo():
   return globa_gguf_metainfo

def get_model_abspath(model_path):
    # 1. 转换为绝对路径，统一处理相对/绝对路径
    abs_model_path = os.path.abspath(model_path)

    # 2. 判断路径类型
    if os.path.isfile(abs_model_path):
        final_path = abs_model_path
    elif os.path.isdir(abs_model_path):
        # 查找文件夹下HiMode开头的gguf文件
        gguf_files = glob.glob(os.path.join(abs_model_path, "HiMode*.gguf"))
        # 严格校验文件数量：必须恰好1个
        if len(gguf_files) == 0:
            print(f"❌ 错误：文件夹 {abs_model_path} 下未找到HiMode开头的gguf文件！", file=sys.stderr)
            sys.exit(1)
        elif len(gguf_files) > 1:
            print(f"❌ 错误：文件夹 {abs_model_path} 下找到多个HiMode开头的gguf文件：{gguf_files}", file=sys.stderr)
            sys.exit(1)
        # 仅当数量为1时，取绝对路径
        final_path = os.path.abspath(gguf_files[0])
    else:
        # 路径既不是文件也不是文件夹（不存在）
        print(f"❌ 错误：路径 {abs_model_path} 不存在或不是有效文件/文件夹！", file=sys.stderr)
        sys.exit(1)

    # 最终可用的模型路径（替换后的gguf文件绝对路径）
    logger.info(f"processid:{os.getpid()} get final path：{final_path}")
    return final_path


def parse_gguf(gguf_file_path, additional_config=None):
    """
    从GGUF文件中提取所有张量元数据，按张量名称保存为文件（兼容无上下文管理器的GGUFReader）
    :param gguf_file_path: GGUF文件的路径（绝对/相对）
    :param additional_config: 额外配置（含visual_model_path）
    """
    # 1. 初始化局部变量（避免全局变量长期持有引用）
    local_gguf_metainfo = {}
    local_gguf_vl_metainfo = {}
    tensors_info = []       # 存储配置文件类张量的元数据对象
    hmm_tensors_info = []   # 存储权重类张量的元数据对象
    main_reader = None      # 主GGUFReader实例（单独命名，避免覆盖）
    vl_reader = None        # VL模型GGUFReader实例

    # 2. 路径处理
    gguf_file_path = get_model_abspath(gguf_file_path)
    output_dir = os.path.dirname(gguf_file_path)
    os.makedirs(output_dir, exist_ok=True)
    gguf_file = Path(gguf_file_path)
    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF文件不存在：{gguf_file_path}")

    try:
        # 3. 读取主GGUF文件（核心修改：手动创建Reader，不用with）
        print(f"正在读取GGUF文件：{gguf_file_path} and additional_config is:{additional_config}")
        # 直接创建Reader实例（无上下文管理器）
        main_reader = GGUFReader(gguf_file_path)

        # 读取KV字段
        print("Key-Value Pairs:")
        max_key_length = max(len(key) for key in main_reader.fields.keys()) if main_reader.fields else 0
        for key, field in main_reader.fields.items():
            value = field.parts[field.data[0]]
            print(f"{key:{max_key_length}} : {value}")
            # 将 numpy.memmap 或 numpy.ndarray 转换为 Python 基本类型，确保 JSON 可序列化
            if hasattr(value, 'tolist'):
                value = value.tolist()
            elif hasattr(value, 'item'):
                value = value.item()
            local_gguf_metainfo[key] = value

        # 读取张量信息（核心修改：存储自定义元数据对象，而非原tensor）
        print("----")
        print("Tensors:")
        tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Offset: {:<10} | Quantization: {}"
        print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Address", "Quantization"))
        print("-" * 80)
        for tensor in main_reader.tensors:
            # 提取核心字段
            # 将 shape 转换为元组，并确保所有元素都是 Python int 类型
            shape_list = list(tensor.shape)
            converted_shape = []
            for item in shape_list:
                if hasattr(item, 'item'):
                    converted_shape.append(int(item.item()))
                elif hasattr(item, 'tolist'):
                    converted_shape.append(item.tolist())
                else:
                    converted_shape.append(int(item))
            shape = tuple(converted_shape)
            n_elements = int(tensor.n_elements)
            quantization_str = str(tensor.tensor_type.name)
            data_offset = int(tensor.data_offset)
            tensor_name = str(tensor.name)
            shape_str = "x".join(map(str, shape))

            # 打印信息（逻辑不变）
            print(tensor_info_format.format(tensor_name, shape_str, str(n_elements), data_offset, quantization_str))

            # 创建自定义元数据对象，替代原tensor对象存储
            tensor_meta = TensorMetadata(
                shape=shape,
                n_elements=n_elements,
                quantization=quantization_str,
                data_offset=data_offset,
                name=tensor_name  # 保留名称，用于后续文件保存
            )

            # 按类型添加到对应列表（存储元数据对象，而非原tensor）
            if tensor_name.endswith((".json", ".txt", ".jinja", ".md")):
                tensors_info.append(tensor_meta)
            else:
                hmm_tensors_info.append(tensor_meta)

        # 4. 处理VL模型（单独创建Reader，避免覆盖主Reader）
        if additional_config and isinstance(additional_config, dict) and 'visual_model_path' in additional_config:
            logger.info(f"##hm get vl model config in additional_config")
            vl_model_path_value = additional_config["visual_model_path"]
            if not isinstance(vl_model_path_value, str):
                logger.error(f"Error:visual_model_path 类型错误（应为字符串），当前类型：{type(vl_model_path_value)}")
                sys.exit(-1)
            if not vl_model_path_value.strip():
                logger.error(f"Error:visual_model_path is empty")
                sys.exit(-1)

            full_path = os.path.abspath(vl_model_path_value)
            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    # 手动创建VL模型Reader
                    vl_reader = GGUFReader(full_path)
                    print("VL Model Key-Value Pairs:")
                    max_key_length = max(len(key) for key in vl_reader.fields.keys()) if vl_reader.fields else 0
                    for key, field in vl_reader.fields.items():
                        value = field.parts[field.data[0]]
                        print(f"{key:{max_key_length}} : {value}")
                        # 将 numpy.memmap 或 numpy.ndarray 转换为 Python 基本类型，确保 JSON 可序列化
                        if hasattr(value, 'tolist'):
                            value = value.tolist()
                        elif hasattr(value, 'item'):
                            value = value.item()
                        local_gguf_vl_metainfo[key] = value

                    print("----")
                    print("Tensors:")
                    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Address", "Quantization"))
                    print("-" * 80)
                    for tensor in vl_reader.tensors:
                        if tensor.name.endswith((".json", ".txt", ".jinja", ".md")):
                            continue
                        # 提取核心字段
                        # 将 shape 转换为元组，并确保所有元素都是 Python int 类型
                        shape_list = list(tensor.shape)
                        converted_shape = []
                        for item in shape_list:
                            if hasattr(item, 'item'):
                                converted_shape.append(int(item.item()))
                            elif hasattr(item, 'tolist'):
                                converted_shape.append(item.tolist())
                            else:
                                converted_shape.append(int(item))
                        shape = tuple(converted_shape)
                        n_elements = int(tensor.n_elements)
                        quantization_str = str(tensor.tensor_type.name)
                        data_offset = int(tensor.data_offset)
                        tensor_name = str(tensor.name)
                        shape_str = "x".join(map(str, shape))

                        # 打印信息（逻辑不变）
                        print(tensor_info_format.format(tensor_name, shape_str, str(n_elements), data_offset, quantization_str))

                        # 创建自定义元数据对象并添加
                        tensor_meta = TensorMetadata(
                            shape=shape,
                            n_elements=n_elements,
                            quantization=quantization_str,
                            data_offset=data_offset,
                            name=tensor_name
                        )
                        hmm_tensors_info.append(tensor_meta)
                else:
                    logger.error(f"Error:visual_model_path is dir or something, Please change to vl modelfile abs path")
                    sys.exit(-1)

        # 5. 保存配置文件张量（核心修改：使用元数据对象的字段）
        # 将 TensorMetadata 对象转换为字典，确保 JSON 可序列化
        local_gguf_metainfo['configfiles_ingguf_info'] = [meta.to_dict() for meta in tensors_info]
        local_gguf_metainfo['hmm_ingguf_info'] = [meta.to_dict() for meta in hmm_tensors_info]


        # 保存文件时，读取元数据对象的字段（替代原tensor）
        with open(gguf_file_path, 'rb') as f:
            for tensor_meta in tensors_info:
                # 从元数据对象获取字段，而非原tensor
                f.seek(tensor_meta.data_offset)
                data = f.read(tensor_meta.n_elements)
                output_file_path = os.path.join(output_dir, tensor_meta.name)
                file_dir = os.path.dirname(output_file_path)
                os.makedirs(file_dir, exist_ok=True)
                try:
                    with open(output_file_path, "wb") as out_f:
                        out_f.write(data)
                    print(f"数据已成功保存到：{output_file_path}")
                except Exception as e:
                    raise IOError(f"保存文件失败：{e}") from e

    finally:
        # 核心修复：手动关闭Reader，解除mmap映射（兼容有无close方法的情况）
        # 关闭主Reader
        if main_reader:
            if hasattr(main_reader, 'close'):
                try:
                    main_reader.close()
                except Exception as e:
                    logger.warning(f"关闭主GGUFReader时警告：{e}")
            main_reader = None  # 清空引用，帮助GC回收

        # 关闭VL模型Reader
        if vl_reader:
            if hasattr(vl_reader, 'close'):
                try:
                    vl_reader.close()
                except Exception as e:
                    logger.warning(f"关闭VL GGUFReader时警告：{e}")
            vl_reader = None  # 清空引用

    # 6. 释放所有大对象引用（关键：让GC能回收原tensor对象）
    global globa_gguf_metainfo, globa_gguf_vl_metainfo
    globa_gguf_metainfo = local_gguf_metainfo.copy()
    globa_gguf_vl_metainfo = local_gguf_vl_metainfo.copy()
    if additional_config is not None and isinstance(additional_config, dict):
            additional_config["globa_gguf_metainfo"] = globa_gguf_metainfo

    # 清空局部列表（仅清空引用，元数据对象的字段值仍保留在全局变量中）
    # 注意：此处clear后，global_gguf_metainfo中已保存的元数据不受影响
    tensors_info.clear()
    hmm_tensors_info.clear()
    local_gguf_metainfo.clear()
    local_gguf_vl_metainfo.clear()

    # 手动触发GC，加速回收原tensor对象（元数据对象不受影响）
    gc.collect()
    print(f"##hm after clear globa_gguf_metainfo is:{globa_gguf_metainfo} ")

    print(f"\n✅ 所有张量已保存至：{os.path.abspath(output_dir)}")


import time
import threading
import tracemalloc
import objgraph
from pathlib import Path

class VLLMMemoryMonitor:
    """vLLM内存泄漏监控类"""

    def __init__(self, inter_second=600, top_count=20, stack_depth=25, snapshot_dir="./tracemalloc_dir_szj"):
        """
        初始化监控类
        :param inter_second: 监控间隔（秒），默认10分钟
        :param top_count: 显示TOP N的内存增长项，默认20
        :param stack_depth: tracemalloc堆栈保留层数，默认25
        :param snapshot_dir: 内存快照保存目录，默认./tracemalloc_dir_szj
        """
        # 配置参数
        self.inter_second = inter_second
        self.top_count = top_count
        self.stack_depth = stack_depth
        self.snapshot_dir = Path(snapshot_dir)

        # 线程相关
        self.monitor_thread = None
        self.is_running = False  # 线程运行状态标记
        self.snapshot1 = None    # 初始内存快照

    def _init_tracemalloc(self):
        """初始化tracemalloc和objgraph"""
        # 创建快照保存目录
        self.snapshot_dir.mkdir(exist_ok=True)

        # 启动tracemalloc，设置堆栈深度
        tracemalloc.start(self.stack_depth)

        # 获取初始快照
        self.snapshot1 = tracemalloc.take_snapshot()

        # 打印初始对象增长（基准）
        print("=== Initial Object Growth (Baseline) ===")
        objgraph.show_growth(limit=self.top_count)
        print(f"\n✅ 内存监控初始化完成，监控间隔：{self.inter_second}秒")

    def _monitor_loop(self):
        """监控循环（内部方法，线程执行体）"""
        # 等待vLLM进程启动完成（避免初始阶段干扰）
        #print(f"\n⌛ 等待60秒，让vLLM进程完全启动...")
        #time.sleep(60)

        while self.is_running:
            try:
                print(f"\n========================================")
                print(f"##hm 开始新一轮监控 - {time.ctime()}")
                print(f"##hm 当前进程ID(PID)：{os.getpid()} | 父进程ID(PPID)：{os.getppid()}")
                print(f"========================================")

                # 1. 打印对象增长（看哪些对象在持续增加）
                print("\n=== Object Growth (since last check) ===")
                objgraph.show_growth(limit=self.top_count)

                # 2. 对比内存快照，找出内存增长的代码行
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(self.snapshot1, 'lineno')
                print("\n=== Top Memory Growth (by line) ===")
                for stat in top_stats[:self.top_count]:
                    print(stat)

                # 更新基准快照
                self.snapshot1 = snapshot2

                # 3. 保存内存快照到文件
                snapshot_file = self.snapshot_dir / f"memory_snapshot_pid_{os.getpid()}_{int(time.time())}.dat"
                snapshot2.dump(str(snapshot_file))
                print(f"\n📸 内存快照已保存到：{snapshot_file}")

                # 4. 等待下一次监控（可被停止信号中断）
                wait_seconds = self.inter_second
                while wait_seconds > 0 and self.is_running:
                    time.sleep(1)
                    wait_seconds -= 1

            except Exception as e:
                print(f"\n❌ 监控循环出错：{e}")
                # 出错后仍继续监控，仅打印错误
                time.sleep(10)

    def start_monitor(self):
        """启动监控线程（对外暴露的方法）"""
        if self.is_running:
            print("⚠️ 内存监控已在运行中，无需重复启动")
            return

        # 初始化监控环境
        self._init_tracemalloc()

        # 设置运行状态
        self.is_running = True

        # 创建并启动守护线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,  # 守护线程，主进程退出时自动终止
            name="VLLMMemoryMonitorThread"
        )
        self.monitor_thread.start()

        print(f"\n🚀 内存监控线程已启动（线程名：{self.monitor_thread.name}）")

    def stop_monitor(self):
        """停止监控线程（对外暴露的方法）"""
        if not self.is_running:
            print("⚠️ 内存监控未运行，无需停止")
            return

        # 设置停止标记
        self.is_running = False

        # 等待线程退出
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)  # 最多等待5秒

        # 停止tracemalloc
        tracemalloc.stop()

        print("\n🛑 内存监控线程已停止")

    def __del__(self):
        """析构函数，确保监控线程正常停止"""
        if self.is_running:
            self.stop_monitor()

# ====================== 使用示例 ======================
if 0:
    # 1. 创建监控实例（可自定义参数）
    monitor = VLLMMemoryMonitor(
        inter_second=600,  # 每10分钟监控一次
        top_count=20,      # 显示TOP20的增长项
        snapshot_dir="./tracemalloc_dir_szj"  # 快照保存目录
    )

    # 2. 启动监控
    monitor.start_monitor()

    # 3. 模拟主进程运行（替换为你的vLLM启动逻辑）
    try:
        print("\n📌 主进程运行中，按Ctrl+C停止...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n📤 接收到停止信号，正在退出...")
        # 4. 停止监控（可选，守护线程也会随主进程退出）
        monitor.stop_monitor()

