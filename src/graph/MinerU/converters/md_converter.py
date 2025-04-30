"""
Markdown 文件处理工具

该模块用于将 Markdown 格式的教材内容解析为结构化的 JSON 数据。
主要功能包括：
1. 提取 Markdown 文件中的标题和内容
2. 过滤掉图片链接和特定类型的标题（如章节、单元等）
3. 将处理后的内容保存为 JSON 格式
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any

# 导入工具函数
from ..utils.file_utils import get_absolute_file_paths
from ..utils.text_utils import remove_jpg_lines, parse_md

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 添加项目根目录到系统路径
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from utils import saveData


def process_directory(directory_path: str, file_type: str = "md") -> None:
    """
    处理指定目录下的所有 Markdown 文件
    
    参数:
        directory_path (str): 目录路径
        file_type (str): 要处理的文件类型，默认为 "md"
    """
    file_paths = get_absolute_file_paths(directory_path, file_type)
    logger.info(f"找到 {len(file_paths)} 个 {file_type} 文件")
    
    for md_file in file_paths:
        logger.info(f"处理文件: {md_file}")
        # 移除图片行
        remove_jpg_lines(md_file)
        # 解析并保存为 JSON
        save_path = md_file.split(".md")[0] + ".json"
        parsed_data = parse_md(md_file)
        saveData(parsed_data, save_path)
        logger.info(f"已保存到: {save_path}")