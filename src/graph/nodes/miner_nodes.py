"""
MinerU节点模块

该模块提供了处理PDF文件并转换为结构化数据的节点函数。
主要功能包括：
1. 判断输入是否为PDF文件路径
2. 处理PDF文件并转换为Markdown格式
3. 将Markdown文件解析为JSON格式
"""

import os
import re
import sys
from typing import List, Dict, Any, Literal
from langgraph.types import Command

# 导入State类型
from .quiz_types import State

# 导入MinerU相关功能
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from RAG.data_process_process.MinerU import process_pdf, get_absolute_file_paths
from RAG.data_process_process.md2json_MinerU import parse_md, remove_jpg_lines

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def is_pdf_path(query: str) -> bool:
    """
    判断查询是否为PDF文件路径
    
    参数:
        query (str): 用户查询
        
    返回:
        bool: 如果查询是PDF文件路径则返回True，否则返回False
    """
    # 检查是否包含.pdf后缀
    if re.search(r'\.pdf$', query, re.IGNORECASE):
        # 检查文件是否存在
        if os.path.exists(query) and os.path.isfile(query):
            return True
    return False

def miner_router(state: State) -> Command[Literal["miner_processor", "coordinator"]]:
    """
    路由节点，判断是否需要启用MinerU处理
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU router checking input")
    query = state["ori_query"]
    
    if is_pdf_path(query):
        logger.info(f"Detected PDF file: {query}, routing to MinerU processor")
        return Command(goto="miner_processor")
    else:
        logger.info("Input is not a PDF file, proceeding with normal flow")
        return Command(goto="coordinator")

def miner_processor(state: State) -> Command[Literal["coordinator"]]:
    """
    处理PDF文件并转换为结构化数据
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU processor starting")
    pdf_path = state["ori_query"]
    
    try:
        # 处理PDF文件并转换为Markdown
        md_path = process_pdf(pdf_path, remain_image=False)
        logger.info(f"PDF converted to Markdown: {md_path}")
        
        # 移除图片行
        remove_jpg_lines(md_path)
        
        # 解析Markdown文件
        parsed_data = parse_md(md_path)
        
        # 更新查询内容为解析后的数据
        processed_content = "\n\n".join([f"# {item['title']}\n{item['content']}" for item in parsed_data])
        
        logger.info("PDF processing completed successfully")
        
        return Command(
            update={
                "ori_query": f"以下是从PDF文件提取的内容，请基于这些内容生成题目：\n\n{processed_content}"
            },
            goto="coordinator"
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return Command(
            update={
                "ori_query": f"无法处理PDF文件 {pdf_path}，错误信息：{str(e)}。请提供其他查询。"
            },
            goto="coordinator"
        )