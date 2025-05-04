"""
MinerU节点模块

该模块提供了处理PDF文件并转换为结构化数据的节点函数。
主要功能包括：
1. 判断输入是否为PDF文件路径
2. 处理PDF文件并转换为Markdown格式
3. 将Markdown文件解析为JSON格式
4. 支持自定义知识库的处理
"""

import os
import re
import sys
import json
from typing import List, Dict, Any, Literal, Optional
from langgraph.types import Command

# 导入State类型
from .quiz_types import State

# 导入MinerU处理器
from ..MinerU.core.processor import MinerUProcessor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)   

# 创建MinerU处理器实例
miner_processor = MinerUProcessor()

def miner_router(state: State) -> Command[Literal["pdf_processor", "image_processor", "ppt_processor", "coordinator"]]:
    """
    路由节点，判断是否需要启用MinerU处理或自定义知识库处理
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU router checking input")
    print(f'State: miner_router')

    # 检查是否有自定义知识库
    custom_kb = state.get("custom_knowledge_base", None)
    if custom_kb:
        kb_type = custom_kb.get("type")
        if kb_type == "pdf":
            logger.info("检测到PDF自定义知识库")
            return Command(goto="pdf_processor")
        elif kb_type == "image":
            logger.info("检测到图片自定义知识库")
            return Command(goto="image_processor")
        elif kb_type == "ppt":
            logger.info("检测到PPT自定义知识库")
            return Command(goto="ppt_processor")
    
    # 其他情况
    logger.info("没有检测到自定义知识库，继续正常流程")
    return Command(goto="coordinator")

def pdf_processor(state: State) -> Command[Literal["coordinator"]]:
    """
    处理PDF文件并转换为结构化数据
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU PDF处理器启动")
    pdf_path = state["custom_knowledge_base"]['path']
    print(f'State: pdf_processor: {pdf_path}')

    try:
        embedding_model=state["rag"]["embedding_model"]
        # 使用MinerU处理器处理PDF
        result = miner_processor.process_custom_kb({"type": "pdf", "path": pdf_path},embedding_model)
        
        logger.info("PDF处理成功完成")

        return Command(
            update={
                "ori_query": f"以下是从PDF文件提取的内容，请基于这些内容生成题目：\n\n{result['processed_content']}",
                "custom_knowledge_base": result,
                "use_custom_kb": True  # 标记使用自定义知识库
            },
            goto="coordinator"
        )
    except Exception as e:
        logger.error(f"处理PDF文件时出错: {e}")
        return Command(
            update={
                "ori_query": f"无法处理PDF文件 {pdf_path}，错误信息：{str(e)}。请提供其他查询。"
            },
            goto="coordinator"
        )

def image_processor(state: State) -> Command[Literal["coordinator"]]:
    """
    处理图片文件并转换为结构化数据
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU 图片处理器启动")
    image_path = state["custom_knowledge_base"]['path']
    
    try:
        # 使用MinerU处理器处理图片
        result = miner_processor.process_custom_kb({"type": "image", "path": image_path})
        
        logger.info("图片处理成功完成")
        
        return Command(
            update={
                "ori_query": f"以下是从图片文件提取的内容，请基于这些内容生成题目：\n\n{result['processed_content']}",
                "custom_knowledge_base": result,
                "use_custom_kb": True
            },
            goto="coordinator"
        )
    except Exception as e:
        logger.error(f"处理图片文件时出错: {e}")
        return Command(
            update={
                "ori_query": f"无法处理图片文件 {image_path}，错误信息：{str(e)}。请提供其他查询。"
            },
            goto="coordinator"
        )

def ppt_processor(state: State) -> Command[Literal["coordinator"]]:
    """
    处理PPT文件并转换为结构化数据
    
    参数:
        state (State): 当前状态
        
    返回:
        Command: 下一步操作的命令
    """
    logger.info("MinerU PPT处理器启动")
    ppt_path = state["custom_knowledge_base"]['path']
    
    try:
        # 使用MinerU处理器处理PPT
        result = miner_processor.process_custom_kb({"type": "ppt", "path": ppt_path})
        
        logger.info("PPT处理成功完成")
        
        return Command(
            update={
                "ori_query": f"以下是从PPT文件提取的内容，请基于这些内容生成题目：\n\n{result['processed_content']}",
                "custom_knowledge_base": result,
                "use_custom_kb": True
            },
            goto="coordinator"
        )
    except Exception as e:
        logger.error(f"处理PPT文件时出错: {e}")
        return Command(
            update={
                "ori_query": f"无法处理PPT文件 {ppt_path}，错误信息：{str(e)}。请提供其他查询。"
            },
            goto="coordinator"
        )