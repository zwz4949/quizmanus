"""
MinerU处理器核心模块

提供统一的文件处理接口
"""

import logging
from typing import Dict, Any, Optional

# 导入工具函数
from ..utils.file_utils import detect_file_type
from ..converters.pdf_converter import process_pdf_to_structured
from ..converters.image_processor import process_image
from ..converters.ppt_processor import process_ppt

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MinerUProcessor:
    """
    MinerU处理器类
    
    用于处理不同类型的文件，包括PDF、图片和PPT等，
    并将处理结果转换为结构化数据。
    """
    
    def __init__(self):
        """初始化MinerU处理器"""
        logger.info("MinerU处理器初始化")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理文件，自动检测文件类型并调用相应的处理函数
        
        参数:
            file_path (str): 文件路径
            
        返回:
            Dict[str, Any]: 处理结果
        """
        file_type = detect_file_type(file_path)
        
        if file_type == 'pdf':
            return process_pdf_to_structured(file_path)
        elif file_type == 'image':
            return process_image(file_path)
        elif file_type == 'ppt':
            return process_ppt(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
    
    def process_custom_kb(self, custom_kb: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理自定义知识库
        
        参数:
            custom_kb (Dict[str, Any]): 自定义知识库信息
            
        返回:
            Dict[str, Any]: 处理结果
        """
        kb_type = custom_kb.get("type")
        kb_path = custom_kb.get("path")
        
        if not kb_type or not kb_path:
            raise ValueError("自定义知识库缺少类型或路径信息")
        
        if kb_type == "pdf":
            return process_pdf_to_structured(kb_path)
        elif kb_type == "image":
            return process_image(kb_path)
        elif kb_type == "ppt":
            return process_ppt(kb_path)
        else:
            raise ValueError(f"不支持的知识库类型: {kb_type}")