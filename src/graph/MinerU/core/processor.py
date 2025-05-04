"""
MinerU处理器核心模块

提供统一的文件处理接口，支持向量存储缓存
"""

import os
import logging
from typing import Dict, Any, Optional

# 导入工具函数
from ..utils.file_utils import detect_file_type
from ..converters.pdf_converter import process_pdf_to_structured
from ..converters.image_processor import process_image
from ..converters.ppt_processor import process_ppt

# 导入向量存储相关模块
from ....RAG.vector_store_utils import get_collection_minerU
from ....config.rag import DB_URI,TEMP_DB_URI

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MinerUProcessor:
    """
    MinerU处理器类
    
    用于处理不同类型的文件，包括PDF、图片和PPT等，
    并将处理结果转换为结构化数据。
    支持向量存储缓存，避免重复生成嵌入向量。
    """
    
    def __init__(self):
        """初始化MinerU处理器"""
        logger.info("MinerU处理器初始化")
        self.collections_cache = {}  # 用于缓存已创建的集合
    
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        处理文件，自动检测文件类型并调用相应的处理函数
        
        参数:
            file_path (str): 文件路径
            **kwargs: 额外参数
            
        返回:
            Dict[str, Any]: 处理结果
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {"error": "文件不存在", "path": file_path}
            
        file_type = detect_file_type(file_path)
        
        if file_type == 'pdf':
            return self.process_pdf(file_path, **kwargs)
        elif file_type == 'image':
            return process_image(file_path)
        elif file_type == 'ppt':
            return process_ppt(file_path)
        elif file_type == 'md':
            return self.process_markdown(file_path, **kwargs)
        else:
            logger.error(f"不支持的文件类型: {file_path}")
            return {
                "error": f"不支持的文件类型: {file_path}",
                "path": file_path,
                "type": "unknown"
            }
    
    def process_pdf(self, pdf_path: str, remain_image: bool = False, structured: bool = True) -> Dict[str, Any]:
        """
        处理PDF文件
        
        参数:
            pdf_path (str): PDF文件路径
            remain_image (bool): 是否保留图片
            structured (bool): 是否返回结构化数据
            
        返回:
            Dict[str, Any]: 处理结果
        """
        try:
            if structured:
                # 使用结构化处理
                result = process_pdf_to_structured(pdf_path, remain_image)
                return result
        except Exception as e:
            logger.error(f"处理PDF文件失败: {e}")
            return {
                "error": f"处理PDF文件失败: {str(e)}",
                "path": pdf_path,
                "type": "pdf",
                "collection_id": os.path.basename(pdf_path)
            }
    
    def process_markdown(self, md_path: str, **kwargs) -> Dict[str, Any]:
        """
        处理Markdown文件
        
        参数:
            md_path (str): Markdown文件路径
            **kwargs: 额外参数
            
        返回:
            Dict[str, Any]: 处理结果
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {
                "type": "markdown",
                "path": md_path,
                "processed_content": content,
                "collection_id": os.path.basename(md_path)
            }
            return result
        except Exception as e:
            logger.error(f"处理Markdown文件失败: {e}")
            return {
                "error": f"处理Markdown文件失败: {str(e)}",
                "path": md_path,
                "type": "markdown",
                "collection_id": os.path.basename(md_path)
            }
    
    def process_custom_kb(self, custom_kb: Dict[str, Any], embedding_model=None) -> Dict[str, Any]:
        """
        处理自定义知识库并创建向量存储
        
        参数:
            custom_kb (Dict[str, Any]): 自定义知识库信息
            embedding_model: 嵌入模型，用于创建向量存储
            
        返回:
            Dict[str, Any]: 处理结果，包含向量存储集合
        """
        kb_type = custom_kb.get("type", "unknown")
        kb_path = custom_kb.get("path")
        
        if not kb_path or not os.path.exists(kb_path):
            logger.error("自定义知识库路径无效")
            return {"error": "自定义知识库路径无效"}
        # 根据类型处理知识库
        if kb_type == "pdf":
            # 处理PDF
            result = self.process_pdf(kb_path, structured=True)
            if "error" in result:
                return result
 
            # 检查是否已有缓存的集合
            collection_id = result["collection_id"]
            if collection_id in self.collections_cache:
                logger.info(f"使用缓存的集合: {collection_id}")
                result["collection"] = self.collections_cache[collection_id]
            elif embedding_model:  # 直接检查嵌入模型
                # 创建集合并缓存
                logger.info(f"为知识库创建新的向量存储: {collection_id}")
                
                # 处理collection_id，确保只包含数字、字母和下划线
                import re
                import hashlib
                
                # 移除文件扩展名
                collection_name = os.path.splitext(collection_id)[0]
                
                # 替换非法字符为下划线
                safe_collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
                
                # 如果包含中文或其他非ASCII字符，使用哈希值
                if not all(ord(c) < 128 for c in collection_name):
                    # 创建哈希值并保留部分原始名称作为前缀
                    hash_suffix = hashlib.md5(collection_name.encode()).hexdigest()[:8]
                    # 取原始名称的前10个字符（如果有）作为前缀，替换非法字符
                    prefix = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name[:10])
                    safe_collection_name = f"{prefix}_{hash_suffix}"
                
                # 确保名称不以数字开头（某些数据库要求）
                if safe_collection_name[0].isdigit():
                    safe_collection_name = f"kb_{safe_collection_name}"
                
                # 限制名称长度
                if len(safe_collection_name) > 50:
                    safe_collection_name = safe_collection_name[:50]
                
                col = get_collection_minerU(
                    context=result["processed_content"],
                    uri=TEMP_DB_URI,
                    embedding_model=embedding_model,  # 直接使用传入的嵌入模型
                    col_name=f"user_kb_{safe_collection_name}",  # 使用安全的集合名称
                    text_max_length=4096,
                    batch_size=100
                )
                
                # 缓存集合
                if col:
                    self.collections_cache[collection_id] = col
                    result["collection"] = col
            
            return result
        elif kb_type == "image":
            return process_image(kb_path)
        elif kb_type == "ppt":
            return process_ppt(kb_path)
        elif kb_type == "markdown" or kb_type == "md":
            return self.process_markdown(kb_path)
        else:
            logger.error(f"不支持的知识库类型: {kb_type}")
            return {"error": f"不支持的知识库类型: {kb_type}"}
