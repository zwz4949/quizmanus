"""
文件操作工具模块

提供文件路径获取、文件类型检测等通用功能
"""

import os
import re
from typing import List, Optional


def get_absolute_file_paths(absolute_dir: str, file_type: str) -> List[str]:
    """
    获取指定目录下特定类型的文件的绝对路径列表
    
    参数:
        absolute_dir (str): 目录的绝对路径
        file_type (str): 文件类型，如 "pdf", "md", "json" 等
        
    返回:
        List[str]: 文件绝对路径列表
    """
    return [os.path.join(absolute_dir, f) for f in os.listdir(absolute_dir) 
            if f.endswith(f".{file_type}")]


def is_file_path(path: str, file_type: str) -> bool:
    """
    判断路径是否为指定类型的文件
    
    参数:
        path (str): 文件路径
        file_type (str): 文件类型，如'pdf', 'jpg', 'png', 'ppt'
        
    返回:
        bool: 如果是指定类型的文件则返回True，否则返回False
    """
    if not path:
        return False
        
    # 检查文件后缀
    pattern = r'\.{}$'.format(file_type.lower())
    if re.search(pattern, path, re.IGNORECASE):
        # 检查文件是否存在
        if os.path.exists(path) and os.path.isfile(path):
            return True
    return False


def is_pdf_path(path: str) -> bool:
    """判断路径是否为PDF文件"""
    return is_file_path(path, 'pdf')


def is_image_path(path: str) -> bool:
    """判断路径是否为图片文件(JPG/PNG)"""
    return is_file_path(path, 'jpg') or is_file_path(path, 'png')


def is_ppt_path(path: str) -> bool:
    """判断路径是否为PPT文件"""
    return is_file_path(path, 'ppt') or is_file_path(path, 'pptx')


def detect_file_type(path: str) -> Optional[str]:
    """
    检测文件类型
    
    参数:
        path (str): 文件路径
        
    返回:
        Optional[str]: 文件类型，如'pdf', 'image', 'ppt'，如果不是支持的类型则返回None
    """
    if is_pdf_path(path):
        return 'pdf'
    elif is_image_path(path):
        return 'image'
    elif is_ppt_path(path):
        return 'ppt'
    return None