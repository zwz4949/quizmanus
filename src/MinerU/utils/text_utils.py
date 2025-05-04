"""
文本处理工具模块

提供文本清理、格式转换等功能
"""

import re
from typing import List, Dict


def remove_jpg_lines(md_file_path: str) -> None:
    """
    读取 Markdown 文件并删除包含 .jpg 图片的行
    
    参数:
        md_file_path (str): Markdown 文件的路径
    """
    with open(md_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 使用正则表达式匹配包含 .jpg 的行
    pattern = re.compile(r'.*\.jpg.*')
    new_lines = [line for line in lines if not pattern.search(line)]
    
    output_content = ''.join(new_lines)

    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.write(output_content)


def parse_md(file_path: str) -> List[Dict[str, str]]:
    """
    解析 Markdown 文件，提取标题和内容
    
    参数:
        file_path (str): Markdown 文件路径
        
    返回:
        List[Dict[str, str]]: 包含标题和内容的字典列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    result = []
    current_title = None
    current_content = []
    
    # 需要过滤的标题关键词
    filter_keywords = ["第", "章", "单元", "节", "上册", "下册", "目录"]
    
    for line in lines:
        if line.startswith('#'):
            # 处理前一个标题的内容
            if current_title is not None:
                content = '\n'.join(current_content)
                # 过滤掉不需要的标题和空内容
                if (content.strip() != "" and 
                    not any(keyword in current_title for keyword in filter_keywords)):
                    result.append({
                        "title": current_title.strip(), 
                        "content": content
                    })
            current_title = line
            current_content = []
        else:
            if current_title is not None:
                current_content.append(line)
    
    # 处理最后一个标题的内容
    if current_title is not None:
        content = '\n'.join(current_content)
        if (content.strip() != "" and 
            not any(keyword in current_title for keyword in filter_keywords)):
            result.append({
                "title": current_title.strip(), 
                "content": content
            })
    
    return result