"""
Markdown 文件处理工具

该模块用于将 Markdown 格式的教材内容解析为结构化的 JSON 数据。
主要功能包括：
1. 提取 Markdown 文件中的标题和内容
2. 过滤掉图片链接和特定类型的标题（如章节、单元等）
3. 将处理后的内容保存为 JSON 格式
"""

# 命令行工具使用示例
'''
# 使用 magic-pdf 工具从 PDF 提取 Markdown
magic-pdf -p "/path/to/pdf/file.pdf" -o "/output/directory" -m auto

# 查看 magic-pdf 帮助
magic-pdf --help
'''
import os
import re
import json
import sys
from typing import List, Dict, Any

# 添加项目根目录到系统路径
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from utils import saveData

def get_absolute_file_paths(absolute_dir: str, file_type: str) -> List[str]:
    """
    获取指定目录下特定类型的文件的绝对路径列表
    
    参数:
        absolute_dir (str): 目录的绝对路径
        file_type (str): 文件类型，如 "md", "json" 等
        
    返回:
        List[str]: 文件绝对路径列表
    """
    return [os.path.join(absolute_dir, f) for f in os.listdir(absolute_dir) 
            if f.endswith(f".{file_type}")]

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

def process_directory(directory_path: str, file_type: str = "md") -> None:
    """
    处理指定目录下的所有 Markdown 文件
    
    参数:
        directory_path (str): 目录路径
        file_type (str): 要处理的文件类型，默认为 "md"
    """
    file_paths = get_absolute_file_paths(directory_path, file_type)
    print(f"找到 {len(file_paths)} 个 {file_type} 文件")
    
    for md_file in file_paths:
        print(f"处理文件: {md_file}")
        # 移除图片行
        remove_jpg_lines(md_file)
        # 解析并保存为 JSON
        save_path = md_file.split(".md")[0] + ".json"
        parsed_data = parse_md(md_file)
        saveData(parsed_data, save_path)
        print(f"已保存到: {save_path}")

def main():
    """主函数，程序入口点"""
    # 设置要处理的目录
    target_directory = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md'
    process_directory(target_directory, "md")

# 当脚本直接运行时执行 main 函数
if __name__ == "__main__":
    main()

