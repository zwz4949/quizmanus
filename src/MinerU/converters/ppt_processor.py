"""
PPT处理模块

提供PPT文件的处理功能，将PPT文件转换为结构化数据，并可选择保存为Markdown文件
"""

import logging
import os
from typing import Dict, Any, List
from magic_doc.pdf_transform import S3Config, DocConverter, ParsePDFType
from magic_doc.progress.filepupdator import FileBaseProgressUpdator

# 导入工具函数
from ..utils.text_utils import remove_jpg_lines, parse_md

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def process_ppt(ppt_path: str, remain_image: bool = False) -> str:
    """
    处理PPT文件并转换为Markdown格式
    
    参数:
        ppt_path (str): PPT文件路径
        remain_image (bool): 是否保留图片，默认为False
        
    返回:
        str: 生成的Markdown文件路径
    """
    logger.info(f"开始处理PPT文件: {ppt_path}")
    
    # 检查文件是否存在
    if not os.path.exists(ppt_path):
        logger.error(f"PPT文件不存在: {ppt_path}")
        raise FileNotFoundError(f"PPT文件不存在: {ppt_path}")
    
    # 提取文件名（不含扩展名）
    if remain_image:
        name_without_suffix = os.path.splitext(ppt_path)[0]
        # 设置输出目录
        local_image_dir = "output/ppt/images"
        local_md_dir = "output/ppt/md"
        output_md_path = f"{name_without_suffix}.md"
    else:
        name_without_suffix = os.path.splitext(os.path.basename(ppt_path))[0]
        # 设置输出目录
        local_image_dir = "output/ppt/images"
        local_md_dir = "output/ppt/md"
        output_md_path = f"{local_md_dir}/{name_without_suffix}.md"
    
    # 获取图片目录名
    image_dir = os.path.basename(local_image_dir)
    
    # 确保目录存在
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    
    # 初始化DocConverter
    s3_config = S3Config(ak="", sk="", endpoint="")  
    converter = DocConverter(s3_config, parse_pdf_type=ParsePDFType.FULL)
    
    # 设置进度文件路径
    progress_file_path = os.path.join(os.path.dirname(output_md_path), "progress.txt")
    
    os.makedirs(os.path.dirname(progress_file_path), exist_ok=True)
    
    # 转换PPT文件为Markdown
    markdown_content, time_cost = converter.convert(ppt_path, progress_file_path)
    
    # 保存Markdown内容到文件
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"完成处理PPT文件: {ppt_path}, 耗时: {time_cost}秒")
    return output_md_path


def process_ppt_to_structured(ppt_path: str, remain_image: bool = False, save_output: bool = True) -> Dict[str, Any]:
    """
    处理PPT文件并转换为结构化数据
    
    参数:
        ppt_path (str): PPT文件路径
        remain_image (bool): 是否保留图片
        save_output (bool): 是否保存输出文件
        
    返回:
        Dict[str, Any]: 处理结果，包含processed_data和processed_content
    """
    logger.info(f"开始处理PPT文件: {ppt_path}")
    
    try:
        # 处理PPT文件并转换为Markdown
        md_path = process_ppt(ppt_path, remain_image=remain_image)
        logger.info(f"PPT转换为Markdown: {md_path}")
        
        # 移除图片行
        remove_jpg_lines(md_path)
            
        # 解析Markdown文件
        parsed_data = parse_md(md_path)
            
        # 生成处理后的内容
        processed_content = "\n\n".join([f"# {item['title']}\n{item['content']}" for item in parsed_data])
        
        # 如果不需要保存输出文件，则删除生成的Markdown文件
        if not save_output and os.path.exists(md_path):
            os.remove(md_path)
            logger.info(f"已删除临时Markdown文件: {md_path}")
        
        # 返回处理结果
        result = {
            "type": "ppt",
            "path": ppt_path,
            "processed_data": parsed_data,
            "processed_content": processed_content,
            "collection_id": os.path.basename(os.path.splitext(ppt_path)[0])  # 只使用文件名，不包含扩展名
        }
        
        logger.info("PPT处理成功完成")
        return result
        
    except Exception as e:
        logger.error(f"处理PPT文件时出错: {e}")
        return {
            "type": "ppt",
            "path": ppt_path,
            "error": f"处理PPT文件时出错: {str(e)}",
            "processed_content": f"[PPT内容处理失败: {ppt_path}]",
            "collection_id": os.path.basename(ppt_path)
        }


def batch_process_ppts(subject_dirs: list) -> None:
    """
    批量处理多个目录下的PPT文件
    
    参数:
        subject_dirs (List[str]): 包含PPT文件的目录列表
    """
    from ..utils.file_utils import get_absolute_file_paths
    
    for subject_dir in subject_dirs:
        logger.info(f"处理目录: {subject_dir}")
        ppt_paths = get_absolute_file_paths(subject_dir, ["ppt", "pptx"])
        logger.info(f"找到 {len(ppt_paths)} 个PPT文件")
        
        for ppt_path in ppt_paths:
            process_ppt(ppt_path)
