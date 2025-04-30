"""
PDF 文件处理工具

该模块使用 magic_pdf 库从 PDF 文件中提取内容并转换为 Markdown 格式。
主要功能包括：
1. 自动判断 PDF 是否需要 OCR 处理
2. 提取 PDF 中的文本和结构
3. 可选择是否保留图片
4. 将处理后的内容保存为 Markdown 文件
"""

import os
from typing import List

# 设置 GPU 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 导入 magic_pdf 相关库
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


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


def process_pdf(pdf_file_path: str, remain_image: bool = False) -> str:
    """
    处理 PDF 文件并转换为 Markdown 格式
    
    参数:
        pdf_file_path (str): PDF 文件路径
        remain_image (bool): 是否保留图片，默认为 False
        
    返回:
        str: 生成的 Markdown 文件路径
    """
    print(f"开始处理: {pdf_file_path}")
    
    # 提取文件名（不含扩展名）
    if remain_image:
        name_without_suffix = pdf_file_path.split(".")[0]
        # 设置输出目录
        local_image_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/images"
        local_md_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md"
        output_md_path = f"{name_without_suffix}.md"
    else:
        name_without_suffix = pdf_file_path.split(".")[0].split('/')[-1]
        # 设置输出目录
        local_image_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/md/images"
        local_md_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/md"
        output_md_path = f"{local_md_dir}/{name_without_suffix}.md"
    
    # 获取图片目录名
    image_dir = os.path.basename(local_image_dir)
    
    # 确保目录存在
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    
    # 创建数据写入器
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)
    
    # 读取 PDF 文件内容
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_file_path)
    
    # 创建数据集实例
    dataset = PymuDocDataset(pdf_bytes)
    
    # 根据 PDF 类型选择处理方式
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        # 需要 OCR 处理
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        # 不需要 OCR 处理
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    
    # 保存为 Markdown 文件
    pipe_result.dump_md(md_writer, output_md_path, image_dir)
    
    print(f"完成处理: {pdf_file_path}")
    return output_md_path


def batch_process_pdfs(subject_dirs: List[str]) -> None:
    """
    批量处理多个目录下的 PDF 文件
    
    参数:
        subject_dirs (List[str]): 包含 PDF 文件的目录列表
    """
    for subject_dir in subject_dirs:
        print(f"处理目录: {subject_dir}")
        pdf_paths = get_absolute_file_paths(subject_dir, "pdf")
        print(f"找到 {len(pdf_paths)} 个 PDF 文件")
        
        for pdf_path in pdf_paths:
            process_pdf(pdf_path)


def main():
    """主函数，程序入口点"""
    # 高中学科目录列表
    subject_dirs = [
        '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中政治',
        '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中生物',
        '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中历史',
        '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中地理'
    ]
    
    batch_process_pdfs(subject_dirs)


# 当脚本直接运行时执行 main 函数
if __name__ == "__main__":
    main()