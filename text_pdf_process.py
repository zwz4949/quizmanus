"""
测试MinerU处理器的PDF处理功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MinerU处理器
from src.graph.MinerU.core.processor import MinerUProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_processing():
    """测试PDF处理功能并将结果保存到文本文件"""
    
    # PDF文件路径
    pdf_path = "/hpc2hdd/home/fye374/LJJ/PDF/高途高考基础2000题-政治伴学讲义.pdf"
    
    # 输出文件路径
    output_path = "PDF_context.txt"
    
    logger.info(f"开始处理PDF文件: {pdf_path}")
    
    try:
        # 初始化处理器
        processor = MinerUProcessor()
        
        # 处理PDF文件
        result = processor.process_file(pdf_path)
        
        # 提取处理后的内容
        processed_content = result.get("processed_content", "")
        
        if not processed_content:
            logger.warning("处理结果中没有找到内容")
            return
        
        # 将内容写入文本文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_content)
        
        logger.info(f"PDF内容已成功写入文件: {output_path}")
        logger.info(f"内容长度: {len(processed_content)} 字符")
        
        # 打印内容的前200个字符作为预览
        preview = processed_content[:200].replace("\n", " ")
        logger.info(f"内容预览: {preview}...")
        
    except Exception as e:
        logger.error(f"处理PDF文件时出错: {e}", exc_info=True)

if __name__ == "__main__":
    test_pdf_processing()