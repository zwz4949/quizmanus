"""
MinerU命令行工具

提供命令行接口，用于处理文件和批量处理
"""

import os
import sys
import argparse
import logging
from typing import List

# 导入相关模块
from ..converters.pdf_converter import process_pdf, batch_process_pdfs
from ..converters.md_converter import process_directory
from ..core.processor import MinerUProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数，命令行入口点"""
    parser = argparse.ArgumentParser(description='MinerU文件处理工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # PDF处理命令
    pdf_parser = subparsers.add_parser('pdf', help='处理PDF文件')
    pdf_parser.add_argument('--path', type=str, required=True, help='PDF文件路径或目录')
    pdf_parser.add_argument('--batch', action='store_true', help='批量处理目录下的所有PDF')
    pdf_parser.add_argument('--keep-images', action='store_true', help='保留图片')
    
    # Markdown处理命令
    md_parser = subparsers.add_parser('md', help='处理Markdown文件')
    md_parser.add_argument('--dir', type=str, required=True, help='Markdown文件目录')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.command == 'pdf':
        if args.batch:
            if os.path.isdir(args.path):
                batch_process_pdfs([args.path])
            else:
                logger.error(f"指定的路径不是目录: {args.path}")
        else:
            if os.path.isfile(args.path):
                process_pdf(args.path, remain_image=args.keep_images)
            else:
                logger.error(f"指定的路径不是文件: {args.path}")
    
    elif args.command == 'md':
        if os.path.isdir(args.dir):
            process_directory(args.dir)
        else:
            logger.error(f"指定的路径不是目录: {args.dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()