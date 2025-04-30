# MinerU 模块文档

## 项目概述

MinerU 是一个强大的文件处理工具包，专门设计用于从各种格式的文件（如PDF、图片、PPT等）中提取和结构化内容。该模块作为 quizmanus 项目的一部分，主要用于将不同类型的教育资源转换为可用于生成题目的结构化数据。

## 目录结构

```
MinerU/
├── __init__.py                 # 包初始化文件
├── core/                       # 核心功能
│   ├── __init__.py
│   └── processor.py            # MinerUProcessor类，提供统一处理接口
├── converters/                 # 各种格式转换器
│   ├── __init__.py
│   ├── pdf_converter.py        # PDF相关转换功能
│   ├── md_converter.py         # Markdown相关转换功能
│   ├── image_processor.py      # 图片处理功能
│   └── ppt_processor.py        # PPT处理功能
├── utils/                      # 通用工具函数
│   ├── __init__.py
│   ├── file_utils.py           # 文件操作相关工具
│   └── text_utils.py           # 文本处理相关工具
└── cli/                        # 命令行工具
    ├── __init__.py
    └── commands.py             # 命令行入口点
```

## 模块详细说明

### 1. core 模块

#### processor.py

`MinerUProcessor` 类是整个模块的核心，提供了统一的文件处理接口。

**主要功能**：
- 自动检测文件类型并调用相应的处理函数
- 处理自定义知识库
- 支持PDF、图片和PPT等多种文件格式

**示例**：
```python
from MinerU.core.processor import MinerUProcessor

# 初始化处理器
processor = MinerUProcessor()

# 处理单个文件
result = processor.process_file("path/to/file.pdf")
print(result["processed_content"])

# 处理自定义知识库
custom_kb = {
    "type": "pdf",
    "path": "path/to/knowledge_base.pdf"
}
kb_result = processor.process_custom_kb(custom_kb)
```

**输入**：文件路径或自定义知识库信息
**输出**：包含处理结果的字典，通常包含以下字段：
```python
{
    "type": "pdf",  # 文件类型
    "path": "path/to/file.pdf",  # 文件路径
    "processed_data": [...],  # 结构化数据
    "processed_content": "提取的文本内容"  # 处理后的文本内容
}
```

### 2. converters 模块

#### pdf_converter.py

提供PDF文件处理功能，使用magic_pdf库从PDF文件中提取内容并转换为Markdown格式。

**主要功能**：
- 自动判断PDF是否需要OCR处理
- 提取PDF中的文本和结构
- 可选择是否保留图片
- 将处理后的内容保存为Markdown文件
- 将Markdown转换为结构化数据

**示例**：
```python
from MinerU.converters.pdf_converter import process_pdf_to_structured

# 处理PDF文件
result = process_pdf_to_structured("path/to/file.pdf")
print(result["processed_content"])

# 批量处理PDF文件
from MinerU.converters.pdf_converter import batch_process_pdfs
batch_process_pdfs(["path/to/directory"])
```

**输入**：PDF文件路径
**输出**：包含处理结果的字典，结构如下：
```python
{
    "type": "pdf",
    "path": "path/to/file.pdf",
    "processed_data": [
        {"title": "章节标题1", "content": "章节内容..."},
        {"title": "章节标题2", "content": "章节内容..."},
        # ...
    ],
    "processed_content": "# 章节标题1\n章节内容...\n\n# 章节标题2\n章节内容..."
}
```

#### md_converter.py

提供Markdown文件处理功能，将Markdown格式的教材内容解析为结构化的JSON数据。

**主要功能**：
- 提取Markdown文件中的标题和内容
- 过滤掉图片链接和特定类型的标题（如章节、单元等）
- 将处理后的内容保存为JSON格式

**示例**：
```python
from MinerU.converters.md_converter import process_directory

# 处理目录下的所有Markdown文件
process_directory("path/to/md_directory")
```

**输入**：Markdown文件目录
**输出**：为每个Markdown文件生成对应的JSON文件，包含标题和内容的结构化数据

#### image_processor.py

提供图片文件处理功能。

**示例**：
```python
from MinerU.converters.image_processor import process_image

# 处理图片文件
result = process_image("path/to/image.jpg")
print(result["processed_content"])
```

**输入**：图片文件路径
**输出**：包含处理结果的字典

#### ppt_processor.py

提供PPT文件处理功能。

**示例**：
```python
from MinerU.converters.ppt_processor import process_ppt

# 处理PPT文件
result = process_ppt("path/to/presentation.ppt")
print(result["processed_content"])
```

**输入**：PPT文件路径
**输出**：包含处理结果的字典

### 3. utils 模块

#### file_utils.py

提供文件路径获取、文件类型检测等通用功能。

**主要功能**：
- 获取指定目录下特定类型的文件的绝对路径列表
- 判断路径是否为指定类型的文件
- 检测文件类型

**示例**：
```python
from MinerU.utils.file_utils import get_absolute_file_paths, detect_file_type

# 获取目录下所有PDF文件
pdf_files = get_absolute_file_paths("path/to/directory", "pdf")

# 检测文件类型
file_type = detect_file_type("path/to/file")
print(f"文件类型: {file_type}")  # 输出: 文件类型: pdf
```

#### text_utils.py

提供文本清理、格式转换等功能。

**主要功能**：
- 读取Markdown文件并删除包含.jpg图片的行
- 解析Markdown文件，提取标题和内容

**示例**：
```python
from MinerU.utils.text_utils import remove_jpg_lines, parse_md

# 移除Markdown文件中的图片行
remove_jpg_lines("path/to/file.md")

# 解析Markdown文件
parsed_data = parse_md("path/to/file.md")
print(f"解析到{len(parsed_data)}个章节")
```

### 4. cli 模块

#### commands.py

提供命令行接口，用于处理文件和批量处理。

**主要功能**：
- 提供PDF处理命令
- 提供Markdown处理命令
- 支持批量处理和单文件处理

**示例**：
```bash
# 处理单个PDF文件
python -m MinerU.cli.commands pdf --path path/to/file.pdf

# 批量处理目录下的所有PDF文件
python -m MinerU.cli.commands pdf --path path/to/directory --batch

# 处理目录下的所有Markdown文件
python -m MinerU.cli.commands md --dir path/to/md_directory
```

## 与图谱系统集成

MinerU模块已经与quizmanus项目的图谱系统集成，通过`miner_nodes.py`中的节点函数提供服务。

### 集成流程

1. `miner_router`节点判断是否需要启用MinerU处理或自定义知识库处理
2. 根据文件类型，路由到相应的处理器节点（`pdf_processor`、`image_processor`或`ppt_processor`）
3. 处理器节点调用MinerU处理器处理文件，并将结果更新到状态中
4. 处理完成后，流程继续到`coordinator`节点
