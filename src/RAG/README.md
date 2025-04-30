# RAG (Retrieval-Augmented Generation) 模块

本模块负责处理与检索增强生成相关的任务，包括数据处理和检索逻辑。

## 目录结构与文件说明
src/RAG/
├── 数据处理/
│   ├── 1.去除无用段落.py        # (推测功能) 脚本，用于初步去除文本数据中的无用段落。
│   ├── 4.最终合并.ipynb        # Jupyter Notebook，用于合并不同类型（如单选、多选、主观题）的处理后数据。
│   ├── MinerU.ipynb           # Jupyter Notebook，使用 magic_pdf 库从 PDF 文件提取内容，处理后可能保存为 Markdown 或 JSON。
│   ├── MinerU_不用大标题.py   # 脚本，解析 Markdown 文件，提取标题和内容，去除图片链接和特定类型的标题（如章节、单元），保存为 JSON。
│   ├── 去除content无用信息.py # 脚本，调用 API (call_Hkust_api) 清洗 JSON 文件中的 'content' 字段，去除练习题、小结等非知识性信息。
│   ├── 高中初始合并_给id.ipynb # Jupyter Notebook，合并指定目录下的 JSON 文件，为每条记录添加唯一 ID、学科和年级信息。
│   └── 高中去除无用信息.py    # 脚本，针对合并后的数据 ( merge.json )，多线程调用 API (call_Hkust_api) 清洗文本内容，去除无关信息，保存为 JSONL ( 去无用_merge.jsonl )。
├── retrieval.py             # Python 脚本，包含使用 Milvus 进行向量检索（密集、稀疏、混合）的函数，支持按学科过滤。
└── README.md                # 本说明文件。


## 主要功能

1.  **数据处理 (`数据处理/` 目录)**：
    *   从 PDF 课本提取原始文本和结构 (`MinerU.ipynb`)。
    *   将 Markdown 格式的内容解析为结构化的 JSON 数据 (`MinerU_不用大标题.py`)。
    *   合并初步处理后的数据并添加元信息 (`高中初始合并_给id.ipynb`)。
    *   通过 API 清洗和精炼文本内容，去除与核心知识无关的部分 (`去除content无用信息.py`, `高中去除无用信息.py`)。
    *   最终整合不同类型的数据 (`4.最终合并.ipynb`)。
2.  **检索 (`retrieval.py`)**：
    *   提供基于 Milvus 向量数据库的检索功能。
    *   支持密集向量检索 (`dense_search`)、稀疏向量检索 (`sparse_search`) 和混合检索 (`hybrid_search`)。
    *   混合检索支持根据 `subject` 字段进行过滤。

## 使用说明

*   **数据处理**：按照文件名中的数字顺序或根据 Notebook/脚本内的说明逐步执行，以完成数据的提取、转换、清洗和合并。请注意检查脚本和 Notebook 中的文件路径是否需要根据您的环境进行修改。
*   **检索**：`retrieval.py` 中的函数可以被其他模块导入和调用，以实现基于向量相似度的内容检索。确保 Milvus 服务正在运行并且集合（collection）已正确加载数据。
