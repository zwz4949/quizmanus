import re
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from src.graph.nodes.quiz_types import embeddings, reranker

def get_collection_minerU(context, uri, embedding_model, 
                            col_name="user_docs", 
                            chunk_size=800, 
                            overlap=200,
                            batch_size=50,
                            text_max_length=1024,
                            consistency_level="Strong"):
    """
    将文本内容处理并存入Milvus向量数据库
    
    Args:
        context (str): 文本内容
        uri (str): Milvus数据库URI
        embedding_model: 嵌入模型
        col_name (str): 集合名称
        chunk_size (int): 文本切片大小
        overlap (int): 切片重叠大小
        batch_size (int): 批量插入大小
        text_max_length (int): 文本字段最大长度
        consistency_level (str): 一致性级别，可选"Strong"或"Eventually"
        
    Returns:
        Collection: Milvus集合对象
    """
    # 连接到Milvus
    connections.connect(uri=uri)
    
    # 1. 文档切片
    documents = slice_document_by_heading(context)
    
    print(f"文档切片完成，共{len(documents)}个切片")
    
    # 2. 生成嵌入向量
    docs_embeddings = embedding_model(documents)
    
    # 3. 创建集合
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=text_max_length),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.dim["dense"]),
    ]
    
    schema = CollectionSchema(fields)
    
    # 如果集合已存在，则删除
    if utility.has_collection(col_name):
        Collection(col_name).drop()
    
    # 创建新集合
    col = Collection(col_name, schema, consistency_level=consistency_level)
    
    # 4. 创建索引
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    
    # 5. 批量插入数据
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batched_entities = [
            documents[i:end_idx],
            docs_embeddings["sparse"][i:end_idx],
            docs_embeddings["dense"][i:end_idx],
        ]
        col.insert(batched_entities)
        print("hello")
    
    print(f"数据插入完成，共{col.num_entities}条记录")
    
    # 加载集合以便搜索
    col.load()
    
    return col

def slice_document_by_heading(context):
    """
    根据单个#标题格式切片文档
    支持以下格式:
    1. "#" 开头的标题 (如 "# 标题")
    """
    # 预处理：去除每行开头的空白字符
    lines = context.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    context = '\n'.join(cleaned_lines)
    
    # 确保文档以换行符开始和结束，便于匹配
    context = "\n" + context.strip() + "\n"
    
    # 匹配单个#开头的标题格式
    pattern = r'(?:\n)(# .+?)(?:\n)(.+?)(?=\n# .+?|\Z)'
    matches = re.findall(pattern, context, re.DOTALL)
    
    # 如果没有匹配到任何标题，尝试按段落切分
    if not matches:
        print("未找到标题匹配，将按段落切分")
        return slice_by_paragraphs(context, max_length=800)
    
    documents = []
    for match in matches:
        title = match[0].strip()
        content = match[1].strip()
        # 过滤掉过短的内容
        if len(content) > 20:  # 可以调整最小内容长度
            documents.append(f"{title}\n{content}")
    
    return documents
# 使用示例
if __name__ == "__main__":
    from milvus_model.hybrid import BGEM3EmbeddingFunction
    
    with open('./PDF_context.txt', 'r+') as file:
            context = file.read()
    # 测试切片效果
    documents = slice_document_by_heading(context)
    
    print("切片结果:")
    for i, doc in enumerate(documents):
        print(f"\n--- 切片 {i+1} ---\n{doc}\n")
    
 
    
    # 创建集合并插入数据
    col = get_collection_minerU(
        context=context,
        uri="/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/RAG/vector_store/subjects.db",
        embedding_model=embeddings,
        text_max_length=4096,  # 增加文本长度限制
        batch_size=100         # 增加批处理大小
    )
    
    print(col)
    print('hello')