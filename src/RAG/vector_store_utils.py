from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from typing import List,Dict
from tqdm import tqdm
import re

def create_collection(db_uri,col_name, dense_dim):
    '''
    dense_dim的获取示例：
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    ef = BGEM3EmbeddingFunction(
        model_name = "/hpc2hdd/home/fye374/models/BAAI/bge-m3",
        use_fp16=False, 
        device="cuda:0")
    dense_dim = ef.dim["dense"]
    '''
    connections.connect(uri=db_uri)
    # col_name = "subjects"
    if not utility.has_collection(col_name):
        fields = [
            # Use auto generated id as primary key
            FieldSchema(
                name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            # Store the original text to retrieve based on semantically distance
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="grade", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
            # Milvus now supports both sparse and dense vectors,
            # we can store each in a separate field to conduct hybrid search on both vectors
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        ]
        schema = CollectionSchema(fields)

        
        # if utility.has_collection(col_name):
        #     Collection(col_name).drop()
        col = Collection(col_name, schema, consistency_level="Strong")
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
        print(f"collection {col_name} 创建成功，并返回它")
        col.load()
    else:
        print(f"collection {col_name} already exists, 直接返回已有{col_name}")
        col = Collection(col_name)
    return col
def get_collection(db_uri, col_name, alias="default"):
    try:
        # 如果已有同名连接，先断开
        if connections.has_connection(alias):
            connections.disconnect(alias)
            print(f"已断开现有的 {alias} 连接")
        
        # 创建新连接
        connections.connect(uri=db_uri, alias=alias)
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        return None
        
    try:
        col = Collection(col_name, using=alias)
    except:
        print(f"collection {col_name} not found, 请创建collection")
        return None
    col.load(using=alias)
    return col


def add_data(col, data,docs_embeddings):
    '''
    data:List
    docs_embeddings:Dict[List]
    data:
    "subject": data[i]['subject'],  # 你的subject字段数据
    "grade": data[i]['grade'],  # 你的grade字段数据
    "title": data[i]['title'],
    "content": data[i]['content'],  # 你的text字段数据

    docs_embeddings:
    "sparse_vector": docs_embeddings["sparse"]._getrow(i).todok(),  # 转换为字典格式
    "dense_vector": docs_embeddings["dense"][i]
    '''
    try:
        for i in tqdm(range(len(data))):
            tmp = {
                "subject": data[i]['subject'],  # 你的subject字段数据
                "grade": data[i]['grade'],  # 你的grade字段数据
                "title": data[i]['title'],
                "content": data[i]['content'],  # 你的text字段数据
                "sparse_vector": docs_embeddings["sparse"]._getrow(i).todok(),  # 转换为字典格式
                "dense_vector": docs_embeddings["dense"][i]
            }
            col.insert(tmp)
    except Exception as e:
        print(f"添加失败，原因：{e}")
        return
    print(f"添加成功")
    

def slice_document_by_heading(context):
    """
    根据单个#标题格式切片文档
    支持以下格式:
    1. "#" 开头的标题 (如 "# 标题")
    """
    import os
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "original_context.txt")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(context)
        print(f"原始内容已保存到: {log_file}")
    except Exception as e:
        print(f"保存原始内容时出错: {e}")
        
    # 预处理：去除每行开头的空白字符
    lines = context.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    context = '\n'.join(cleaned_lines)
    
    # 确保文档以换行符开始和结束，便于匹配
    context = "\n" + context.strip() + "\n"
    
    # 匹配单个#开头的标题格式
    pattern = r'(?:\n)(# .+?)(?:\n)(.+?)(?=\n# .+?|\Z)'
    matches = re.findall(pattern, context, re.DOTALL)
    
    
    documents = []
    for match in matches:
        title = match[0].strip()
        content = match[1].strip()
        # 过滤掉过短的内容
        if len(content) > 20:  # 可以调整最小内容长度
            documents.append(f"{title}\n{content}")
    
    return documents

def get_collection_minerU(context, uri, embedding_model, 
                            col_name="user_docs", 
                            chunk_size=800, 
                            overlap=200,
                            batch_size=200,
                            text_max_length=2048,
                            consistency_level="Strong",
                            num_workers=4,
                            alias="user_kb"):  # 添加别名参数
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
                                num_workers (int): 并行处理的工作线程数
                                alias (str): 连接别名，默认为"user_kb"
                                
                            Returns:
                                Collection: Milvus集合对象
                            """
                            import time
                            from concurrent.futures import ThreadPoolExecutor
                            import numpy as np
                            from scipy.sparse import vstack, csr_matrix
                            
                            start_time = time.time()
                            
                            # 连接到Milvus，使用自定义别名
                            try:
                                # 如果已有同名连接，先断开
                                if connections.has_connection(alias):
                                    connections.disconnect(alias)
                                    print(f"已断开现有的 {alias} 连接")
                                
                                # 创建新连接
                                connections.connect(uri=uri, alias=alias)
                                print(f"已创建新的连接，别名: {alias}")
                            except Exception as e:
                                print(f"连接数据库时出错: {e}")
                                return None
                            
                            # 1. 文档切片
                            print("开始文档切片...")
                            documents = slice_document_by_heading(context)
                            
                            print(f"文档切片完成，共{len(documents)}个切片，耗时: {time.time() - start_time:.2f} 秒")
                            
                            # 2. 生成嵌入向量 (并行处理)
                            embed_start_time = time.time()
                            print("开始生成嵌入向量...")
                            
                            # 将文档分成多个批次
                            batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
                            
                            # 定义批处理函数
                            def process_batch(batch_texts):
                                return embedding_model(batch_texts)
                            
                            # 并行处理每个批次，并添加进度条
                            all_embeddings = {"sparse": [], "dense": []}
                            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                                # 使用tqdm显示进度
                                batch_results = list(tqdm(
                                    executor.map(process_batch, batches),
                                    total=len(batches),
                                    desc="生成嵌入向量"
                                ))
                            
                            # 合并结果
                            for result in batch_results:
                                all_embeddings["sparse"].append(result["sparse"])
                                all_embeddings["dense"].extend(result["dense"])
                            
                            # 合并稀疏矩阵
                            if all_embeddings["sparse"]:
                                all_embeddings["sparse"] = vstack(all_embeddings["sparse"])
                            
                            docs_embeddings = all_embeddings
                            
                            print(f"嵌入向量生成完成，耗时: {time.time() - embed_start_time:.2f} 秒")
                        
                            # 3. 创建集合
                            collection_start_time = time.time()
                            print("开始创建集合...")
                            
                            fields = [
                                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                                FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128),
                                FieldSchema(name="grade", dtype=DataType.VARCHAR, max_length=128),
                                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=128),
                                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=text_max_length),
                            
                                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.dim["dense"]),
                            ]
                            
                            schema = CollectionSchema(fields)
                            
                            # 如果集合已存在，则删除 (注意使用别名)
                            if utility.has_collection(col_name, using=alias):
                                Collection(col_name, using=alias).drop()
                            
                            # 创建新集合 (注意使用别名)
                            col = Collection(col_name, schema, consistency_level=consistency_level, using=alias)
                            
                            print(f"集合创建完成，耗时: {time.time() - collection_start_time:.2f} 秒")
                            
                            # 5. 批量插入数据 (先插入数据，后创建索引)
                            insert_start_time = time.time()
                            print("开始插入数据...")
                            
                            try:
                                # 使用tqdm显示总体进度
                                for i in tqdm(range(0, len(documents), batch_size), desc="批量插入数据"):
                                    end_idx = min(i + batch_size, len(documents))
                                    
                                    # 准备批量数据 - 使用实体列表格式，与add_data函数保持一致
                                    batch_data = []
                                    
                                    for j in range(i, end_idx):
                                        # 从文档中提取标题和内容
                                        doc_parts = documents[j].split('\n', 1)
                                        title = doc_parts[0].replace('# ', '') if len(doc_parts) > 0 else "无标题"
                                        content = doc_parts[1] if len(doc_parts) > 1 else documents[j]
                                        
                                        # 构建数据字典 - 与add_data函数相同的格式
                                        data_item = {
                                            "subject": "",  # 空字符串代替None
                                            "grade": "",    # 空字符串代替None
                                            "title": title,
                                            "content": content,
                                            "sparse_vector": docs_embeddings["sparse"]._getrow(j).todok(),  # 与add_data相同
                                            "dense_vector": docs_embeddings["dense"][j]
                                        }
                                        batch_data.append(data_item)
                                    
                                    # 批量插入
                                    col.insert(batch_data)
                                
                                print(f"数据插入完成，共{col.num_entities}条记录，耗时: {time.time() - insert_start_time:.2f} 秒")
                                
                                # 4. 创建索引 (移到数据插入后)
                                index_start_time = time.time()
                                print("开始创建索引...")
                                
                                sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
                                col.create_index("sparse_vector", sparse_index)
                                
                                dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
                                col.create_index("dense_vector", dense_index)
                                col.load(using=alias)
                                
                                print(f"索引创建完成，耗时: {time.time() - index_start_time:.2f} 秒")
                            except Exception as e:
                                print(f"处理失败，原因：{e}")
                                return None
                            
                            # 加载集合以便搜索
                            load_start_time = time.time()
                            print("开始加载集合...")
                            col.load()
                            print(f"集合加载完成，耗时: {time.time() - load_start_time:.2f} 秒")
                            
                            total_time = time.time() - start_time
                            print(f"总处理时间: {total_time:.2f} 秒")
                            
                            return col
