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
def get_collection(db_uri,col_name):
    
    connections.connect(uri=db_uri)
    try:
        col = Collection(col_name)
    except:
        print(f"collection {col_name} not found, 请创建collection")
        return None
    col.load()
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
    
    