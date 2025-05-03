from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)


def dense_search(col, query_dense_embedding, limit=10):
    '''
    usage
    dense_results = dense_search(col, query_embeddings["dense"][0])
    '''
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["content"],
        param=search_params,
    )[0]
    return [hit.get("content") for hit in res]


def sparse_search(col, query_sparse_embedding, limit=10):
    '''
    usage
    sparse_results = sparse_search(col, query_embeddings["sparse"]._getrow(0))
    '''
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["content"],
        param=search_params,
    )[0]
    return [hit.get("content") for hit in res]


# def hybrid_search(
#     col,
#     query_dense_embedding,
#     query_sparse_embedding,
#     sparse_weight=1.0,
#     dense_weight=1.0,
#     limit=10,
# ):
#     '''
#     usage
#     hybrid_results = hybrid_search(
#         col,
#         query_embeddings["dense"][0],
#         query_embeddings["sparse"]._getrow(0),
#         subject_value="your_subject_value",  # 指定 subject 值
#         sparse_weight=0.7,
#         dense_weight=1.0,
#     )
#     '''
#     dense_search_params = {"metric_type": "IP", "params": {}}
#     dense_req = AnnSearchRequest(
#         [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
#     )
#     sparse_search_params = {"metric_type": "IP", "params": {}}
#     sparse_req = AnnSearchRequest(
#         [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
#     )
#     rerank = WeightedRanker(sparse_weight, dense_weight)
#     res = col.hybrid_search(
#         [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["content"]
#     )[0]
#     return [hit.get("content") for hit in res]


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    subject_value= None,  # 新增参数
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):  
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], 
        "dense_vector", 
        dense_search_params, 
        limit=limit,
        expr=f'subject == "{subject_value}"' if subject_value else None,  # 添加过滤条件
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], 
        "sparse_vector", 
        sparse_search_params, 
        limit=limit,
        expr=f'subject == "{subject_value}"' if subject_value else None  # 添加过滤条件
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], 
        rerank=rerank, 
        limit=limit, 
        output_fields=["content"]
    )[0]
    return [hit.get("content") for hit in res]