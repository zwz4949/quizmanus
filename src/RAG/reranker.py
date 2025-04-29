# from FlagEmbedding import FlagReranker
from pymilvus.model.reranker import BGERerankFunction
def rerank(query_text: str, search_results: list, reranker: BGERerankFunction =None, topk = 3):
    """
    对搜索结果进行重排序
    Args:
        query_text: 原始查询文本
        search_results: 初始搜索结果列表
        reranker: 重排序模型实例
    Returns:
        重排序后的结果列表
    ## usage
    reranker = FlagReranker('/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3', use_fp16=False)
    # 然后对结果进行重排序
    reranked_results = rerank_results(
        query_text=query,
        search_results=hybrid_results,
        reranker = reranker
    )
    """
    if reranker is None:
        # from FlagEmbedding import FlagReranker
        # reranker = FlagReranker('/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3', use_fp16=False)
        reranker = BGERerankFunction(
            model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",  
            device="cuda",
            use_fp16=False
        )
    
    # # 准备重排序的数据对
    # pairs = [[query_text, doc] for doc in search_results]
    
    # # 计算相关性分数
    # scores = reranker.compute_score(pairs)
    
    # # 将结果和分数组合并排序
    # scored_results = list(zip(search_results, scores))
    # scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # # 返回重排序后的结果
    # return [item[0] for item in scored_results]
    results = reranker(
        query=query_text,
        documents=search_results,
        top_k=topk
    )
    return [res.text for res in results]

