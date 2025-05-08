from langgraph.graph import StateGraph, START
from .nodes.rag_nodes import (
    rag_hyde,
    rag_retrieve,
    rag_router,
    rag_reranker,
    rag_generator,
)
from .nodes.nodes import (
    main_coordinator,
    main_supervisor,
    main_planner,
    main_browser_generator,
    main_reporter,
    main_rag,
    main_rag_browser,
)
# 导入MinerU相关节点
from .nodes.miner_nodes import (
    miner_router,
    pdf_processor,
    image_processor,
    ppt_processor,
)
from .nodes.quiz_types import State

def build_main():
    builder = StateGraph(State)
    # 修改入口点，先经过miner_router判断
    builder.add_edge(START, "miner_router")
    # builder.add_edge(START, "coordinator")

    # 添加MinerU相关节点
    builder.add_node("miner_router", miner_router)
    builder.add_node("pdf_processor", pdf_processor)
    builder.add_node("image_processor", image_processor)
    builder.add_node("ppt_processor", ppt_processor)
    # 原有节点
    builder.add_node("coordinator", main_coordinator)
    builder.add_node("supervisor", main_supervisor)
    builder.add_node("planner", main_planner)
    # builder.add_node("browser_generator", main_browser_generator)
    builder.add_node("reporter", main_reporter)
    builder.add_node("rag_er", main_rag)
    builder.add_node("rag_and_browser", main_rag_browser)
    
    # 添加从处理器到coordinator的边
    # builder.add_edge("pdf_processor", "coordinator")
    # builder.add_edge("image_processor", "coordinator")
    # builder.add_edge("ppt_processor", "coordinator")
    # builder.add_edge("miner_router", "coordinator")
    
    return builder.compile()

def build_rag():
    builder = StateGraph(State)
    builder.add_edge(START, "rewrite")
    builder.add_node("rewrite", rag_hyde)
    builder.add_node("retriever", rag_retrieve)
    builder.add_node("router", rag_router)
    builder.add_node("reranker", rag_reranker)
    builder.add_node("generator", rag_generator)
    return builder.compile()


