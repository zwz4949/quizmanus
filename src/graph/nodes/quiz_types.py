# from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from typing import List, Dict, TypedDict, Literal, Optional, Annotated, Union
from operator import add


from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from langgraph.graph import StateGraph
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
embeddings = BGEM3EmbeddingFunction(
    model_name = "/hpc2hdd/home/fye374/models/BAAI/bge-m3",
    use_fp16=False, 
    device="cuda")

reranker = BGERerankFunction(
    model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",  
    device="cuda",
    use_fp16=False
)


class RAGState(MessagesState):
    
    hyde_query: str
    selected_subject: str
    retrieved_docs: List[str]
    reranked_docs: List[str]
    embedding_model: BGEM3EmbeddingFunction
    reranker_model: BGERerankFunction
    enable_browser: bool
    outer_knowledge: str


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # # Constants
    # TEAM_MEMBERS: list[str]
    # TEAM_MEMBER_CONFIGRATIONS: dict[str, dict]

    # Runtime Variables
    # messages: Annotated[List,add]
    ori_query: str
    rag_graph: StateGraph
    existed_qa: Annotated[List,add]
    next: str
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
    next_work: str
    rag: RAGState
    generate_tokenizer: AutoTokenizer
    generate_model: Optional[Union[AutoModelForCausalLM, PeftModel]]
    quiz_url: str
    # 在State类型中添加custom_knowledge_base字段
    State = TypedDict("State", {
        "custom_knowledge_base": Optional[Dict[str, Any]],  # 添加自定义知识库字段
    })


