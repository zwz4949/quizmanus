from typing import List, Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from jinja2 import Template
# import sys
# import ollama
# from ..config.rag import VECTORSTORES
# from ..config.rag import reranker, embedding_model
from .quiz_types import State
from langgraph.types import Command
# from langchain_community.vectorstores import FAISS
# 定义状态机结构
from ..llms.llms import get_llm_by_type

from ..agents.agents import knowledge_based_browser

from ...config.rag import DB_URI, COLLECTION_NAME, SUBJECTS
from ...RAG.vector_store_utils import get_collection,get_collection_minerU
from ...RAG.retrieval import hybrid_search
from ...RAG.reranker import rerank
from langchain_core.messages import HumanMessage, SystemMessage
from ...config.llms import llm_type,generator_model
from ...config.nodes import QUESTION_TYPES
from ...utils import get_json_result
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def rag_hyde(state: State):
    # 1. 定义 JSON 输出解析器
    parser = JsonOutputParser()
    messages = [
        SystemMessage(
            content='''
            [角色] 查询的假设性文档生成器
            [任务] 用自身知识对当前查询进行改写，改写成和查询相关的课本知识，文档要为陈述句，不要直接生成题目。
            再次强调，不要直接生成题目！！！
            严格返回以下 JSON 格式：
            {
                "hyde_query": "改写后的查询内容"
            }
            '''
        ),
        HumanMessage(content=f'''当前查询：{state["next_work"]}''')
    ]
    
    # for i in range(3):
    # 4. 调用模型（假设 ollama.generate 返回原始文本）
    rewrite_res = get_llm_by_type(type = llm_type).invoke(messages).content
    # 5. 用 JsonOutputParser 解析结果
    try:
        parsed_output = parser.parse(rewrite_res)
        print("hyde",parsed_output)
        updated_rag = {
            **state['rag'],
            "hyde_query": parsed_output["hyde_query"]
        }
        return Command(
            update = {
                "rag":updated_rag
                
            },
            goto = "router"
        )
    except Exception as e:
        # 如果解析失败，返回原始查询或抛出错误
        print(f"第{i+1}次尝试：res: {rewrite_res} error: {e}")
    
    return Command(
        goto = "__end__"
    )


def rag_router(state: State):
    messages = [
        SystemMessage(
            content='''
            [任务] 学科与题型选择决策
    选择标准：
    - 选择与查询语义最相关的1个学科
    - 判断查询是想要生成说明类型的题目
    - 只能返回上述JSON格式，不要包含额外内容
    请严格按以下JSON格式返回结果：
    {
        "subject": "学科名称",
        "question_type": "题型"
    }
            '''
        ),
        HumanMessage(content=f'''可选学科：
    {SUBJECTS}
    可选题型：
    单选题、多选题、主观题

    当前查询："{state["next_work"]}"
    当前扩展查询："{state["rag"]["hyde_query"]}"
    回答：''')
    ]
    
    # 4. 调用模型（建议开启JSON模式）
    # for i in range(3):
    try:
        response = get_llm_by_type(type = llm_type).invoke(messages).content
        
        # 5. 解析JSON输出
        parser = JsonOutputParser()
        result = parser.parse(response)
        
        # 6. 验证结果是否在可用知识库中
        # valid_sources = {t["name"] for t in VECTORSTORES}
        if result["subject"] not in SUBJECTS:
            print(f"第{i+1}次尝试：选择的知识库不存在: {result['subject']}")
            # return Command(
            #     goto = "__end__"
            # )
            
        print("router",result["subject"])
        print("type",result["question_type"])
        updated_rag = {
            **state['rag'],
            "subject": result["subject"],
            "type": result["question_type"],
        }
        return Command(
            update = {
                "rag":updated_rag
            },
            goto = "retriever"
        )
        
    except Exception as e:
        print(f"模型返回非法JSON: {response} {e}")
    except KeyError as e:
        print(f"模型返回缺少必要字段: {response} {e}")
    return Command(
        goto = "__end__"
    )


# 3. 检索执行组件
# 修改检索函数，支持自定义知识库
def rag_retrieve(state: State):
    query_embeddings = state["rag"]['embedding_model']([state["rag"]['hyde_query']])
    
    custom_kb = state.get("custom_knowledge_base", None)
    if custom_kb:
        logger.info("Using custom knowledge base for retrieval")
        kb_type = custom_kb.get("type", "unknown")
        
        if kb_type == "pdf":
            # 使用处理后的PDF内容作为检索结果
            processed_content = custom_kb.get("processed_content", "")
            # 创建集合并插入数据
            col = get_collection_minerU(
                context=processed_content,
                uri=DB_URI,
                embedding_model=state["rag"]['embedding_model'],
                text_max_length=4096,  # 增加文本长度限制
                batch_size=100         # 增加批处理大小
            )
 
            hybrid_results = hybrid_search(
                col,
                query_embeddings["dense"][0],
                query_embeddings["sparse"]._getrow(0),
                subject_value=state["rag"]['subject'],  # 指定 subject 值
                sparse_weight=0.7,
                dense_weight=1.0,
                limit = 10
            )
            updated_rag = {
                **state['rag'],
                "retrieved_docs": hybrid_results
            }
    else:
        col = get_collection(DB_URI,COLLECTION_NAME)
        hybrid_results = hybrid_search(
            col,
            query_embeddings["dense"][0],
            query_embeddings["sparse"]._getrow(0),
            subject_value=state["rag"]['subject'],  # 指定 subject 值
            sparse_weight=0.7,
            dense_weight=1.0,
            limit = 10
        )
        # print("retrieved_docs",hybrid_results)
        updated_rag = {
            **state['rag'],
            "retrieved_docs": hybrid_results
        }
    
    return Command(
        update = {
            "rag":updated_rag
        },
        goto = "reranker"
    )



def rag_reranker(state: State):
    reranked_docs = rerank(
        query_text = state["rag"]['hyde_query'], 
        search_results = state["rag"]['retrieved_docs'], 
        reranker = state["rag"]['reranker_model'],
        topk = 1)
    updated_rag = {
        **state['rag'], 
        "reranked_docs": reranked_docs,
        'outer_knowledge':""
    }
    if state["rag"]['enable_browser']:
        ##

        # for i in range(3):
        try: 
            message_state = {
                "messages":[
                    HumanMessage(content=f'''课本知识：{''.join(reranked_docs)}''')
                ]
            }
            result = knowledge_based_browser.invoke(message_state)
            logger.info("Browser agent completed task")
            response_content = result["messages"][-1].content
            outer_knowledge = get_json_result(response_content)['课外知识']
            updated_rag = {
                **updated_rag, 
                "outer_knowledge": outer_knowledge
            }
        except Exception as e:
            logger.error(f"Browser agent failed with error: {e}\n===============\n{response_content}\n==============")
            outer_knowledge = response_content
            updated_rag = {
                **updated_rag, 
                "outer_knowledge": outer_knowledge
            }
                # return Command(goto="__end__")
        # 尝试修复可能的JSON输出
        # response_content = repair_json_output(response_content)
        # logger.debug(f"Browser agent response: {response_content}")
        print(f"Browser agent response: {outer_knowledge}")

    return Command(
        update = {
            "rag":updated_rag
        },
        goto = "generator"
    )


# 6. 生成组件
def rag_generator(state: State):
    parser = JsonOutputParser()
    
    pass_qa = str(state["existed_qa"])
    # print(state["rag"]["rerank_docs"])
    if len(state["rag"]["reranked_docs"]) == 0:
        context = "\n\n".join(
            state["rag"]["retrieved_docs"]
        )
    else:
        context = "\n\n".join(
            state["rag"]["reranked_docs"]
        )
        
    
    if generator_model == "qwen":
        SYSTEM_PROMPT = '''# 角色说明
你是一个根据课本内容和课外内容生成{type}的专家，给定一段{subject}的课本内容和一段相关的课外内容，请根据他们生成一道高考{type}。

# 回答格式
题干；...
参考答案：...
解析：...'''
        messages=[
            {'role':'system','content':SYSTEM_PROMPT.format(type= state['rag']['type'],subject = state['rag']['subject'])}, 
            {'role':'user','content': f"课本内容：{context}\n课外内容：{state['rag']['outer_knowledge']}"}, 
            {'role':'assistant','content': ''}
        ]
        final_answer = get_llm_by_type(type = "qwen",model = state['generate_model'],tokenizer =state['generate_tokenizer']).invoke(messages)
    else:
        SYSTEM_PROMPT = '''# 角色说明
你是一个根据课本内容和课外内容生成{type}的专家，给定一段{subject}的课本内容和一段相关的课外内容，请根据他们生成一道高考{type}。题目要完整，如果要引用材料信息就在题干包括具体的材料信息。不要在题目中包含课本内容、课内知识、课外知识等字眼。

# 题型说明
{question_type}

# 回答格式
题干；...
参考答案：...
解析：...'''
        question_type = '\n'.join([QUESTION_TYPES[key]['desc_for_llm'] for key in QUESTION_TYPES])
        messages=[
            {'role':'system','content':SYSTEM_PROMPT.format(type= state['rag']['type'],subject = state['rag']['subject'],question_type = question_type)}, 
            {'role':'user','content': f"课本内容：{context}\n课外内容：{state['rag']['outer_knowledge']}"}, 
            {'role':'assistant','content': ''}
        ]
        final_answer = get_llm_by_type(type = generator_model).invoke(messages).content
    
    # parsed_output = parser.parse(final_answer)
    print("final_answer: ",final_answer)
    # return parsed_output
    return Command(
        update = {
            "existed_qa": [
                final_answer
            ]
        },
        goto = "__end__"
    )

