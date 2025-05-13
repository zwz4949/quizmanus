from typing import List, Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage,messages_to_dict
from .quiz_types import State
from langgraph.types import Command
from ..prompts.prompts import get_prompt_template,apply_prompt_template
from ..llms.llms import get_llm_by_type
from ...config.nodes import TEAM_MEMBERS
from ..agents.agents import browser_generator, knowledge_searcher
from ..tools.search import tavily_tool
from ...config.llms import llm_type,generator_model,reporter_llm_type,planner_llm_type,supervisor_llm_type
from copy import deepcopy
import json
import logging
import json_repair
from ...utils import get_json_result,call_Hkust_api
from ...config.llms import llm_type
from ...config.rag import SUBJECTS
import re
# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def main_coordinator(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    system_message = get_prompt_template("coordinator")
    messages = [
        SystemMessage(
            content = system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    
    response_content = re.sub(r'<think>.*?</think>', '', get_llm_by_type(type = llm_type).invoke(messages).content, flags=re.DOTALL).strip()
    logger.info(f"Current state messages: {state['messages']}")
    # 尝试修复可能的JSON输出
    logger.info(f"Coordinator response: {response_content}")

    goto = "__end__"
    if "handoff_to_planner" in response_content:
        goto = "planner"

    # 更新response.content为修复后的内容
    # response.content = response_content

    return Command(
        goto=goto,
    )
def main_planner(state: State):
    parser = JsonOutputParser()
    def inner_router():
        for i in range(3):
            try:
                system_message = get_prompt_template("knowledge_store_router")
                messages = [
                    SystemMessage(
                        content = system_message
                    ),
                    HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
                ]
                response = re.sub(r'<think>.*?</think>', '', get_llm_by_type(type = llm_type).invoke(messages).content, flags=re.DOTALL).strip()
                # 5. 解析JSON输出
                parser = JsonOutputParser()
                result = parser.parse(response)
                
                # 6. 验证结果是否在可用知识库中
                # valid_sources = {t["name"] for t in VECTORSTORES}
                if result["subject"] not in SUBJECTS:
                    logger.warning(f"第{i+1}次尝试：选择的知识库不存在: {result['subject']}")
                    # return Command(
                    #     goto = "__end__"
                    # )
                    continue
                logger.info(f"router: {result["subject"]}")
                return result["subject"]
            except Exception as e:
                logger.warning(f"模型返回非法JSON: {response} {e}")
            except KeyError as e:
                logger.warning(f"模型返回缺少必要字段: {response} {e}")
        return "无可用知识库"
    # 1. 定义 JSON 输出解析器
    subject = inner_router()
    system_message = get_prompt_template("planner",SUBJECT=subject)
    messages = [
        SystemMessage(
            content = system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    llm = get_llm_by_type(type = planner_llm_type)
    
    if state.get("search_before_planning"):
        searched_content = str(knowledge_searcher.invoke(state)["messages"][-1].content)
        messages = deepcopy(messages)
        messages[
            -1
        ].content += f"\n\n# 相关搜索结果\n\n{searched_content}"
    full_response = ""
    # 报错重传
    for i in range(3):
        try:
            stream = llm.stream(messages)
            full_response = ""
            for chunk in stream:
                full_response += chunk.content
        except Exception as e:
            logger.warning(f"plan生成报错：{e}，重试第{i+1}次。")
            if i == 2:
                try:
                    full_response = llm.invoke(messages)
                except Exception as e1:
                    logger.warning(f"plan生成报错：{e1}")
                    raise Exception("网络错误")
    full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
    logger.info(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    messages_tmp=[]
    try:
        repaired_response = json_repair.loads(full_response)
        generator_agents = set(['rag_er','rag_and_browser'])
        need_to_generate = [resi for resi in repaired_response['steps'] if resi['agent_name'] in generator_agents]
        full_response = json.dumps(repaired_response, ensure_ascii=False, indent=2)
        ## asyncio_generator 串行生成每道题目，目前还没实现并行
        existed_qa,messages = asyncio_generator(state,need_to_generate)
        state['existed_qa'].extend(existed_qa)
        messages_tmp.extend(messages)

    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"
    return_value_of_extend = [HumanMessage(content=full_response,name="planner")]
    return_value_of_extend.extend(messages_tmp) # return_value_of_extend is None
    return Command(
        update={
            "messages": return_value_of_extend,
            "full_plan": full_response,
        },
        goto=goto,
    )

def asyncio_generator(state: State,need_to_generate: List):
    '''
    串行生成每道题目，目前还没实现并行
    返回existed_qa和messages
    '''
    existed_qa = []
    messages = []
    inputs = []
    ## 循环获取batch生成题目的输入messages
    for needi in need_to_generate:
        try:
            updated_rag = {
                **state['rag'],
                "enable_browser": False if needi['agent_name'] == "rag_er" else True
            }
            ## True为获取输入用于batch生成
            if generator_model == "qwen":
                updated_rag['get_input'] = True
            else:
                updated_rag['get_input'] = False
            if "note" in needi and len(needi['note'].strip())>0:
                next_step_content = f"title: {needi['title']}\ndescription: {needi['description']}\nnote:{needi['note']}"
            else:
                next_step_content = f"title: {needi['title']}\ndescription: {needi['description']}"
            
            logger.info("Browser agent starting task")
            needi_state = {**state}
            needi_state["next_work"] = next_step_content
            needi_state["rag"] = updated_rag
            rag_state = state['rag_graph'].invoke(needi_state)
            if generator_model == "qwen":
                inputs.append(rag_state['existed_qa'][-1])
            else:
                new_qa = rag_state['existed_qa'][-1]
                existed_qa.append(new_qa)
            new_q = f"题目内容已省略，概括内容为{next_step_content}"
            messages.append(
                HumanMessage(
                    content=new_q,
                    name=needi['agent_name'],
                )
            )
        except Exception as e:
            logger.error(f"asyncio_generator error: {e}")
    if generator_model == "qwen":
        ## batch生成题目
        existed_qa = get_llm_by_type(type = "qwen",model = state['generate_model'],tokenizer =state['generate_tokenizer']).invoke(inputs)
    # existed_qa.
    return existed_qa,messages

    
RESPONSE_FORMAT = "{}的回复:\n\n<response>\n{}\n</response>\n\n*请执行下一步.*"
def main_supervisor(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    parser = JsonOutputParser()
    messages = apply_prompt_template("supervisor",state)
    # preprocess messages to make supervisor execute better.
    messages = deepcopy(messages)
    reports = []
    for message in messages:
        if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
            if message.name == "reporter":
                reports.append(message.content)
            message.content = RESPONSE_FORMAT.format(message.name, message.content)
    for i in range(3):
        try:
            if len(messages)>119:
                dict_messages = messages_to_dict(messages)
                role_mapping = {
                    "system": "system",
                    "human": "user",
                    "ai": "assistant"
                }
                openai_format = [
                    {"role": role_mapping[msg["type"]], "content": msg['data']["content"],"name":msg['data']['name']}
                    for msg in dict_messages
                ]
                logger.info("使用hkust-deepseek-r1")
                response = call_Hkust_api(prompt = "",messages = openai_format)
                parsed_response = get_json_result(response)
            else:
                response = get_llm_by_type(supervisor_llm_type).invoke(messages).content
                response_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                parsed_response = get_json_result(response_content)
            goto = parsed_response["next"]
            next_step_content = parsed_response["next_step_content"]
            break
        except Exception as e:
            logger.warning(f"supervisor出错了：{e}")
    logger.info(f"Current state messages: {state['messages']}")
    logger.info(f"Supervisor response: {response_content}")

    if goto == "FINISH":
        goto = "__end__"
        
        with open(state['quiz_url'], "w", encoding="utf-8") as f:
            f.write(reports[-1])
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")
    if goto == "rag_er":
        updated_rag = {
            **state['rag'],
            "enable_browser": False
        }
    elif goto == "rag_and_browser":
        updated_rag = {
            **state['rag'],
            "enable_browser": True
        }
    else:
        updated_rag = {
            **state['rag']
        }
    return Command(goto=goto, update={"next": goto, "next_work": next_step_content, "rag":updated_rag})


def main_browser_generator(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    for i in range(3):
        try: 
            result = browser_generator.invoke(state)
            logger.info("Browser agent completed task")
            response_content = result["messages"][-1].content
            break
        except Exception as e:
            logger.error(f"Browser agent failed with error: {e}")
            response_content = ""
            return Command(goto="__end__")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="browser_generator",
                )
            ]
        },
        goto="supervisor",
    )


def main_rag(state: State) -> Command[Literal["supervisor"]]:
    """Node for the RAG that performs RAG tasks."""
    logger.info("Browser agent starting task")
    rag_state = state['rag_graph'].invoke(state)
    new_qa = str(rag_state['existed_qa'][-1])
    new_q = f"题目内容已省略，概括内容为{state['next_work']}"
    # if "参考答案" in new_qa:
    #     new_q = new_qa.split("参考答案")[0].strip()
    # elif "答案" in new_qa:
    #     new_q = new_qa.split("答案")[0].strip()
    # else:
    #     new_q = new_qa
    logger.info("RAG agent completed task")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"RAG agent response: {new_qa}")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_er",
                )
            ]
        },
        goto="supervisor",
    )

def main_rag_browser(state: State) -> Command[Literal["supervisor"]]:
    """Node for the RAG that performs RAG tasks."""
    logger.info("Browser agent starting task")
    rag_state = state['rag_graph'].invoke(state)
    new_qa = str(rag_state['existed_qa'][-1])
    # if "参考答案" in new_qa:
    #     new_q = new_qa.split("参考答案")[0].strip()
    # elif "答案" in new_qa:
    #     new_q = new_qa.split("答案")[0].strip()
    # else:
    #     new_q = new_qa
    new_q = f"题目内容已省略，概括内容为{state['next_work']}"
    logger.info("RAG agent completed task")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"RAG agent response: {new_qa}")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_er",
                )
            ]
        },
        goto="supervisor",
    )


def main_reporter(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    tmp_state = {
        "messages":[
            state['messages'][0],
            state['messages'][1],
            HumanMessage(content = '\n\n\n\n'.join(state['existed_qa']))
        ]
    }
    messages = apply_prompt_template("reporter", tmp_state)
    response_content = re.sub(r'<think>.*?</think>', '', get_llm_by_type(reporter_llm_type).invoke(messages).content, flags=re.DOTALL).strip()
    logger.info(f"Current state messages: {state['messages']}")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"reporter response: {response_content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )