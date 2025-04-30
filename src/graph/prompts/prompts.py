
from datetime import datetime
import os
from jinja2 import Environment, FileSystemLoader
from ...config.nodes import TEAM_MEMBERS, TEAM_MEMBER_CONFIGRATIONS,QUESTION_TYPES
from ...config.rag import SUBJECTS,DETIALED_SUBJECTS
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import SystemMessage
# 获取当前脚本所在的目录

def get_prompt_template(node_name,**kwargs):
    '''
    node_name: coordinator
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(script_dir))
    CURRENT_TIME = datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")
    context = {
        "CURRENT_TIME": CURRENT_TIME,
        "TEAM_MEMBERS": TEAM_MEMBERS,
        "TEAM_MEMBER_CONFIGRATIONS": TEAM_MEMBER_CONFIGRATIONS,
        "SUBJECTS": SUBJECTS,
        "QUESTION_TYPES": QUESTION_TYPES,
        "DETIALED_SUBJECTS": DETIALED_SUBJECTS
    }
    if 'SUBJECT' in kwargs:
        context = {**context,"SUBJECT": kwargs['SUBJECT']}
    
    template = env.get_template(f'{node_name}.md')
    # 渲染模板
    system_message = template.render(**context)
    return system_message

def apply_prompt_template(prompt_name: str, state: AgentState,**kwargs) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    # state_vars = {
    #     "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
    #     "TEAM_MEMBERS": TEAM_MEMBERS,
    #     "TEAM_MEMBER_CONFIGRATIONS": TEAM_MEMBER_CONFIGRATIONS,
    #     **state,
    # }

    try:
        # template = env.get_template(f"{prompt_name}.md")
        # system_prompt = template.render(**state_vars)
        system_message = get_prompt_template(prompt_name,**kwargs)
        system_message = SystemMessage(content = system_message)
        messages = [system_message] + state["messages"]
        return messages
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
