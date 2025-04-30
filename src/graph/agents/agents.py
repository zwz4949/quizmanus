from langgraph.prebuilt import create_react_agent

from ..prompts.prompts import apply_prompt_template
from ...graph.tools.crawler import crawl_tool
from ...graph.tools.search import tavily_tool
from ...config.llms import llm_type

from ...graph.llms.llms import get_llm_by_type
# from src.config.agents import AGENT_LLM_MAP


# Create agents using configured LLM types
def create_agent(agent_type: str, tools: list, prompt_template: str):
    """Factory function to create agents with consistent configuration."""
    return create_react_agent(
        get_llm_by_type(llm_type),
        tools=tools,
        prompt=lambda state: apply_prompt_template(agent_type, state),
    )

# browser_agent = create_agent("browser", [browser_tool], "browser")
browser_generator = create_agent("browser_generator", [tavily_tool, crawl_tool], "browser_generator")

knowledge_based_browser= create_agent("knowledge_based_browser", [tavily_tool, crawl_tool], "knowledge_based_browser")

knowledge_searcher = create_agent("knowledge_searcher", [tavily_tool, crawl_tool], "knowledge_searcher")