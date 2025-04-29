import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from ...config.tools import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool
# import sys
# sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
# from src.config.tools import TAVILY_MAX_RESULTS
# from src.graph.tools.decorators import create_logged_tool

from dotenv import load_dotenv
load_dotenv()  # 这行代码需要在导入任何使用环境变量的模块之前执行
# from decorators import create_logged_tool
logger = logging.getLogger(__name__)

# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)

# searched_content = tavily_tool.invoke({"query": "出五道关于细胞壁的题目"})
# print(searched_content)
