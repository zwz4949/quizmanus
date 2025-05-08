"""
OpenAI API和浏览器搜索功能测试脚本
用于测试OpenAI API的连接和响应，以及浏览器搜索功能
"""

import os
import sys
import json
from openai import OpenAI
import httpx
import traceback
from langchain_core.messages import HumanMessage, SystemMessage

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from src.config.llms import openai_model, openai_api_key, openai_api_base
from src.graph.agents.agents import browser_generator
from src.graph.tools.search import tavily_tool
from src.graph.llms.llms import get_llm_by_type

def test_openai_direct():
    """直接使用OpenAI客户端测试API"""
    print("===== 测试直接OpenAI客户端 =====")
    
    try:
        # 创建客户端
        client = OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key,
            http_client=httpx.Client(
                base_url=openai_api_base,
                follow_redirects=True,
            ),
        )
        
        # 发送请求
        print(f"使用模型: {openai_model}")
        print(f"API基础URL: {openai_api_base}")
        print("发送请求...")
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "user", "content": "你好，请用一句话介绍自己。"}
            ]
        )
        
        # 打印原始响应
        print("\n原始响应对象:")
        print(response)
        
        # 打印响应内容
        print("\n响应内容:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 如果是JSON解析错误，打印原始响应
        if "JSONDecodeError" in str(e):
            print("\n尝试打印原始响应...")
            try:
                # 获取原始响应文本
                error_str = str(e)
                if "response=" in error_str:
                    raw_response = error_str.split("response=")[1].split(",")[0]
                    print(f"原始响应: {raw_response}")
            except:
                print("无法提取原始响应")

def test_tavily_search():
    """测试Tavily搜索工具"""
    print("\n===== 测试Tavily搜索工具 =====")
    
    try:
        # 测试搜索
        query = "量子力学的基本原理"
        print(f"搜索查询: {query}")
        
        result = tavily_tool.invoke({"query": query})
        
        print("\n搜索结果:")
        print(result[:500] + "..." if len(result) > 500 else result)
        
        return True
    except Exception as e:
        print(f"搜索错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        
        return False

def test_browser_generator():
    """测试浏览器生成器"""
    print("\n===== 测试浏览器生成器 =====")
    
    # 创建模拟状态
    state = {
        "ori_query": "解释量子力学的基本原理",
        "next_work": "解释量子力学的基本原理和应用",
        "messages": [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="我需要了解量子力学的基本原理")
        ]
    }
    
    # 打印状态信息
    print("="*50)
    print("调试信息 - Browser Generator 输入:")
    print(f"State包含的键: {list(state.keys())}")
    print(f"Next work内容: {state.get('next_work', '无')}")
    if 'messages' in state:
        print(f"消息数量: {len(state['messages'])}")
        for i, msg in enumerate(state['messages']):
            if hasattr(msg, 'name') and hasattr(msg, 'content'):
                print(f"消息 {i} - 名称: {msg.name if hasattr(msg, 'name') else '无名称'}, 内容前50个字符: {msg.content[:50]}...")
    print("="*50)
    
    # 测试browser_generator
    for i in range(3):
        try:
            print(f"尝试调用browser_generator，第{i+1}次")
            result = browser_generator.invoke(state)
            print("浏览器生成器任务完成")
            
            response_content = result["messages"][-1].content
            
            # 打印输出内容
            print("="*50)
            print("调试信息 - Browser Generator 输出:")
            print(f"输出内容前200个字符: {response_content[:200]}...")
            print("="*50)
            
            return True
        except Exception as e:
            print(f"浏览器生成器失败，错误: {e}")
            
            # 打印详细错误信息
            print("="*50)
            print("调试信息 - Browser Generator 错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("详细错误信息:")
            print(traceback.format_exc())
            print("="*50)
            
            if i == 2:  # 最后一次尝试
                return False

if __name__ == "__main__":
    print("开始测试OpenAI API和浏览器搜索功能...")
    
    # 打印配置信息
    print(f"OpenAI模型: {openai_model}")
    print(f"OpenAI API基础URL: {openai_api_base}")
    print(f"OpenAI API密钥: {openai_api_key[:5]}...{openai_api_key[-5:]}")
    print(f"Tavily API密钥: {os.environ.get('TAVILY_API_KEY', '未设置')[:5]}...{os.environ.get('TAVILY_API_KEY', '未设置')[-5:] if len(os.environ.get('TAVILY_API_KEY', '未设置')) > 10 else ''}")
    
    # 运行测试函数
    openai_success = False
    try:
        test_openai_direct()
        openai_success = True
    except Exception as e:
        print(f"OpenAI测试失败: {e}")
    
    # 只有当OpenAI测试成功时才测试搜索功能
    if openai_success:
        tavily_success = test_tavily_search()
        
        # 只有当Tavily搜索成功时才测试浏览器生成器
        if tavily_success:
            test_browser_generator()
    
    print("\n测试完成！")