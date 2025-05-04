"""
LLM 测试演示文件
用于测试 graph/llms/llms.py 中的函数
"""

from graph.llms.llms import get_llm_response, get_llm_by_type
from langchain_core.messages import HumanMessage, SystemMessage
import httpx

def test_get_llm_response():
    """测试直接获取 LLM 响应的函数"""
    print("===== 测试 get_llm_response 函数 =====")
    
    # 测试 Ollama 模型
    prompt = "请用一句话解释什么是量子力学"
    print("\n使用 Ollama 模型:")
    try:
        response = get_llm_response(
            prompt=prompt,
            model="qwen2.5:14b",
            model_type="ollama",
            options={"format": "json", "num_ctx": 8192, "device": "cuda:0"}
        )
        print(f"响应: {response}")
    except Exception as e:
        print(f"Ollama 模型调用出错: {e}")
    
    # 测试 OpenAI 模型
    print("\n使用 OpenAI 模型:")
    try:
        response = get_llm_response(
            prompt=prompt,
            model="deepseek-ai/DeepSeek-V3",
            model_type="openai"
        )
        print(f"响应: {response}")
    except Exception as e:
        print(f"OpenAI 模型调用出错: {e}")

def test_get_llm_by_type():
    """测试获取 LLM 实例的函数"""
    print("\n===== 测试 get_llm_by_type 函数 =====")
    
    # 构建测试消息
    messages = [
        SystemMessage(content="你是一个物理学教授"),
        HumanMessage(content="用简单的比喻解释量子隧穿效应")
    ]
    
    # 测试 OpenAI 类型
    print("\n使用 OpenAI 类型:")
    try:
        llm = get_llm_by_type("openai")
        response = llm.invoke(messages)
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"OpenAI 类型调用出错: {e}")
    
    # 测试 Ollama 类型
    print("\n使用 Ollama 类型:")
    try:
        llm = get_llm_by_type("ollama")
        response = llm.invoke(messages)
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"Ollama 类型调用出错: {e}")
    
    # 注意: Qwen 类型需要额外的模型和分词器参数，这里不进行测试

if __name__ == "__main__":
    print("开始测试 LLM 函数...")
    
    # 修复可能的 httpx 导入问题
    try:
        import httpx
    except ImportError:
        print("httpx 库未安装，请使用 pip install httpx 安装")
        exit(1)
    
    # 运行测试函数
    test_get_llm_response()
    test_get_llm_by_type()
    
    print("\n测试完成！")