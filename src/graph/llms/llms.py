
from openai import OpenAI
import ollama
import sys
from ...config.llms import openai_model, openai_api_key, openai_api_base, ollama_model

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from langchain.schema.runnable import RunnableLambda

import torch
def getClient()->OpenAI:
    client = OpenAI(
        base_url=openai_api_base, 
        api_key=openai_api_key,# gpt35  
        http_client=httpx.Client(
            base_url=openai_api_base,
            follow_redirects=True,
        ),
    )
    return client

def call_api(prompt, model):
    try:
        response = getClient().chat.completions.create(
            model=model,
            # temperature=float(temperature),
            # max_tokens=int(max_tokens),
            # top_p=float(top_p),
            messages=[
                # {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def get_llm_response(prompt, model, model_type = "ollama", options={"format": "json","num_ctx": 8192,"device": "cuda:0"}):
    '''
    model="qwen2.5:14b"
    model_type = "ollama"
    ======
    model="gpt-4o-mini"
    model_type = "openai"
    '''
    if model_type == "ollama":
        response = ollama.generate(
            model="qwen2.5:14b",
            prompt=prompt,
            options=options,  # 强制JSON输出
        )["response"]
    elif model_type == "openai":
        response = call_api(prompt,model)
    return response
    
    



def get_llm_by_type(type,model = None,tokenizer = None):
    '''
    # paramter:
    type: "ollama","openai","qwen2.5-7b","qwen2.5-3b"
    # usage:
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm_by_type("ollama")
    # 构建消息
    messages = [
        SystemMessage(content="你是一个物理学教授"),
        HumanMessage(content="用简单的比喻解释量子隧穿效应")
    ]
    # 调用模型
    response = llm.invoke(messages)
    print("回答：", response.content)
    '''
    
    if type == "openai":
        llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            base_url=openai_api_base,
            temperature=0.7,
            max_retries=3,
            # 添加以下配置以支持工具调用
            model_kwargs={
                "tools": None  # 设置为None表示不使用工具，避免自动添加工具定义
            }
        )
    elif type == "ollama":
        # 初始化 Ollama（默认连接本地 http://localhost:11434）
        # llm = ChatOllama(
        #     model=ollama_model,  # 可替换为其他本地模型如 "mistral"、"qwen" 等
        #     temperature=0.7,
        #     # 如果 Ollama 服务地址不是默认的，可通过 base_url 修改：
        #     # base_url="http://your-ollama-host:11434"
        # )
        
        llm = ChatOllama(
            model=ollama_model,
            # match your previous options dict
            num_ctx=25600,       # context window size
            temperature=0.7,     # randomness
            stream=False         # synchronous invoke
        )
    elif "qwen" in type.lower():
        # 检查是否有可用的GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def prepare_input(messages, tokenizer):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                # continue_final_message=True
            )
            return prompt.replace("\n<|im_end|>\n",'')+"\n"
        def single_generate(x, model, tokenizer):
            # 将模型移动到当前设备（如果尚未移动）
            model.to(device)
            
            # 单条输入处理并移动到相同设备
            inputs = tokenizer(
                prepare_input(x, tokenizer), 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(device)  # 关键修改：整个inputs移动到模型所在设备
            
            # 单条生成
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                repetition_penalty=1.1
            )
            
            # 解码单条结果（注意移回CPU再解码）
            completion_ids = generated_ids[0][len(inputs.input_ids[0]):].cpu()
            return tokenizer.decode(completion_ids, skip_special_tokens=True)
        
            
        # 将模型包装为 Runnable
        llm = RunnableLambda(lambda x: single_generate(x, model, tokenizer))
    return llm
    

