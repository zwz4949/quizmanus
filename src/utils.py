import httpx
from openai import OpenAI
import requests
import torch 
import time
import json
import os
import re
from typing import List, Dict, Union
from tqdm import tqdm
import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
import ALL_KEYS
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

def getData(path:str)->list:
    if not os.path.exists(path):
        return []
    
    if path.endswith('.json'):    
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.endswith('.txt'):   
        with open(path, 'r', encoding='utf-8') as f:
            # 对于txt文件，逐行读取并将每一行添加到数据列表中
            data = [line.strip() for line in f]
    elif path.endswith('.jsonl'):   
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in tqdm(f) if line.strip()]
    elif path.endswith(".md"):
        # 使用 pathlib 处理路径
        md_file = Path(path)
        data = md_file.read_text(encoding='utf-8')
    return data

def saveData(data:list, path:str)->None:
    if path.endswith('json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith('jsonl'):
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(data, (list, dict)):
                for item in data if isinstance(data, list) else data.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif path.endswith('txt'):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write("%s\n" % item)
    else:
        raise ValueError("Unsupported file type: %s" % path)
    
def removeDuplicates(data:list)->list:
    '''
    去重，取最后出现的。
    '''
    ids = []
    for item in data:
        if item['id'] not in ids:
            ids.append(item['id'])
    tmp_dict = {}
    result_list = []
    for item in data:
        tmp_dict[item['id']] = item
    for id in ids:
        result_list.append(tmp_dict[id])
    return result_list

def getHkustClient(api_type = "DeepSeek-R1-671B"):
    
    client = OpenAI(
        base_url = ALL_KEYS.hkust_openai_base_url,
        api_key = ALL_KEYS.hkust_openai_key,
        http_client=httpx.Client(
            base_url=ALL_KEYS.hkust_openai_base_url,
            follow_redirects=True,
        ),
    )
    return client


def call_Hkust_api(prompt, messages = [],remain_reasoning = False, api_type = "DeepSeek-R1-671B",config = {"temperature":0.7}):
    try:
        # 调试输入参数
        print("="*50)
        print("调试信息 - call_Hkust_api 输入:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Messages 数量: {len(messages)}")
        for i, msg in enumerate(messages[:3]):  # 只打印前3条消息
            print(f"消息 {i}: role={msg.get('role', 'unknown')}, content前50个字符: {msg.get('content', '')[:50]}...")
        print(f"remain_reasoning: {remain_reasoning}")
        print(f"api_type: {api_type}")
        print(f"config: {config}")
        print("="*50)
        
        url = ALL_KEYS.hkust_openai_base_url
        headers = { 
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {ALL_KEYS.Authorization_hkust_key}" #Please change your KEY. If your key is XXX, the Authorization is "Authorization": "Bearer XXX"
        }
        data = { 
        "model": "DeepSeek-R1-671B", # # "gpt-3.5-turbo" version in gpt-4o-mini, "gpt-4" version in gpt-4o-2024-08-06
        "messages": [{"role": "user", "content": prompt}] if messages ==[] else messages, 
        **config
        }
        
        # 打印请求数据
        print("请求URL:", url)
        print("请求头:", {k: v[:10]+"..." if k == "Authorization" and len(v) > 10 else v for k, v in headers.items()})
        print("请求数据:", {
            "model": data["model"],
            "messages_count": len(data["messages"]),
            "config": {k: v for k, v in data.items() if k != "messages" and k != "model"}
        })
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # 打印响应状态和头信息
        print("="*50)
        print("调试信息 - call_Hkust_api 响应:")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        # 检查响应是否成功
        if response.status_code != 200:
            print(f"错误响应: {response.text}")
            return ""
            
        # 解析响应内容
        response_json = response.json()
        print(f"响应JSON结构: {list(response_json.keys())}")
        
        result = ""
        if remain_reasoning:
            result = response_json['choices'][0]['message']['content']
        else:
            result = re.sub(r'<think>.*?</think>', '', response_json['choices'][0]['message']['content'], flags=re.DOTALL).strip()
        
        # 打印结果
        print(f"结果前200个字符: {result[:200]}...")
        print("="*50)
        
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return ""

def getClient(api_type = "gpt-4o-mini")->OpenAI:

    client = OpenAI(
        base_url = ALL_KEYS.common_openai_base_url,
        api_key = ALL_KEYS.common_openai_key,
        http_client=httpx.Client(
            base_url=ALL_KEYS.common_openai_base_url,
            follow_redirects=True,
        ),
    )
    return client

def call_api(prompt, api_type = 'gpt-4o-mini'):
    try:
        response = getClient(api_type).chat.completions.create(
            model=api_type,
            # temperature=float(temperature),
            # max_tokens=int(max_tokens),
            # top_p=float(top_p),
            messages=[
                # {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
        
import re
def get_list_text(text):

    # text = "dawd[\"asdada\"]dadaw"
    match = re.search(r'\[(.*?)\]', text)
    if match:
        full_match = match.group(0)  # 这将取出包括方括号在内的完整匹配
        return full_match
    return -1
def get_json_text(text):

    # text = "dawd[\"asdada\"]dadaw"
    match = re.search(r'\{(.*?)\}', text)
    if match:
        full_match = match.group(0)  # 这将取出包括方括号在内的完整匹配
        return full_match
    return -1


def get_absolute_file_paths(absolute_dir,file_type)->List[str]:
    '''
    absolute_dir: 文件夹
    file_type: "md","json"...
    '''
    json_files = [os.path.join(absolute_dir,f) for f in os.listdir(absolute_dir) if f.endswith(f".{file_type}")]
    return json_files


from langchain_core.output_parsers import JsonOutputParser
import json_repair
def get_json_result(text):
    # parser = JsonOutputParser()
    # parsed_response = parser.parse(text)
    parsed_response = json_repair.loads(text)
    if isinstance(parsed_response,list):
        parsed_response = parsed_response[-1]
    return parsed_response
