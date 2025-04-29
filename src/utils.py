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
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if remain_reasoning:
            return response.json()['choices'][0]['message']['content']
        else:
            return re.sub(r'<think>.*?</think>', '', response.json()['choices'][0]['message']['content'], flags=re.DOTALL).strip()
    except Exception as e:
        print(f"An error occurred: {e}")
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
