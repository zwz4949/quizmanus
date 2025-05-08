import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.graph.builder import build_rag,build_main
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from src.graph.nodes.quiz_types import embeddings, reranker
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from src.utils import getData,get_json_result,saveData
from tqdm import tqdm
from src.config.llms import generator_model,qwen_model_path

import numpy as np

from src.config.llms import eval_llm_type,eval_model

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# 如果是GPU环境
torch.cuda.manual_seed_all(seed)

load_dotenv()  # 加载 .env 文件

test_file_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/test_qwen.json"
save_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_2_5_72b"
# save_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_14b_quiz_5244"

def run():
    graph = build_main()
    if generator_model == "qwen":
        model_path = qwen_model_path
        

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=True,
            padding_side='left'  # 新增参数，指定左填充
        )
        tokenizer.pad_token = tokenizer.eos_token

        # 2. 使用更高效的模型加载方式
        compute_dtype = torch.float16

        # 3. 启用Flash Attention (如果模型支持)
        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            # "torch_dtype": compute_dtype,
        }

        # 检查是否支持Flash Attention
        # if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        #     kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs
        )

        # 4. 调整模型以适应新的tokenizer大小
        model.resize_token_embeddings(len(tokenizer))

        # 6. 编译模型 (PyTorch 2.0+特性)
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")

        # 设置为评估模式
        model.eval()
    else:
        model = None
        tokenizer = None
    
    os.makedirs(save_dir, exist_ok=True)
    tmp_test = getData(test_file_path)
    for item in tmp_test:
        item['quiz_url'] = os.path.join(save_dir,f"{item['id']}.md")
    saveData(tmp_test,test_file_path)
    for idx,file_item in enumerate(tqdm(getData(test_file_path))):
        # if idx+1 <3:
        #     continue
        # if idx+1 not in [3,7,11,12,14]:
        #     continue
        user_input = file_item['query']

        # embeddings
        # reranker
        graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "ori_query": user_input,
            "quiz_url": file_item['quiz_url'],
            "rag_graph": build_rag(),
            "search_before_planning": False,
            "generate_tokenizer":tokenizer,
            "generate_model":model,
            "rag": {
                "embedding_model": embeddings,
                "reranker_model": reranker
            }
        },
        config={"recursion_limit": 100})

from evaluate import evaluate_quiz

import json

if eval_llm_type == "hkust":
    tail = ""
else:
    tail = f"_{eval_llm_type}_{eval_model}"
def test():
    print("开始evaluate")
    evaluate_quiz(getData(test_file_path),f"{save_dir}/eval_result{tail}.jsonl")


from collections import *
def statistic():
    cnt = defaultdict(int)
    eval_res = getData(f"{save_dir}/eval_result{tail}.jsonl")
    for item in eval_res:
        for key in item['eval_res']:
            cnt[key]+=item['eval_res'][key]
    for key in cnt:
        print(key,cnt[key]/len(eval_res))
    llama3_1 = getData(f"{save_dir}/eval_result_ollama_llama3.1:70b.jsonl")
    qwen3 = getData(f"{save_dir}/eval_result_ollama_qwen3:32b.jsonl")
    r1 = getData(f"{save_dir}/eval_result.jsonl")
    n = llama3_1+qwen3+r1
    exist_n = 0
    if len(llama3_1)>0:
        exist_n+=1
    if len(qwen3)>0:
        exist_n+=1
    if len(r1)>0:
        exist_n+=1
    cnt_all = defaultdict(int)
    # print("llama")
    for item in llama3_1:
        for key in item['eval_res']:
            cnt_all[key]+=item['eval_res'][key]/exist_n
    # print("qwen3")
    for item in qwen3:
        for key in item['eval_res']:
            cnt_all[key]+=item['eval_res'][key]/exist_n
    # print("deepseek_r1")
    for item in r1:
        for key in item['eval_res']:
            cnt_all[key]+=item['eval_res'][key]/exist_n
    ave = 0
    for key in cnt_all:
        ave+=cnt_all[key]/len(eval_res)
        print(key,cnt_all[key]/len(eval_res))
    print("平均分：",ave/len(cnt_all))
    
        
            

run()
# test()
# statistic()