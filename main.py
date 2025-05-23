import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
# os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '1'
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

from vllm import LLM, SamplingParams
from src.graph.builder import build_rag,build_main
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from src.graph.nodes.quiz_types import embeddings, reranker
# from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from src.utils import getData,get_json_result,saveData
from tqdm import tqdm
from src.config.llms import generator_model,qwen_model_path,qwen_tokenizer_path

import numpy as np

from src.config.llms import eval_llm_type,eval_model
import torch
from transformers import AutoTokenizer

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# 如果是GPU环境
torch.cuda.manual_seed_all(seed)

load_dotenv()  # 加载 .env 文件

test_file_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/test备份.json"
# save_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_2_5_72b"
save_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_7b_full_quiz_gemini_40303"


## 配置logging
# import sys
# import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
#     handlers=[
#         logging.StreamHandler(sys.stdout) # 输出到标准输出
#     ]
#     # force=True # Python 3.8+ 如果需要覆盖其他可能的早期配置
# )
import sys
import logging

# 创建 StreamHandler 实例
console_handler = logging.StreamHandler(sys.stdout)

# *** 在这里设置 StreamHandler 的级别为 INFO ***
console_handler.setLevel(logging.INFO)

# 配置 basicConfig，将设置好级别的 console_handler 传入
# basicConfig 的 level 仍然可以保留 DEBUG，这样如果以后你添加了其他 Handler (比如 FileHandler)
# 它们可以根据自己的设置来决定是否处理 DEBUG 消息
logging.basicConfig(
    level=logging.DEBUG, # Logger 的级别仍然是 DEBUG，可以捕获所有消息
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        console_handler # 将已设置好级别的 handler 传入
    ]
    # force=True # Python 3.8+ 如果需要覆盖其他可能的早期配置
)


def run():
    
    graph = build_main()
    if generator_model == "qwen":
        model_path = qwen_model_path

        # 1. Tokenizer 保持不变，左填充 + EOS 作为 pad
        tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token

        # # 2. vLLM LLM 对象
        # compute_dtype = torch.bfloat16
        # model = LLM(
        #     model=model_path,
        #     tensor_parallel_size=1,
        #     trust_remote_code=True,
        #     load_format="auto",
        #     gpu_memory_utilization=0.3,
        #     max_model_len=4608,
        #     max_num_seqs=16,
        #     enforce_eager=False,
        #     # prefill_parallelism = 1,
        #     # num_generate_workers = 1
        # )
        # pass
        model = None
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
            # continue
        if idx+1 <=13:
            continue
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
    
        
            
if __name__ == '__main__':

    # run()
    test()
    statistic()