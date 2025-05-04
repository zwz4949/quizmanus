import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from src.graph.builder import build_rag, build_main
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from src.graph.nodes.quiz_types import embeddings, reranker
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from src.utils import getData, get_json_result, saveData
from tqdm import tqdm
from src.config.llms import generator_model
import json
import re
import numpy as np
import shutil  # 导入shutil模块用于文件操作

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# 如果是GPU环境
torch.cuda.manual_seed_all(seed)

load_dotenv()  # 加载 .env 文件

# 导入配置和数据库相关模块
from src.config.rag import TEMP_DB_URI
import os

def is_pdf_path(path: str) -> bool:
    """判断是否为PDF文件路径"""
    if re.search(r'\.pdf$', path, re.IGNORECASE):
        if os.path.exists(path) and os.path.isfile(path):
            return True
    return False

def is_json_path(path: str) -> bool:
    """判断是否为JSON文件路径"""
    if re.search(r'\.json$', path, re.IGNORECASE):
        if os.path.exists(path) and os.path.isfile(path):
            return True
    return False

def reset_temp_database():
    """重置临时数据库"""
    # 确保TEMP_DB_URI是一个有效的文件路径
    if os.path.exists(TEMP_DB_URI):
        try:
            # 删除现有数据库文件
            os.remove(TEMP_DB_URI)
            print(f"已重置临时数据库: {TEMP_DB_URI}")
        except Exception as e:
            print(f"重置数据库时出错: {e}")
    else:
        print(f"临时数据库文件不存在，无需重置: {TEMP_DB_URI}")
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(TEMP_DB_URI)
    os.makedirs(db_dir, exist_ok=True)

def run():
    graph = build_main()
    if generator_model == "qwen":
        model_path = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-14b-qlora-gaokao-1072'
        
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
    
    test_file_path = "/hpc2hdd/home/fye374/LJJ/quizmanus/dataset/test.json"
    save_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_14b_quiz_1072"
    os.makedirs(save_dir, exist_ok=True)
    tmp_test = getData(test_file_path)
    for item in tmp_test:
        item['quiz_url'] = os.path.join(save_dir, f"{item['id']}.md")
    saveData(tmp_test, test_file_path)
    
    # 检查是否有PDF文件需要处理，如果有则重置数据库
    has_pdf_to_process = False
    for idx, file_item in enumerate(getData(test_file_path)):
        custom_kb = file_item.get('knowledge_base', None)
        if custom_kb and is_pdf_path(custom_kb):
            has_pdf_to_process = True
            break
    
    # 如果有PDF文件需要处理，重置临时数据库
    if has_pdf_to_process:
        print("检测到PDF文件需要处理，重置临时数据库...")
        reset_temp_database()
    
    for idx, file_item in tqdm(enumerate(getData(test_file_path))):
        print(f"#######开始生成 第{idx}个试卷")
        user_input = file_item['query']
        custom_kb = file_item.get('knowledge_base', None)  # 获取可选的知识库路径
        print(f'knowledge_base: {custom_kb}')
        # 初始化自定义知识库为None
        custom_knowledge_base = None
        
        # 如果提供了知识库路径，检查其类型
        if custom_kb:
            if is_pdf_path(custom_kb):
                # PDF将由miner_processor处理，这里只需标记类型
                custom_knowledge_base = {
                    "type": "pdf",
                    "path": custom_kb
                }
            elif is_json_path(custom_kb):
                # 加载JSON知识库
                pass

        # 调用图执行
        graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "ori_query": user_input,
            "quiz_url": file_item['quiz_url'],
            "rag_graph": build_rag(),
            "search_before_planning": False,
            "generate_tokenizer": tokenizer,
            "generate_model": model,
            "custom_knowledge_base": custom_knowledge_base,  # 添加自定义知识库
            "rag": {
                "embedding_model": embeddings,
                "reranker_model": reranker
            }
        },
        config={"recursion_limit": 100})

    try:
        from pymilvus import connections
        connections.disconnect("default")
        for alias in ["user_kb", "custom_kb"]:
            try:
                if connections.has_connection(alias):
                    connections.disconnect(alias)
            except:
                pass
        print("已断开所有数据库连接")
        
        # 清理FlagEmbedding资源
        if 'embeddings' in globals() and hasattr(embeddings, 'stop_self_pool'):
            if callable(embeddings.stop_self_pool):
                embeddings.stop_self_pool()
        
        if 'reranker' in globals() and hasattr(reranker, 'stop_self_pool'):
            if callable(reranker.stop_self_pool):
                reranker.stop_self_pool()
        
        print("已清理嵌入模型资源")
    except Exception as e:
        print(f"清理资源时出错: {e}")

from evaluate import evaluate_quiz

import json
def test():
    print("开始evaluate")
    evaluate_quiz(getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/test.json"),"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")


from collections import *
def statistic():
    cnt = defaultdict(int)
    eval_res = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")
    for item in eval_res:
        for key in item['eval_res']:
            cnt[key]+=item['eval_res'][key]
    for key in cnt:
        print(key,cnt[key]/len(eval_res))


if __name__ == "__main__":
    try:
        run()
        # test()
        # statistic()
        from pymilvus import connections
        connections.disconnect_all()
        print("程序正常结束，已断开所有数据库连接")
    except Exception as e:
        print(f"程序执行出错: {e}")

