import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '1'

import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")

from fastapi import FastAPI, Request
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
from src.config.llms import generator_model,qwen_model_path,qwen_tokenizer_path,vllm_sampling_params

import numpy as np
# from src.config.llms import openai_model, openai_api_key, openai_api_base, ollama_model, ollama_num_ctx,vllm_sampling_params
from src.config.llms import eval_llm_type,eval_model
import torch
from transformers import AutoTokenizer
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

app = FastAPI()
model: LLM = None
embeddings: BGEM3EmbeddingFunction = None
reranker: BGERerankFunction = None


# --- Pydantic 模型定义 ---

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="需要生成嵌入的文本列表。", example=["你好世界", "这是一个测试文档"])

class EmbedResponse(BaseModel):
    embeddings: Dict = Field(description="生成的嵌入向量列表。")
    model_name: str = Field(description="所使用的嵌入模型名称。")

class RerankRequest(BaseModel):
    query: str = Field(..., description="用于重排的查询。", example="相关话题")
    documents: List[str] = Field(..., description="需要被重排的文档列表。", example=["关于苹果的文档。", "讨论橙子的文档。", "关于各种话题的信息，包括相关的一个。"])
    top_k: Optional[int] = Field(None, description="返回前 N 个重排后的文档。如果为 None，则返回所有文档的重排结果。")

# class RerankedDocument(BaseModel):
#     text: str = Field(description="文档内容。")

class RerankResponse(BaseModel):
    reranked_documents: List[str] = Field(description="重排后的文档列表，按得分降序排列。")
    model_name: str = Field(description="所使用的重排模型名称。")


@app.on_event("startup")
async def startup_event():
    global model
    model_path = qwen_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        qwen_tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16
    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        load_format="auto",
        gpu_memory_utilization=0.5,
        # max_model_len=4608,
        # max_num_seqs=16,
        enforce_eager=False,
    )
    """
    启动 embedding and reranker
    """
    # global embeddings, reranker
    # print("正在加载模型...")
    # try:
    #     embeddings = BGEM3EmbeddingFunction(
    #         model_name="/hpc2hdd/home/fye374/models/BAAI/bge-m3",
    #         use_fp16=False,  # 根据您的模型和硬件调整
    #         device="cuda"    # 确保 CUDA 可用且设备 ID 正确
    #     )
    #     print("Embedding 模型加载成功。")

    #     reranker = BGERerankFunction(
    #         model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",
    #         device="cuda",   # 确保 CUDA 可用
    #         use_fp16=False   # 根据您的模型和硬件调整
    #     )
    #     print("Reranker 模型加载成功。")
    # except Exception as e:
    #     print(f"模型加载失败: {e}")
    #     # 您可以选择在这里抛出错误或允许应用启动但功能受限
    #     # raise RuntimeError(f"模型加载失败: {e}")



@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    gen = model.generate(body["prompts"], vllm_sampling_params)
    result = [geni.outputs[0].text.split("assistant")[-1].strip() for geni in gen]
    return result


# @app.post("/embed", response_model=EmbedResponse)
# async def get_embeddings(request: EmbedRequest):
#     """
#     接收文本列表并返回其嵌入向量。
#     """
#     if not embeddings:
#         raise HTTPException(status_code=503, detail="Embedding 模型尚未加载或加载失败。")
#     if not request.texts:
#         return EmbedResponse(embeddings=[], model_name=embeddings.model_name if hasattr(embeddings, 'model_name') else "bge-m3")

#     try:
#         # BGEM3EmbeddingFunction返回 List[List[float]]
#         print(11111)
#         encoded_vectors = embeddings(request.texts)
#         print(type(encoded_vectors['dense']))
#         print(type(encoded_vectors['sparse']))
#         return EmbedResponse(
#             embeddings=encoded_vectors,
#             model_name=getattr(embeddings, '_model_name', '/hpc2hdd/home/fye374/models/BAAI/bge-m3') # 获取模型名称
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"生成嵌入时出错: {str(e)}")

# @app.post("/rerank", response_model=RerankResponse)
# async def rerank_documents(request: RerankRequest):
#     """
#     接收查询和文档列表，返回重排后的文档列表。
#     """
#     if not reranker:
#         raise HTTPException(status_code=503, detail="Reranker 模型尚未加载或加载失败。")
#     if not request.documents:
#          return RerankResponse(reranked_documents=[], model_name=reranker.model_name if hasattr(reranker, 'model_name') else "bge-reranker-v2-m3")

#     try:
#         # BGERerankFunction 接受 query, documents, 和 top_k
#         # 如果 request.top_k 为 None，则 reranker 将使用其默认的 top_k 值（通常是5）
#         # 或者，如果想在 top_k 未指定时返回所有文档，可以这样设置：
#         effective_top_k = request.top_k if request.top_k is not None else len(request.documents)

#         # BGERerankFunction 返回 pymilvus.model.reranker.RankedDocument 对象列表
#         results = reranker(
#             query=request.query,
#             documents=request.documents,
#             top_k=effective_top_k
#         )

#         # 将结果转换为 Pydantic 模型
#         response_docs = [
#             doc.text
#             for doc in results
#         ]
#         return RerankResponse(
#             reranked_documents=response_docs,
#             model_name=getattr(reranker, '_model_name', "/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3") # 获取模型名称
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"重排文档时出错: {str(e)}")


# 如果你直接运行这个脚本来启动 FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18001)



##############################################################
### 更鲁棒性的写法如下
##############################################################
# import os
# # 建议在启动 Uvicorn 服务器之前，在您的 shell 环境中设置 CUDA_VISIBLE_DEVICES
# # 例如: export CUDA_VISIBLE_DEVICES='1'
# # 此处保留您原始脚本中的设置，它会在模块首次导入时执行。
# # 如果 PyTorch 或相关库在此行之前已被其他模块导入，则可能无效。
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any

# # 从您的原始代码中导入模型和类型
# from milvus_model.hybrid import BGEM3EmbeddingFunction
# from pymilvus.model.reranker import BGERerankFunction
# # pymilvus.model.reranker.RankedDocument 用于类型提示，但 BGERerankFunction 会直接返回它
# # from pymilvus.model.reranker import RankedDocument as MilvusRankedDocument # 用于显式类型提示

# # 全局变量用于存储加载的模型
# embeddings: Optional[BGEM3EmbeddingFunction] = None
# reranker: Optional[BGERerankFunction] = None

# app = FastAPI(title="Embedding and Reranking API", version="1.0.0")

# @app.on_event("startup")
# async def startup_event():
#     """
#     FastAPI 启动时加载模型。
#     """
#     global embeddings, reranker
#     print("正在加载模型...")
#     try:
#         embeddings = BGEM3EmbeddingFunction(
#             model_name="/hpc2hdd/home/fye374/models/BAAI/bge-m3",
#             use_fp16=False,  # 根据您的模型和硬件调整
#             device="cuda"    # 确保 CUDA 可用且设备 ID 正确
#         )
#         print("Embedding 模型加载成功。")

#         reranker = BGERerankFunction(
#             model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",
#             device="cuda",   # 确保 CUDA 可用
#             use_fp16=False   # 根据您的模型和硬件调整
#         )
#         print("Reranker 模型加载成功。")
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         # 您可以选择在这里抛出错误或允许应用启动但功能受限
#         # raise RuntimeError(f"模型加载失败: {e}")

# # --- Pydantic 模型定义 ---

# class EmbedRequest(BaseModel):
#     texts: List[str] = Field(..., description="需要生成嵌入的文本列表。", example=["你好世界", "这是一个测试文档"])

# class EmbedResponse(BaseModel):
#     embeddings: List[List[float]] = Field(description="生成的嵌入向量列表。")
#     model_name: str = Field(description="所使用的嵌入模型名称。")

# class RerankRequest(BaseModel):
#     query: str = Field(..., description="用于重排的查询。", example="相关话题")
#     documents: List[str] = Field(..., description="需要被重排的文档列表。", example=["关于苹果的文档。", "讨论橙子的文档。", "关于各种话题的信息，包括相关的一个。"])
#     top_k: Optional[int] = Field(None, description="返回前 N 个重排后的文档。如果为 None，则返回所有文档的重排结果。")

# class RerankedDocument(BaseModel):
#     text: str = Field(description="文档内容。")
#     score: float = Field(description="文档的重排得分。")
#     index: int = Field(description="文档在原始列表中的索引。")

# class RerankResponse(BaseModel):
#     reranked_documents: List[RerankedDocument] = Field(description="重排后的文档列表，按得分降序排列。")
#     model_name: str = Field(description="所使用的重排模型名称。")

# # --- API 端点 ---

# @app.post("/embed", response_model=EmbedResponse)
# async def get_embeddings(request: EmbedRequest):
#     """
#     接收文本列表并返回其嵌入向量。
#     """
#     if not embeddings:
#         raise HTTPException(status_code=503, detail="Embedding 模型尚未加载或加载失败。")
#     if not request.texts:
#         return EmbedResponse(embeddings=[], model_name=embeddings.model_name if hasattr(embeddings, 'model_name') else "bge-m3")

#     try:
#         # BGEM3EmbeddingFunction 的 encode_documents 方法返回 List[List[float]]
#         encoded_vectors = embeddings.encode_documents(request.texts)
#         return EmbedResponse(
#             embeddings=encoded_vectors,
#             model_name=getattr(embeddings, '_model_name', '/hpc2hdd/home/fye374/models/BAAI/bge-m3') # 获取模型名称
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"生成嵌入时出错: {str(e)}")

# @app.post("/rerank", response_model=RerankResponse)
# async def rerank_documents(request: RerankRequest):
#     """
#     接收查询和文档列表，返回重排后的文档列表。
#     """
#     if not reranker:
#         raise HTTPException(status_code=503, detail="Reranker 模型尚未加载或加载失败。")
#     if not request.documents:
#          return RerankResponse(reranked_documents=[], model_name=reranker.model_name if hasattr(reranker, 'model_name') else "bge-reranker-v2-m3")

#     try:
#         # BGERerankFunction 接受 query, documents, 和 top_k
#         # 如果 request.top_k 为 None，则 reranker 将使用其默认的 top_k 值（通常是5）
#         # 或者，如果想在 top_k 未指定时返回所有文档，可以这样设置：
#         effective_top_k = request.top_k if request.top_k is not None else len(request.documents)

#         # BGERerankFunction 返回 pymilvus.model.reranker.RankedDocument 对象列表
#         results = reranker(
#             query=request.query,
#             documents=request.documents,
#             top_k=effective_top_k
#         )

#         # 将结果转换为 Pydantic 模型
#         response_docs = [
#             RerankedDocument(text=doc.text, score=doc.score, index=doc.index)
#             for doc in results
#         ]
#         return RerankResponse(
#             reranked_documents=response_docs,
#             model_name=getattr(reranker, '_model_name', "/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3") # 获取模型名称
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"重排文档时出错: {str(e)}")

# @app.get("/health")
# async def health_check():
#     """
#     健康检查端点，确认服务是否正在运行以及模型是否已加载。
#     """
#     return {
#         "status": "ok",
#         "embedding_model_loaded": embeddings is not None,
#         "reranker_model_loaded": reranker is not None,
#         "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
#     }

# # --- 运行 FastAPI 应用的说明 (在脚本末尾的注释中) ---
# # 要运行此 FastAPI 应用:
# # 1. 将代码保存为 `main.py`。
# # 2. 确保您的环境中安装了必要的库:
# #    pip install fastapi uvicorn python-multipart pymilvus sentence-transformers torch
# #    (根据 BGEM3EmbeddingFunction 和 BGERerankFunction 的具体依赖，可能还需要其他库)
# # 3. 在您的 shell 中设置 CUDA 设备 (强烈建议):
# #    export CUDA_VISIBLE_DEVICES='1'
# # 4. 使用 Uvicorn 运行应用:
# #    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# #    (`--reload` 选项用于开发，生产中请移除)