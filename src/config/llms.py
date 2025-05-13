import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
import ALL_KEYS
from vllm import SamplingParams
openai_model = "deepseek-v3-250324"
openai_api_key = ALL_KEYS.common_openai_key
openai_api_base = ALL_KEYS.common_openai_base_url
llm_type = "ollama" #openai ollama qwen gemini

outer_knowledge_llm_type = "gemini"
planner_llm_type = "gemini"
reporter_llm_type = "gemini"
supervisor_llm_type = "gemini"
gemini_model = "gemini-2.0-flash"
# gemini_model = 'gemini-2.5-flash-preview-04-17-nothinking'
gemini_api_key = ALL_KEYS.common_openai_key
gemini_api_base = ALL_KEYS.common_openai_base_url

generator_model = "qwen" #gemini qwen
# qwen_model_path = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-14b-qlora-gaokao-21699'
# qwen_model_path = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-7b-lora-gaokao-21699'
qwen_model_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-7b-lora-gaokao-60265"
qwen_tokenizer_path = "/hpc2hdd/home/fye374/models/Qwen/Qwen2.5-7B-Instruct"

vllm_sampling_params = SamplingParams(
    temperature=0.1,
    top_p=1,
    max_tokens=1024,
    # num_beams=1,    # 如需 beam search 可加
)

ollama_model = "qwen3:30b"
ollama_num_ctx = 25000


# eval_model = "qwen3:32b"
eval_model = "llama3.1:70b"
eval_llm_type = "ollama" #openai ollama hkust

# ollama_num_ctx = 25600
# eval_llm_type = "hkust" #openai ollama hkust