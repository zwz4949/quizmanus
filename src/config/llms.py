import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
import ALL_KEYS
from vllm import SamplingParams
openai_model = "deepseek-v3-250324"
openai_api_key = ALL_KEYS.common_openai_key
openai_api_base = ALL_KEYS.common_openai_base_url
llm_type = "ollama" #openai ollama qwen

generator_model = "qwen"
qwen_model_path = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-14b-qlora-gaokao-21699'
qwen_tokenizer_path = "/hpc2hdd/home/fye374/models/Qwen/Qwen2.5-14B-Instruct"

vllm_sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    # num_beams=1,    # 如需 beam search 可加
)

ollama_model = "qwen2.5:72b"
ollama_num_ctx = 25600


# eval_model = "qwen3:32b"
eval_model = "llama3.1:70b"
# ollama_num_ctx = 25600
eval_llm_type = "ollama" #openai ollama hkust