import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
import ALL_KEYS
openai_model = "deepseek-v3-250324"
openai_api_key = ALL_KEYS.common_openai_key
openai_api_base = ALL_KEYS.common_openai_base_url
llm_type = "ollama" #openai ollama qwen

generator_model = "qwen"
qwen_model_path = '/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/models/qwen2.5-14b-qlora-gaokao-5244'


ollama_model = "qwen2.5:72b"
ollama_num_ctx = 25600
