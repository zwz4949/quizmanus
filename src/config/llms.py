import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus")
import ALL_KEYS
openai_model = "deepseek-v3-250324"
openai_api_key = ALL_KEYS.common_openai_key
openai_api_base = ALL_KEYS.common_openai_base_url
llm_type = "openai" #openai ollama qwen

generator_model = "qwen"

ollama_model = "qwen2.5:72b"
