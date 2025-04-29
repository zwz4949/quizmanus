import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src')
from utils import *

# 设置路径
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_root_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/gaokao_data"
# model_root_path = "/hpc2hdd/home/fye374/models/deepseek-ai"
# module_name = "DeepSeek-R1-Distill-Qwen-14B"
model_root_path = "/hpc2hdd/home/fye374/models/Qwen"
module_name = "Qwen2.5-14B-Instruct"
output_dir = f"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/results/{module_name}/train_{current_time}"
# lora_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/results/DeepSeek-R1-Distill-Qwen-14B/train_2025-04-21_00-58-33/checkpoint-3216"
lora_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/results/Qwen2.5-14B-Instruct/train_2025-04-21_14-27-30/checkpoint-3216"


sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from evaluate import getTotalScore

def prepare_input(example, tokenizer):
    """准备模型输入"""
    SYSTEM_PROMPT = '''# 角色说明
你是一个根据课本内容和课外内容生成{type}的专家，给定一段{subject}的课本内容和一段相关的课外内容，请根据他们生成一道高考{type}。

# 回答格式
题干；...
参考答案：...
解析：...'''
    
    messages=[
        {'role':'system','content':SYSTEM_PROMPT.format(type=example['type'], subject=example['subject'])}, 
        {'role':'user','content': f"课本内容：{example['课本内容']}\n课外内容：{example['课外内容']}"}, 
        {'role':'assistant','content': ''}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        # continue_final_message=True
    )
    return prompt.replace("\n<|im_end|>\n",'')+"\n"

def main():
    # 1. 使用更快的tokenizer设置
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path, 
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
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        kwargs["attn_implementation"] = "flash_attention_2"
    compute_dtype = getattr(torch, "float16") 
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_root_path, module_name),
        quantization_config=quant_config,
        **kwargs
    )
    
    # 4. 调整模型以适应新的tokenizer大小
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 5. 加载LoRA权重
    try:
        model = PeftModel.from_pretrained(
            base_model, 
            lora_path,
            torch_dtype=compute_dtype
        )
        print(f"成功加载LoRA权重: {lora_path}")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
        print("使用基础模型进行推理")
        model = base_model
    
    # 6. 编译模型 (PyTorch 2.0+特性)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
    # 设置为评估模式
    model.eval()
    
    # 7. 批量处理数据 (如果显存允许)
    batch_size = 4  # 根据GPU显存调整
    
    test_data_path = os.path.join(data_root_path, "test.json")
    test_data = getData(test_data_path)[:8]
    print(f"加载了 {len(test_data)} 条测试数据")
    
    results = []
    pred = []
    label = []
    
    # 8. 使用torch.inference_mode()替代torch.no_grad()以获得额外优化
    with torch.inference_mode():
        # 9. 分批处理数据
        for i in tqdm(range(0, len(test_data), batch_size), desc="推理中"):
            batch = test_data[i:i+batch_size]
            
            # 准备批量输入
            batch_inputs = [prepare_input(example, tokenizer) for example in batch]
            
            # 批量编码
            inputs = tokenizer(
                batch_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048  # 根据模型最大长度调整
            ).to(model.device)
            
            # 批量生成
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,  # 启用KV缓存
                repetition_penalty=1.1,  # 避免重复生成
            )
            
            # 解码批量结果
            for j in range(len(batch)):
                completion_ids = generated_ids[j][len(inputs.input_ids[j]):]
                generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                result = {
                    "input": batch[j],
                    "prompt": batch_inputs[j],
                    "generated": generated_text
                }
                pred.append(generated_text)
                label.append(f'''问题：{batch[j]['question']}
参考答案：{batch[j]['answer']}
解析：{batch[j]['analysis']}''')
                results.append(result)
                
                # 打印一些示例结果
                if len(results) <= 2:
                    print("\n" + "="*50)
                    print("输入上下文:", batch[j].get('context', '')[:100] + "...")
                    print("生成结果:", generated_text[:100] + "...")
                    print("="*50)
    
    print(getTotalScore(pred, label))
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, "inference_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"推理完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    main()