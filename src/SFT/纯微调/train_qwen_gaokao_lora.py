import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src')
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils import *
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import subprocess

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设定随机种子
seed = 42
set_seed(seed)


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# data_root_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/gaokao_data"
data_root_path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/gaokao_data/final_data"
model_root_path = "/hpc2hdd/home/fye374/models/Qwen"
module_name = "Qwen2.5-7B-Instruct"
# model_root_path = "/hpc2hdd/home/fye374/models/deepseek-ai"
# module_name = "DeepSeek-R1-Distill-Qwen-7B"
output_dir = f"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/SFT/纯微调/results/{module_name}/train_{current_time}"



import torch
from trl import SFTTrainer
import re
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.tokenizer = self.processing_class  # 确保可以访问 tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        # 获取 <think> 和 </think> 的 token ID
        think_start_id = self.processing_class.convert_tokens_to_ids('<think>')
        think_end_id = self.processing_class.convert_tokens_to_ids('</think>')

        # 创建一个 mask，标记需要忽略的 token
        ignore_mask = torch.zeros_like(labels, dtype=torch.bool)

        for i in range(labels.size(0)):
            in_think = False
            for j in range(labels.size(1)):
                token_id = input_ids[i, j].item()
                if token_id == think_start_id:
                    in_think = True
                    ignore_mask[i, j] = True
                elif token_id == think_end_id:
                    ignore_mask[i, j] = True
                    in_think = False
                elif in_think:
                    ignore_mask[i, j] = True

        # 将需要忽略的 token 的 label 设置为 -100
        labels = labels.masked_fill(ignore_mask, -100)
        inputs["labels"] = labels

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)





def main():
    # 1. 加载模型和tokenizer
    
    # 初始化tokenizer并添加特殊字符
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root_path,module_name), trust_remote_code=True)
    kwargs = {
        # "trust_remote_code": True,
        # "device_map": "auto",
        # "torch_dtype": compute_dtype,
    }
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        kwargs["attn_implementation"] = "flash_attention_2"

    # 确保tokenizer有正确的padding设置
    tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = getattr(torch, "bfloat16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=compute_dtype,
        # bnb_4bit_use_double_quant=True,
    )
    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_root_path,module_name),
        # quantization_config=quant_config,
        torch_dtype=torch.bfloat16,     # 或者 torch.bfloat16
        trust_remote_code=True,
        device_map="auto",
        **kwargs
    )
    
    # 调整模型以适应新的tokenizer大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. 配置LoRA (如果需要进行参数高效微调)
    # peft_config = LoraConfig(
    #     r=16,  # LoRA的秩
    #     lora_alpha=32,  # LoRA的alpha参数
    #     lora_dropout=0.05,  # LoRA的dropout率
    #     bias="none",  # 是否对偏置项进行微调
    #     task_type="CAUSAL_LM",  # 任务类型
    #     # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 要应用LoRA的模块
    # )
    peft_config = LoraConfig(
        lora_alpha = 16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # 4. 准备训练数据
    # 这里使用示例数据集，您可以替换为自己的数据集
    # data = getData(f"{data_root_path}/train.json")
    # dataset = Dataset.from_list(data)
    dataset = load_dataset("json", data_files=f"{data_root_path}/gaokao_ds_align_(混合_cn_en_40000)_3584.json")['train']
    print(dataset[0])
    # 5. 配置训练参数
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=10,
        # save_steps=100,
        # save_total_limit=3,
        save_strategy="epoch",
        bf16=True,  # 使用混合精度训练
        max_seq_length=3584,  # 最大序列长度
        # packing=True,  # 启用序列打包以提高效率
        # dataset_text_field="text",  # 数据集中文本字段的名称
        disable_tqdm= False
    )
    
    # 6. 初始化SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,  # 如果不需要LoRA，可以设为None
        processing_class=tokenizer,
        formatting_func=lambda x: format_chat(x, tokenizer),  # 自定义格式化函数，见下方定义
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("训练完成！模型已保存到", training_args.output_dir)

# 自定义格式化函数，用于将数据集中的对话格式化为Qwen模型期望的格式
def format_chat(example,tokenizer):
    messages=[
            {'role':'system','content':example['system']}, 
            {'role':'user','content': example['instruction']}, 
            {'role':'assistant','content': example['response']}
        ]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    
    # 如果数据集不是对话格式，直接返回文本
    return prompt

if __name__ == "__main__":
    main()