# 高考题目生成模型 - SFT 纯微调

本目录包含使用 Supervised Fine-Tuning (SFT) 方法对 Qwen 大语言模型进行微调，以生成高考风格试题的训练和推理脚本。

## 文件说明

### 1. `train_qwen_gaokao.py`

*   **作用**: 该脚本用于对预训练的 Qwen 模型（如 Qwen2.5-14B-Instruct）进行监督微调。它使用 `gaokao_data` 中的数据，学习根据提供的“课本内容”和“课外内容”生成指定科目和类型的“题干”、“参考答案”和“解析”。
*   **核心流程**:
    1.  加载预训练的 Qwen 模型和对应的 Tokenizer。
    2.  使用 `BitsAndBytesConfig` 进行 4-bit 量化加载模型，以节省显存。
    3.  配置 LoRA (Low-Rank Adaptation) 参数进行参数高效微调 (PEFT)。
    4.  加载 `gaokao_data` 中的训练数据集（脚本中当前加载的是 `test.json`，实际训练应使用 `train.json`）。
    5.  定义 `format_chat` 函数，将原始数据格式化为 Qwen 模型期望的对话格式。
    6.  配置 `SFTConfig` 训练参数，如训练轮数、批次大小、学习率、优化器、保存策略等。
    7.  使用 `trl` 库的 `SFTTrainer` 初始化训练器。
    8.  执行 `trainer.train()` 开始模型微调。
    9.  保存微调后的 LoRA 权重和 Tokenizer 到指定的输出目录。
*   **关键技术点**:
    *   **Supervised Fine-Tuning (SFT)**: 使用标注好的问答对数据对模型进行微调。
    *   **Parameter-Efficient Fine-Tuning (PEFT)**: 利用 LoRA 技术，只训练少量参数，提高训练效率。
    *   **Quantization (量化)**: 使用 4-bit (NF4) 量化加载模型，降低显存占用。
    *   **Hugging Face Transformers**: 使用 `AutoModelForCausalLM`, `AutoTokenizer` 加载模型和分词器。
    *   **Hugging Face TRL (Transformer Reinforcement Learning)**: 使用 `SFTTrainer` 和 `SFTConfig` 简化 SFT 流程。
    *   **Hugging Face Datasets**: 加载 JSON 格式的数据集。
    *   **Chat Templates**: 使用 `tokenizer.apply_chat_template` 和自定义的 `format_chat` 函数处理对话格式数据。
    *   **Mixed Precision Training (混合精度训练)**: `fp16=True` 启用 FP16 训练加速。

### 2. `infer_qwen_gaokao.py`

*   **作用**: 该脚本用于加载经过 `train_qwen_gaokao.py` 微调后的模型（基础模型 + LoRA 权重），并根据新的输入（课本内容、课外内容、科目、题型）生成高考题目。
*   **核心流程**:
    1.  加载基础 Qwen 模型（如 Qwen2.5-14B-Instruct），同样使用 4-bit 量化。
    2.  加载对应的 Tokenizer，并设置 `padding_side='left'` 以适应生成任务。
    3.  使用 `PeftModel.from_pretrained` 加载训练好的 LoRA 权重并合并到基础模型中。
    4.  加载 `gaokao_data` 中的测试数据集。
    5.  定义 `prepare_input` 函数，将测试数据的输入部分格式化为模型推理所需的 Prompt 格式（使用 Chat Template）。
    6.  设置推理参数，如 `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `repetition_penalty` 等。
    7.  使用 `torch.inference_mode()` 进行优化。
    8.  以指定的 `batch_size` 批量进行推理 (`model.generate`)。
    9.  解码生成的结果，提取模型输出的题目内容。
    10. （可选）使用 `evaluate.getTotalScore` 评估生成结果与真实标签的相似度。
    11. 将输入、Prompt 和生成结果保存到 JSON 文件中。
*   **关键技术点**:
    *   **PEFT Model Loading**: 加载 LoRA 权重与基础模型结合进行推理。
    *   **Text Generation**: 使用 `model.generate` 方法生成文本。
    *   **Batch Inference**: 批量处理输入数据以提高推理效率。
    *   **Quantization (量化)**: 加载量化后的基础模型。
    *   **Chat Templates**: 使用 `tokenizer.apply_chat_template` 准备推理输入。
    *   **Generation Parameters**: 控制生成过程的参数（采样、温度、Top-p 等）。
    *   **Flash Attention 2**: （如果硬件和环境支持）自动启用以加速注意力计算。
    *   **Torch Compile**: （如果 PyTorch 版本支持）使用 `torch.compile` 优化模型推理速度。
    *   **KV Caching**: `use_cache=True` 在生成过程中使用 KV 缓存加速。

## 使用

1.  **训练**: 修改 `train_qwen_gaokao.py` 中的路径和配置参数，然后运行脚本开始训练。
    ```bash
    python train_qwen_gaokao.py
    ```
2.  **推理**: 修改 `infer_qwen_gaokao.py` 中的模型路径、LoRA 路径和数据路径，然后运行脚本进行推理。
    ```bash
    python infer_qwen_gaokao.py
    ```

**注意**: 脚本中的文件路径（例如 `/hpc2hdd/home/fye374/...`）是特定于开发环境的。请根据您的实际环境修改 `data_root_path`（数据路径）、`model_root_path`（模型路径）、`output_dir`（输出目录）以及 `lora_path`（LoRA 权重路径）等变量。