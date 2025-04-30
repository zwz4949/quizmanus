import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

# 提示模板（可根据需要调整）
PROMPT_TEMPLATE = """请仔细阅读以下文本内容，该内容来自高中教材。你的任务是清理内容，仅保留与学科知识直接相关的部分。需要删除的内容包括但不限于：
- 练习题、思考题、例题
- 学习目标、章节小结
- 与知识点无关的示例、故事、插图说明
- 其他非知识性的辅助内容

如果内容全部为有用知识，保留原样；如果部分有用，提取有用部分并整理成通顺文本；如果全部无用，返回空字符串。

请以严格JSON格式返回处理后的内容，格式为：{{"content": "..."}}

待处理内容：
{content}"""

def get_json_text(text):
    """从文本中提取第一个JSON对象"""
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        return match.group(0) if match else None
    except Exception:
        return None

def process_content(content):
    """处理单个内容并返回清理后的结果"""
    try:
        prompt = PROMPT_TEMPLATE.format(content=content)
        response = call_Hkust_api(prompt)  # 确保这是线程安全的API调用
        json_str = get_json_text(response)
        
        if json_str:
            result = json.loads(json_str)
            return result.get('content', '')
        return ''
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return content  # 失败时返回原内容（可根据需求修改）

def process_file(json_file, max_workers=100):
    """处理单个JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    contents = [item['content'] for item in data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_content, content) for content in contents]
        processed_contents = []
        
        for idx, future in enumerate(futures):
            try:
                processed_contents.append(future.result())
            except Exception as e:
                print(f"Error in file {json_file} content {idx}: {str(e)}")
                processed_contents.append(contents[idx])  # 保留原内容作为保底
    
    # 更新并保存结果
    for item, new_content in zip(data, processed_contents):
        item['content'] = new_content
    
    new_filename = f"{os.path.splitext(json_file)[0]}_processed.json"
    with open(new_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_absolute_file_paths(absolute_dir, file_type) -> List[str]:
    return [os.path.join(absolute_dir, f) 
            for f in os.listdir(absolute_dir) 
            if f.endswith(f".{file_type}")]

def main(absolute_dir, max_workers=100):
    json_files = get_absolute_file_paths(absolute_dir, "json")
    for file in json_files:
        print(f"Processing {file}...")
        process_file(file, max_workers)

if __name__ == "__main__":
    # 使用示例
    ABSOLUTE_DIR = "/path/to/your/json/files"  # 替换为实际路径
    MAX_WORKERS = 100  # 根据API承受能力调整
    
    main(ABSOLUTE_DIR, MAX_WORKERS)