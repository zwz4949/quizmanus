import time
import json
import os
import re
from typing import List, Dict, Union
from tqdm import tqdm




import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from utils import (
    getData,
    saveData,
    get_absolute_file_paths,
    get_json_result,
    removeDuplicates,
    call_Hkust_api
)


train_data = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/merge.json")
# test_data = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/规范data/final_data/test.json")
# print(len(test_data))


prompt_template =  """请仔细阅读以下文本内容，该内容来自人教版{subject}{grade}教材。你的任务是清理内容，仅保留与学科知识直接相关的部分。需要删除的内容包括但不限于：
- 练习题、思考题、例题
- 学习目标、章节小结
- 与知识点无关的示例、故事、插图说明
- 其他非知识性的辅助内容

如果内容全部为有用知识，保留原样；如果部分有用，提取有用部分并整理成通顺文本；如果全部无用，返回空字符串: ""。

请以严格JSON格式返回处理后的内容，但不要包含```json和```，格式为：
{
    "content": "..."
}


待处理内容的伪标题：
{title}
待处理内容：
{content}
输出："""



def get_response(example,prompt_template):
    try: 
        # [dict_[example['type']]]
        pr = prompt_template.replace("{subject}",example['subject'].strip()).replace("{grade}",str(example['grade']).strip()).replace("{content}",example['content'].strip()).replace("{title}",example['title'].strip())
        res = call_Hkust_api(pr)
        context = get_json_result(res)['content']
        return {**example,"modify_content":context}
    except Exception as e:
        print(e)
        return {**example,"modify_content":"-1"}



    
import threading
import os
from tqdm import tqdm
import json
import copy


def fromGPTQuestionGenerateIntentAndSaveEveryone(
        data,
        file_path,
        prompt_template,
        have_change_ids,
        thread_count=0,
        current_thread_num=0,
        is_multi_thread=True,
        bar=None):
    if not bar:
        bar = tqdm(total=len(data))
    with open(file_path, "a", encoding="utf-8") as output_file:
        for index, data_item in enumerate(data):
            if is_multi_thread:
                if index % thread_count != current_thread_num:
                    continue
                if data_item['id'] in have_change_ids:
                    bar.update(1)  # 更新进度条
                    continue
            response_dict = get_response(data_item,prompt_template)
            
            output_line = json.dumps(response_dict, ensure_ascii=False) + "\n"
            output_file.write(output_line)
            output_file.flush()  # 立即刷新文件缓冲区，使得写入的内容可见
            bar.update(1)  # 更新进度条
    if not is_multi_thread:
        saveData(removeDuplicates(getData(file_path)), file_path)


def multiThreadGenerateIntentAndSaveEveryone(
        data,
        file_path,
        prompt_template,
        have_change_ids,
        is_multi_thread=True,
        thread_count=None):
    '''
    '''
    if thread_count is None:
        thread_count = os.cpu_count()  # 自动选择线程数
        print("线程数",thread_count)

    assert file_path.endswith(".jsonl"), "file_path must end with .jsonl"
    bar = tqdm(total=len(data))  # 创建一个进度条，总计data长度
    
    # 创建线程列表
    threads = []
    
    # 创建并启动thread_count个线程
    for i in range(thread_count):
        # 创建线程
        thread = threading.Thread(target=fromGPTQuestionGenerateIntentAndSaveEveryone, args=(
            data,
            file_path,
            prompt_template,
            have_change_ids,
            thread_count,
            i,
            is_multi_thread,
            bar))
        threads.append(thread)
        # 启动线程
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()
    bar.close()  # 关闭进度条

    print("All threads have finished.")


def process(data,path):
    # path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/规范data/train.jsonl"
    import re
    have_change_ids = []
    not_change_ids = []
    while_i = 0
    while(len(data)!= len(have_change_ids) and while_i < 5):
        while_i+=1
        have_change_ids = []
        if os.path.exists(path):
            saveData(
                removeDuplicates(
                    getData(path)),
                path
            )
            generated_data = getData(path)
            for item in generated_data:
                if 'modify_content' in item and isinstance(item['modify_content'],str) and item['modify_content']!="-1":
                    have_change_ids.append(item['id'])
            not_change_ids = list(set([item['id'] for item in data])-set(have_change_ids))
        print(len(data))# train_data是list[dict]
        print(len(have_change_ids))
        multiThreadGenerateIntentAndSaveEveryone(
            data,
            path,
            prompt_template,
            have_change_ids,
            # thread_count= 1,
        )
        saveData(
            removeDuplicates(
                getData(path)),
            path
        )
        saveData(
            getData(path),
            path[:-1]
        )

process(train_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/去无用_merge.jsonl")

# process(test_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/hkust/data加课本课外内容/test.jsonl")