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


train_data = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/1.去无用_merge.json")
# test_data = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/规范data/final_data/test.json")
# print(len(test_data))
# prompt_template = {
# "主观题":'''# 角色说明
# - 你是一个根据题目、答案和分析过程写课本内容的专家，给定一道高考题，请将他涉及的课本知识点写成若干段上下文
# - 要求内容详细，覆盖全面，主要是根据答案和解析来生成，题目作为辅助
# - 要求风格就像是课本里摘抄出来的一样，并且以json格式返回给我： { "context":"..." }

# # 说明
# - 始终保持使用用户相同的语言

# 问题: {question}

# 答案: {answer}

# 解析: {analysis}

# 输出：''',
# "选择题":'''# 角色说明
# - 你是一个根据题目、答案和分析过程写教材内容的专家，给定一道高考题，请将他涉及的知识点写成一段上下文
# - 要求风格就像是课本里摘抄出来的一样，并且以json格式返回给我： { "context":"..." }

# # 说明
# - 始终保持使用用户相同的语言

# # 例子
# 问题: 2．（ 6分）下列关于呼吸作用的叙述，正确的是（ 　　） \nA．无氧呼吸的终产物是丙酮酸   \nB．有氧呼吸产生的 [H]在线粒体基质中与氧结合生成水   \nC．无氧呼吸不需要 O2的参与．该过程最终有 [H]的积累   \nD．质量相同时，脂肪比糖原有氧氧化释放的能量多

# 答案: D

# 解析: 解： A、无氧呼吸的终产物是二氧化碳和酒精，或者分解成乳酸（有机\n物不彻底分解），而不是丙酮酸， A错误；  \nB、有氧呼吸前两个阶段产生的 [H]在线粒体内膜上与氧结合生成水， B错误；  \nC、无氧呼吸不需要 O2的参与。但该过程中没有 [H]的积累， [H]只在细胞质基\n质中参与反应， C错误；  \nD、质量相同的脂肪和糖原 ，脂肪贮存的能量更多 ，因此比糖原有氧氧化释放的\n能量多， D正确。   \n故选： D。

# 输出：
# {
#   "context": "细胞呼吸是生物体将有机物分解以释放能量的过程，主要分为有氧呼吸和无氧呼吸两种形式。有氧呼吸包括三个阶段：糖酵解、柠檬酸循环和电子传递链。其中，糖酵解在线粒体外的细胞质基质中进行，产物丙酮酸可进入线粒体参与后续反应。在有氧呼吸过程中，所产生的[H]主要在线粒体的基质中生成，但其与氧结合生成水的过程实际上在线粒体内膜上的电子传递链中进行。无氧呼吸则在缺氧条件下进行，其终产物为酒精和二氧化碳（如酵母菌）或乳酸（如动物肌细胞），而非丙酮酸，并且不会造成[H]的积累。此外，从单位质量来看，脂肪分子含有更多的C—H键，因此在有氧条件下完全氧化时能释放出比糖原更多的能量。这一特点使脂肪成为更高能量密度的储能物质。"
# }

# # 到你了
# 问题: {question}

# 答案: {answer}

# 解析: {analysis}

# 输出：'''
# }

# prompt_template = ['''# 角色说明
# - 你是一个根据题目、答案和分析过程写课本内容的专家，给定一道高考题，请将他涉及的课本知识点写成若干段上下文
# - 要求内容详细，覆盖全面，主要是根据答案和解析来生成，题目作为辅助
# - 要求风格就像是课本里摘抄出来的一样，并且以json格式返回给我： { "课本内容":"若干段课本内容上下文" }

# # 说明
# - 始终保持使用用户相同的语言

# 问题: {question}

# 答案: {answer}

# 解析: {analysis}

# 输出：''',

prompt_template = '''# 角色说明
你是一个根据课本内容找对应课外内容的专家，给定一段{subject}{grade}的课本内容，请将他涉及的课本知识点生成一段相关的课外内容。
- 利用你的知识生成出给定课本内容可以涉及的课外内容，要求内容详细，覆盖全面
- 生成的课外内容用于辅助学生理解和深刻课本内容

# 说明
- 始终保持使用用户相同的语言
- 要求内容详细，覆盖全面
- 要求课外内容风格像新闻或者学术材料的风格
- 输出格式为：
{ 
    "课外内容": "若干段课外内容上下文"
}
- 不要在输出包含```json和```

课本内容: {content}
输出：'''



def get_response(example,prompt_template):
    dict_ = {
        "单选题":"选择题",
        "多选题":"选择题",
        "主观题":"主观题"
    }
    try: 
        # [dict_[example['type']]]
        pr = prompt_template.replace("{content}",example['modify_content'].strip()).replace("{grade}",str(example['grade']).strip()).replace("{subject}",example['subject'].strip())
        res = call_Hkust_api(pr)
        context = get_json_result(res)['课外内容']
        return {**example,"课本内容":example['modify_content'].strip(),"课外内容":context.strip()}
    except Exception as e:
        print(e)
        return {**example,"课本内容":"","课外内容":""}



    
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
                if '课外内容' in item and isinstance(item['课外内容'],str) and item['课外内容'].strip() != "":
                    have_change_ids.append(item['id'])
            not_change_ids = list(set([item['id'] for item in data])-set(have_change_ids))
        print(len(data))# train_data是list[dict]
        print(len(have_change_ids))
        multiThreadGenerateIntentAndSaveEveryone(
            data,
            path,
            prompt_template,
            have_change_ids,
            thread_count= 256,
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

real_data = [item for item in train_data if item['modify_content'].strip()!=""]
process(real_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/2.加课外内容.jsonl")

# process(test_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/hkust/data加课本课外内容/test.jsonl")