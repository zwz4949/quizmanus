import time
import json
import os
import re
from typing import List, Dict, Union
from tqdm import tqdm




import sys
# sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from src.utils import (
    getData,
    saveData,
    get_absolute_file_paths,
    get_json_result,
    removeDuplicates,
    call_Hkust_api
)
# from src.utils import call_Hkust_api,getData,get_json_result
from src.config.rag import DETIALED_SUBJECTS
from tqdm import tqdm

prompt_template = '''一位老师/学生提出了一个需求：{query}。因此我们创建了一套测验题，其中包含若干道题目。请利用你的自身知识以及给定的学科目录概括，根据以下标准对该测验题的质量进行评估，并为整套题目给出1到5分的评分。

# 学科目录概括：

{catalog}

# 评估标准如下：

1. 教育价值：你认为这些测验题有教育意义吗？学生通过完成这些测验题是否能学到更多？
    - 1：完全没有教育意义，无学习价值。
    - 2：教育意义很小，学习价值有限。
    - 3：中等教育意义，有一定的学习价值。
    - 4：教育意义很强，学习价值高。
    - 5：教育意义极高，学习价值非常大。

2. 多样性：你认为这些测验题是否多样化？测验题是否涵盖了广泛的主题，还是都集中在同一个概念上？
    - 1：非常重复，覆盖范围狭窄。
    - 2：有一定多样性，但主要集中在一个概念上。
    - 3：较为多样，覆盖了几个不同主题。
    - 4：相当多样，涵盖了多个相关主题。
    - 5：极其多样，覆盖了广泛的主题。

3. 主题相关性：这些测验题是否与老师/学生的需求相关？测验题是否针对所学的学科领域（可参考学科目录概括）进行了定制？
    - 1：与需求或学科完全无关。
    - 2：相关性很小，与需求/学科有些许联系。
    - 3：中等相关，与需求/学科较为一致。
    - 4：高度相关，与需求/学科紧密结合。
    - 5：完全相关，直接关联需求/学科。

4. 难度适宜性：你认为这些测验题是否难度分布是否合理（满足简单题数量：中等偏上难度题数量 = 7：3）？是否符合老师/学生的需求？
    - 1：太简单或太难，完全不适合该水平。
    - 2：稍有偏差，测验题可能太简单或太难。
    - 3：中等适宜，测验题在一定程度上符合该水平。
    - 4：基本适宜，测验题非常适合该水平。
    - 5：完全适合学生的教育水平。

5. 全面性：这些测验题是否涵盖了主题的深度和广度？是否全面涉及了关键概念和细节？
    - 1：非常浅显，仅触及主题表面。
    - 2：有些不完整，遗漏了重要方面。
    - 3：中等全面，涵盖了基础知识但缺乏深度。
    - 4：相当全面，涵盖了大多数关键方面并有合理深度。
    - 5：高度全面，深入且详细地覆盖了主题。

6. 规范性：这些测验题是否按照`单选题->多选题->大题`的顺序来排列（不包含某一类题也可以，只要存在的题目按照该顺序即可）？是否每道题都包含了题干、参考答案和解析且每道题的题干、参考答案和解析都相对应？
    - 1：完全不规范：题目顺序杂乱无章，未按单选题→多选题→大题排列；题干、参考答案或解析缺失严重，且不对应。
    - 2：规范性较差：部分题目顺序符合要求，但存在明显错乱；部分题目缺少参考答案或解析，或对应关系不清晰。
    - 3：中等规范：题目顺序基本符合要求，但存在少量偏差；大部分题目包含题干、参考答案和解析，但个别题目可能不完整或对应不准确。
    - 4：较为规范：题目顺序完全符合要求；所有题目均包含题干、参考答案和解析，且基本对应，仅个别细节不够完善。
    - 5：极其规范：严格按单选题→多选题→大题顺序排列；每道题均包含完整且对应的题干、参考答案和解析，无任何遗漏或错误。

7. 题目数量及题型范围准确性：测验题的数量或者每一类题型的数量是否符合用户的需求中所述？题目类型是否都在["单选题","多选题","大题"]范围内？注意：题目类型在每道题的最前面用[中括号]框住，且大题不等于简答题。
    - 1：完全不符合：题目数量或题型数量与用户需求严重不符；题目类型超出 ["单选题", "多选题", "大题"] 范围，或未用 [中括号] 标注题型。
    - 2：部分符合但问题较多：题目数量或题型数量与用户需求有较大偏差；个别题目类型错误或未标注 [中括号]。
    - 3：基本符合但有少量偏差：题目数量或题型数量大致符合用户需求，但存在少量误差（如某类题型多或少1-2道）；所有题目类型均在范围内且标注正确。
    - 4：高度符合，仅个别细节问题：题目数量及题型数量完全或几乎完全符合用户需求；所有题目类型正确且标注规范，仅个别题目格式有小瑕疵。
    - 5：完全符合且完美执行：题目数量、题型数量均严格匹配用户需求；所有题目类型均在 ["单选题", "多选题", "大题"] 范围内，且每道题均用 [中括号] 正确标注。

8. 选择题选项准确性：单选题和多选题的选项个数是否都为4个？是否都用A B C D四个大写字母标号每个选项？单选题的参考答案是否都是一个？多选题的参考答案是否都是2到4个？
    - 1：完全不准确：单选题或多选题的选项数量不符合4个，或未用A/B/C/D标号；参考答案格式错误（如单选题多选、多选题少于2个答案）。
    - 2：准确性较差：部分选择题选项数量或标号错误；参考答案存在明显问题（如单选题有多个答案，多选题答案不足）。
    - 3：中等准确：大部分选择题符合要求，但个别题目选项数量或标号错误；参考答案基本正确，但存在少量偏差。
    - 4：较为准确：所有选择题选项均为4个且标号正确；参考答案基本符合要求，仅极个别题目存在小问题。
    - 5：极其准确：完全符合标准——单选题和多选题均为4个选项且标号A/B/C/D；单选题参考答案唯一，多选题参考答案2-4个，无任何错误。

9. 题目完整性和严谨性：是否每道题的题目都是完整的，比如是否有题目（包括题干、参考答案和解析）说根据材料回答问题但是没给材料？是否有题目（包括题干、参考答案和解析）包含了莫名其妙的词汇（比如课本、课内、课外等）使得题目不严谨？
    - 1：完全不完整且不严谨：多道题目存在严重缺失（如缺少必要材料或参考答案）或包含不严谨表述（如出现"课本""课内"等模糊词汇）；题目无法正常使用。
    - 2：完整性较差且严谨性不足：部分题目缺失关键内容（如解析不完整）或存在多处不严谨表述；影响题目有效性和专业性。
    - 3：基本完整严谨但有明显问题：绝大多数题目完整且表述专业，但个别题目存在材料缺失或不严谨用语（如出现1-2处"课外"等不当表述）。
    - 4：高度完整严谨，仅细微问题：所有题目均完整且表述专业，仅极少数题目存在无关紧要的表述不够精准（如材料描述可更明确）。
    - 5：完全完整严谨且专业：每道题目均提供完整题干、材料、参考答案和解析；表述严谨专业，无任何模糊或不恰当用语。

# 以下是与该需求相关的测验题：
================测验题开始================

{aggregated_quiz}

================测验题结束================

# 请首先逐步分析测验题，然后以以下JSON格式返回你的评估结果：
{
    "教育价值": 分数(int),
    "多样性": 分数(int),
    "主题相关性": 分数(int),
    "难度适宜性": 分数(int),
    "全面性": 分数(int),
    "规范性": 分数(int),
    "题目数量及题型范围准确性": 分数(int),
    "选择题选项准确性": 分数(int),
    "题目完整性和严谨性": 分数(int)
}

记住，输出不要包含```json和```
'''

import string
import random
random.seed(42)
def generate_random_string(length=15):
    # 定义字符集：小写字母 + 数字
    characters = string.ascii_lowercase + string.digits
    # 随机选择字符并拼接成字符串
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def get_response(example,prompt_template):

    try: 
        eval_res_text = ""
        eval_res = {}
        catalog = DETIALED_SUBJECTS[example['catalog']]['desc_for_llm']
        try:
            aggregated_quiz_ = getData(example['quiz_url'])
            aggregated_quiz = aggregated_quiz_.split("### 最终试卷")[-1]
        except Exception as e:
            print(f"evaluate error:{e}")
        finally:
            aggregated_quiz = getData(example['quiz_url'])
        
        query = example['query']
        prompt = prompt_template.replace("{query}",query).replace("{aggregated_quiz}",aggregated_quiz).replace("{catalog}",catalog)
        # print(get_json_result(call_Hkust_api(prompt,remain_reasoning= True, temperature = 0.0,top_p = 1.0)))
        eval_res_text= call_Hkust_api(prompt,remain_reasoning= False, config = {"temperature":0,"top_k":1,"do_sample":False})
        eval_res = get_json_result(eval_res_text)
        print(eval_res_text)
        dict_ = {**example, "eval_res_text": eval_res_text, "eval_res": eval_res}
        return dict_
    except Exception as e:
        print(e)
        return  {**example, "eval_res_text": eval_res_text, "eval_res": eval_res}



    
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


def evaluate_quiz(data,path):
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
                if 'eval_res' in item and 'eval_res_text' in item and len(item['eval_res_text'])!=0 and len(item['eval_res'])!=0:
                    have_change_ids.append(item['id'])
            not_change_ids = list(set([item['id'] for item in data])-set(have_change_ids))
        print(len(data))# train_data是list[dict]
        print(len(have_change_ids))
        multiThreadGenerateIntentAndSaveEveryone(
            data,
            path,
            prompt_template,
            have_change_ids,
            thread_count= 20,
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

# real_data = [item for item in train_data if item['modify_content'].strip()!=""]
# process(real_data,path = f"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/3.伪问答对（{q_type}）.jsonl")

# process(test_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/hkust/data加课本课外内容/test.jsonl")