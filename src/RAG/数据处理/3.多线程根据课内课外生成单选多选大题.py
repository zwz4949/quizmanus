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


train_data = getData("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/2.加课外内容.json")

prompt_template = '''# 角色说明
你是一个根据课本内容和课外内容生成{type}的专家，给定一段{subject}{grade}的课本内容和一段相关的课外内容，请根据他们生成一道高考{type}。

# 步骤
- 利用你的知识，根据给定的课本内容和课外内容，生成高考{type}，要求内容详细，覆盖全面
- 反复检查是否完整生成了题干、参考答案和解析，三者都要存在
- 反复检查题干、参考答案和解析是否和给定内容以及自身知识相一致，保持合理性和逻辑性和教学验证性

# 说明
- 出的题目绝对不能离开课本知识，要紧紧扣住课本内容，课外内容只是作为辅助出题的内容
- 题目内容需包含题干、参考答案和解析
- 参考答案不一定完完全全和给定内容一模一样，可以适当根据你的自身知识扩展
- 题目内容中不要标出是来自课本还是课外
- 始终保持使用用户相同的语言
- 要求内容详细，覆盖全面
- 题目设计由浅入深，兼顾识记、推理与创新能力，符合“重基础、强应用”的命题导向
- {type_intro}
{example}

- 输出格式为：
{ 
    "题干、参考答案、解析": "题干、参考答案、解析的所有内容都写在这"
}
- 不要在输出包含```json和```

课本内容: {inner}
课外内容：{outer}
输出：'''

# q_type = "单选题"
# q_type = "多选题"
q_type = "主观题"
print(q_type,'\n')
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
    
    intro = {
        "单选题":"单选题有四个选项ABCD，参考答案为其中一个选项",
        "多选题":"多选题有四个选项ABCD，参考答案为其中二到四个选项，一般答案为二或者三个选项居多",
        "主观题":"主观题中若有多道小题，参考答案需要一道道针对性给答案，并且每题要逐步增加难度，从识记到应用，同时在思考时每小题需要给出涉及的具体课本知识点。\n政治主观题和历史主观题多为直接给出若干个材料，然后给出若干道需要根据材料回答的简答题，简答题答案需要涉及课本内容和结合课外内容；生物主观题多为在课外内容背景下对课内知识的考查，多为填空题，也可以有简答题，填空用___下划线表示需要填空的地方；地理主观题多为在课外内容背景下对课内知识的考查，答案需结合课本内容和课外内容，多为简答题"
    }
    exam = {
        "高中政治": "题干：阅读材料，完成下列要求。  \n税收是国家治理的基础和重要支柱，在社会经济生活中发挥着巨大的作用。  \n    材料一  \n    党的十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决\n定》强调落实 “税收法定原则 ”，2015年3月，十二届全国人大三次会议表决\n通过新修订的《中华人民共和国立法法 》，明确 “税种的设立，税率的确定和\n税收征收管理等税收基本制度 ”只能由法律规定。  \n    材料二  \n    2014年10月《国务院关于扶持小型微型企业健康发展的意见》提出要 “认\n真落实已经出台的支持小型微型企业税收优惠政策 ”，2015年3月，国家税务\n总局出台十大措施确保小微企业税收优惠政策落实。数据显示， 2015年一季\n度，全国享受企业所得税减半征收的小微企业有 216万户，受惠面在 90%以\n上，减税 51亿元；享受暂免征收增值税和营业税政策的小微企业和个体工商\n户共有 2700万户，减税 189亿元。\n（1）结合材料和所学政治知识，说明我国为什么要坚持税收法定原则。\n（2）结合材料和所学经济知识，分析当前对小微企业实施税收优惠的理由。\n\n参考答案：（1） \n①是全面落实依法治国基本方略、加快社会主义法治国家的要求；  \n  ②有利于完善税法法律制度 ；规范政府行为， 防止随意增减税负 ；保护纳税人权\n益。\n（2） \n①小微企业在国民经济中具有重要地位 ，其健康发展能吸纳阿亮就业 ，促进经济\n发展；  \n②小微企业发展面临税负较重，融资难融资贵等困难；  \n③税收优惠可降低小微企业负担，有利于其生存与发展；  \n④引导和鼓励大众企业、万众创新。\n\n解析：（1）本题要求结合材料和所学政治知识，说明我国为什么要坚持税收\n法定原则。此为原因类试题，主要考查党的领导作用和人大的性质、职权相\n关知识。本题关键知识点是党领导立法，依法执政。党的主张通过全国人大\n的法定程序上升为国家意志。同时运用自主学习过程中获得的重大时事的相\n关信息即我国推进法治建设坚持落实依法治国依宪治国的背景组织答案。\n（2）本题要求结合材料和所学经济知识，分析当前对小微企业实施税收优惠的\n理由，也是原因类试题，主要考查考生调动和运用知识能力，主要表现为能\n够根据题目获取和解读试题的信息。本题破题关键是理解在自主学习过程中\n获得的重大时事的相关信息分析出小微企业存在的问题即融资难、市场竞争\n中地位不平等。结合教材税收的作用、完善基本经济制度相关知识针对小微\n企业的困难因素说明理由。",

        "高中生物":"题干：已知果蝇的灰体和黄体受一对等位基因控制，但这对相对性状的显隐性关系和该等位基因所在的染色体是未知的．同学甲用一只灰体雌蝇与一只黄体雄蝇杂交，子代中 ♀灰体：♀黄体：♂灰体：♂黄体为 1：1：1：1．同学乙用两种不同的杂交实验都证实了控制黄体的基因位于 X染色体上，并表现为隐性．请根据上述结果，回答下列问题：  \n（1）仅根据同学甲的实验， 能不能证明控制黄体的基因位于 X染色体上，并表现为隐性？_______  \n（2）请用同学甲得到的子代果蝇为材料设计两个不同的实验，这两个实验都能独立证明同学乙的结论．（要求：每个实验只用一个杂交组合，并指出支持同学乙结论的预期实验结果．）_______．\n\n参考答案：（1）不能\n（2）实验 1：杂交组合： ♀黄体× ♂灰体  \n预期结果：子一代中所有的雌性都表现为灰体，雄性都表现为黄体  \n实验 2：杂交组合： ♀灰体× ♂灰体  \n预期结果 ：子一代中所有的雌性都表现为灰体，雄性中一半表现为灰体，另一半表现为黄体\n\n解析：（1）同学甲用一只灰体雌蝇与一只黄体雄蝇杂交，子代中 ♀灰体：♀黄体： ♂灰体： ♂黄体为 1：1：1：1，子代的表现型无关，不能说明控制黄体的基因位于 X染色体上．  \n（2）要根据甲同学设计实验来证明控制黄体的基因位于 X染色体上，有两种方法，一是用 ♀黄体×♂灰体杂交，若子代中所有的雌性都表现为灰体，雄性都表现为黄体，二是用 ♀灰体×♂灰体，若子一代中所有的个体都表现为灰体，说明控制黄体的基因位于 X染色体上．",

        "高中历史":"题干：中外历史人物评说  \n材料  \n    鸦片烟流行内地，大为风俗人心之害。节经降旨严饬稽查，而此风未经革除。总由海口守巡员弁（官兵）卖放偷漏，以致蔓延滋甚 ……且止查禁海口洋船，而于民间私熬烟斤，未经议及。嗣后如有洋船夹带鸦片进口，并奸民私种罂粟，煎熬烟膏，开设烟馆，文职地方官员及巡查委员，如能自行拿获究办，免其议处 。其有得规故纵者 ，仍照旧例革职 。﹣﹣《上谕档（道光朝） 》（1823年）  \n（鸦片）每岁易银至数百万两之多 ，非寻常偷漏可比 ，若不极力严禁 ，弊江河所终极。死后该省通市 ，务当恪守定例 ，只准易货 ，毋许易银 。﹣﹣《上谕档（道光朝）》（ 1829年）  \n同心合力 ，不分畛域 ，上紧查拿 ，毋得稍形松劲 ……即文武官员 、军民人等吸食，不知悛改者 ，亦著一体查拿 ，分别处理 。﹣﹣《上谕档（道光朝） 》（1838年）\n（1）根据材料，概括指出道光皇帝对鸦片问题的认识及措施。\n（2）根据材料并结合所学知识，简评道光皇帝的禁烟政策。\n\n参考答案：（1）认识：严重败坏社会风气，腐蚀官吏，白银外流危及财政。  \n措施：禁止民间制作、贩卖鸦片；严禁走私进口；不许以银易货；惩罚办吸食者。\n（2）措施较全面，取得一定成效，虎门销烟；具有正义性；鸦片战争后具有妥协性。\n\n解析：（1）本题主要考查考生对材料的归纳、认识能力。从鸦片走私的“蔓延滋甚”及其危害谈认识。材料中对洋船夹带、奸民私种熬膏开烟馆及军民人等吸食的规定，可归纳出答案中三点措施，隐含禁止用白银买鸦片的规定。\n（2）从整体角度、不同时期（发展变化地）辩证地看待道光帝的禁烟措施。如具有正义性；鸦片战争后具有妥协性等。",

        "高中地理":"题干：我国某地的 “佛手山药 ”有三百多年的种植历史， 2009年国家农产品地理标态认证。 “佛手山药 ”形如手掌，品质优良，味道鲜美，营养丰富，过去，因深藏于大山之中，加之外形不规则，皮薄、贮存期短，长距离运输容易损坏或变质 ，“佛手山药 ”鲜为人知 。近年来，当地政府依托 “佛手山药 ”大力发展乡村旅游，带领农民走上脱贫致富之路。  \n（1）简述依托 “佛手山药 ”发展乡村旅游带来的效益。  \n（2）设计两项依托 “佛手山药 ”开展的旅游项目。\n\n参考答案：“（1）促进种植规模扩大，形成特色观光农业，增强吸引力，增加旅游收入；带动相关产业发展，增加就业机会；利于基础设施的完善，促进经济结构的调整，促进经济发展。  \n（2）农家乐旅游：游客全家一起挖、清洗山药，动手做当地的特色山药菜品或农副产品，体验乡村劳作，尤其是让小孩锻炼做家务的能力，体验生活。  \n特色摄影游：设计一些和佛手山药相关的一些摄影题材和摄影路线，吸引摄影爱好者摄影，提高佛手山药的品牌知名度。  \n特色美食品尝：组织游客参观山药从原料到产品的加工过程，并且参与其中的某些环节，并且品尝美食，体验其中乐趣。\n\n解析：（1）旅游效益主要从经济和社会两方面来分析。从经济效益来说，带动佛手山药种植业的发展，形成当地特色产业，拉动当地经济的发展；从社会效益来说，可以促进当地人口就业。  \n（2）依托佛手山药，可以设计挖掘山药，体验农民劳作；设计一些和佛手山药相关的一些摄影题材和摄影路线，吸引摄影爱好者摄影，提高佛手山药的知名度；开发佛手山药美食等这样一些旅游项目。"
    }
    q_id = generate_random_string(20)
    try: 
        # [dict_[example['type']]]
        pr = prompt_template.replace("{inner}",example['课本内容'].strip()).replace("{outer}",example['课外内容'].strip()).replace("{grade}",str(example['grade']).strip()).replace("{subject}",example['subject'].strip()).replace("{type}",q_type)\
            .replace("{type_intro}",intro[q_type])
        if q_type == "主观题":
            pr = pr.replace("{example}",f'''- 题目样式例子：{exam[example['subject'].strip()]}''')
        else:
            pr = pr.replace("{example}","")
            
        res = call_Hkust_api(pr,remain_reasoning = True)
        think = re.search(r'<think>(.*?)</think>', res, re.DOTALL)
        if think:
            think = think.group(1).strip()  # 去除首尾空白
        else:
            think = ""

        # 提取 <think> 后面的内容（即字典部分）
        content = res.split('</think>')[-1].strip()

        context = get_json_result(content)['题干、参考答案、解析']
        return {"q_id":q_id,**example,"reasoning":think,"题干、参考答案、解析":context.strip(),"type":q_type}
    except Exception as e:
        print(e,f"错误内容：{res}")
        return {"q_id":q_id,**example,"reasoning":"","题干、参考答案、解析":"","type":q_type}



    
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
                if '题干、参考答案、解析' in item and isinstance(item['题干、参考答案、解析'],str) and item['题干、参考答案、解析'].strip() != "" and 'reasoning' in item and isinstance(item['reasoning'],str) and item['reasoning'].strip() != "":
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

real_data = [item for item in train_data if item['modify_content'].strip()!=""]
process(real_data,path = f"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/合并/3.伪问答对（{q_type}）.jsonl")

# process(test_data,path = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/hkust/data加课本课外内容/test.jsonl")