# 基于Agent和RAG的智能试卷生成系统

这是一个基于Agent和RAG的智能试卷生成系统，目前支持生物、地理、政治和历史的试卷生成，文本为主。

该系统可以生成不同难度的题目（简单题、中等偏上难度题），也可以生成不同题目类型的题（单选题、多选题、主观题）。

## 介绍
1. 使用人教版高中学科课本pdf作为RAG的知识库知识，使用MinerU解析pdf获取markdown文件，通过标题作为切分点进行切分，并且基于规则和大模型将诸如讨论、思考、图片等一些与课本知识内容无关的内容去掉，最后使用bge-m3编码并基于Milvus构建分层多粒度向量库。
2. 基于Langgraph实现Agent之间的协同交互，让不同的Agent使用不同的工具完成不同的任务，包括cooridinator、planner、supervisor、RAG、RAG-based browser和reporter。
    - cooridinator：协调者，用于判断是否需要进一步使用planner拆解复杂的任务，还是可以直接解答。
    - planner：计划者，对复杂任务进行拆解，在这里主要是让试卷生成分成若干道题目的单独生成。
    - supervisor：监督者，监督RAG、browser-based RAG和reporter 3个Agent根据planner的计划依次执行任务，并在计划执行完后进一步判断是否需要额外步骤还是直接结束。
    - RAG：使用RAG技术从课本知识向量库中检索信息用来生成题目，多用于生成简单题目；RAG中使用了HyDE、路由技术、以及bge-reranker-v2-m3。
    - RAG-based browser：使用RAG之后的检索文档用于网络搜索相关的新闻时事、历史事件或者一些研究报告的搜索，多用于生成融合了课内课外知识的中等难度偏上的题目。
    - reporter：报告：用于整合先前生成的题目，并且完善题目的不足之处，使之成为一份完整的试卷。
3. 题目生成模型采用 [https://github.com/OpenLMLab/GAOKAO-Bench] 的数据集在Qwen2.5-7B-Instruct下基于LoRA使用一张A800进行SFT，得到quiz-qwen-7B。高考数据集由于只有题目、参考答案和解析，并没有上下文（课本知识和课外知识），因此使用Deepseek-r1对题目进行课本知识和课外知识进行提取和扩写，生成伪课内课外知识用于训练。同时SFT中加入basemodel的伪原训练集（对齐数据集，这里采用Qwen2.5-72B的英文数据集，我们也测试过加入Qwen2-72B的中文数据集，但是发现中文数据集质量可能不太行，会造成降点，因此只用英文数据集，对齐数据集和任务微调数据集比例为1：1；我们也试过用0.05或者0.1：1的比例，但是发现没有效果），以减轻遗忘程度，该训练集从 [https://github.com/magpie-align/magpie/blob/main/navigation.md] 获得。
4. 采用 vLLM 部署生成模型，并结合 FastAPI 构建高性能推理服务，构建稳定、快速的 API 接口，实现推理加速。
5. 评估使用LLM-as-a-judge进行，prompt在 [https://github.com/sofyc/ConQuer] 的评估prompt基础上加以翻译和修改。
6. 同时我们也对SFT之后的生成模型进行在通用数据集（MMLU和MMLU Pro以及CMMLU）上的测试，评估结果如下：

| models                                                 | MMLU Pro | MMLU | CMMLU | llm  |
| ------------------------------------------------------ | -------- | ---- | ----- | ---- |
| Qwen-2.5-7B-Instruct                                   | 53.12   | 74.2 | 77.09 |   -   |
| Qwen-2.5-7B-Instruct-LoRA-Quiz-SFT(没有对齐数据集）    | 44.93    | 70.7 | 75.20 | 4.50 |
| Qwen-2.5-7B-Instruct-LoRA-Quiz-SFT | 48.65    | 71.3 | 76.73 | 4.44 |

## 目前存在问题：
1. 有时候planner并不能按用户的要求给出计划，比如生成10道题目，但是他只生成了9道；目前解决方案是让supervisor在计划执行完毕后对目前生成情况进行判断是否进一步生成题目。
2. 一道一道题目生成会使得在后续token消耗变得超级大。
3. 网络搜索使用的Jina AI有免费token消耗限制。

## 未来研发方向
1. 不止步于基于本低知识库的试卷生成系统，而是支持用户实时上传pdf，实时根据给定pdf生成题目、试卷。
2. 加入问答系统，给用户解答问题。
3. 目前仅仅支持纯文本且题目token量没那么大的生物、地理、历史、政治，未来可以加上语文、英语、数学等。