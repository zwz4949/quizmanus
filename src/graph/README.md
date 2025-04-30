# Graph 模块

本模块负责构建和执行基于 LangGraph 的智能代理图，用于处理复杂的出题任务流程。它协调不同的节点（Agents/Nodes）来完成从理解用户查询、规划、检索知识、生成题目到最终报告的整个过程。

## 目录结构与文件说明
src/graph/
├── agents/
│   └── agents.py           # 定义具体的代理，如 browser_generator 和 knowledge_searcher。
├── llms/
│   └── llms.py             # 封装与大语言模型（LLM）交互的逻辑，支持不同类型的模型（如 OpenAI API, Qwen）。
├── nodes/
│   ├── nodes.py            # 定义图中的核心节点（Coordinator, Planner, Supervisor, Reporter, RAG/Browser Invokers）。
│   ├── rag_nodes.py        # 定义与 RAG (Retrieval-Augmented Generation) 相关的节点（HyDE, Router, Retriever, Reranker, Generator）。
│   └── quiz_types.py       # 定义图状态（State）和 RAG 状态（RAGState）的数据结构。
├── prompts/
│   ├── browser_generator.md # Browser Generator Agent 的系统提示。
│   ├── coordinator.md      # Coordinator Node 的系统提示。
│   ├── knowledge_store_router.md # Planner Node 中用于选择知识库的路由提示。
│   ├── planner.md          # Planner Node 的系统提示。
│   ├── reporter.md         # Reporter Node 的系统提示。
│   ├── sub_reporter.md     # (推测) 子报告或特定格式输出的提示。
│   └── supervisor.md       # Supervisor Node 的系统提示。
├── tools/
│   ├── decorators.py       # 提供工具日志记录功能的装饰器。
│   └── search.py           # 封装搜索工具（如 Tavily Search）。
├── builder.py              # 构建 LangGraph 图实例的脚本。
└── README.md               # 本说明文件。



## 主要功能

1.  **任务协调与规划 (`nodes/nodes.py`, `prompts/coordinator.md`, `prompts/planner.md`)**:
    *   `Coordinator`: 理解用户初始查询，判断是否需要转交给 Planner。
    *   `Planner`: 根据用户查询和可选知识库，制定详细的出题计划（通常为 JSON 格式）。支持在规划前进行搜索 (`search_before_planning`)。
2.  **流程控制 (`nodes/nodes.py`, `prompts/supervisor.md`)**:
    *   `Supervisor`: 根据当前状态和计划，决定下一步应该由哪个 Agent/Node 执行，直到任务完成 (`FINISH`)。
3.  **RAG 与知识检索 (`nodes/rag_nodes.py`, `RAG/`)**:
    *   实现一个子图 (RAG Graph) 来执行检索增强生成任务。
    *   `HyDE`: 生成假设性文档以改进检索。
    *   `Router`: 根据查询选择合适的学科知识库和题型。
    *   `Retriever`: 从 Milvus 向量数据库进行混合检索。
    *   `Reranker`: 对检索结果进行重排序。
    *   `Generator`: 基于检索到的课内知识和可能的课外知识（通过 `knowledge_based_browser` 获取）生成题目。
4.  **工具使用 (`tools/`, `agents/agents.py`)**:
    *   集成外部工具，如 `Tavily Search` 进行网页搜索。
    *   定义 `browser_generator` Agent 使用搜索和爬取工具。
    *   定义 `knowledge_searcher` Agent 进行知识点搜索。
5.  **结果生成与报告 (`nodes/nodes.py`, `prompts/reporter.md`)**:
    *   `Reporter`: 整合所有步骤的结果，生成最终的报告或题目集。
6.  **图构建 (`builder.py`)**:
    *   定义和编译主图 (`build_main`) 和 RAG 子图 (`build_rag`) 的结构，连接所有节点和边。
7.  **LLM 交互 (`llms/llms.py`)**:
    *   提供统一接口调用不同的大语言模型，包括基于 API 的模型和本地模型（如 Qwen）。

## 工作流程（推测）

1.  用户输入查询 (`ori_query`)。
2.  `Coordinator` 接收查询，进行初步处理，决定是否交给 `Planner`。
3.  `Planner` (可能先进行搜索) 制定详细计划 (`full_plan`)，并选择知识库。
4.  `Supervisor` 根据计划，依次调用不同的执行节点：
    *   `rag_er` 或 `rag_and_browser`: 调用 RAG 子图生成题目。RAG 子图内部执行 HyDE -> Router -> Retriever -> Reranker -> (Optional Browser) -> Generator 流程。
    *   (可能还有其他自定义节点，如 `browser_generator`)
5.  每次执行后，结果更新到状态中，`Supervisor` 决定下一步。
6.  当 `Supervisor` 判断任务完成时，调用 `Reporter`。
7.  `Reporter` 生成最终结果。
8.  流程结束 (`__end__`)，结果写入文件 (`quiz_url`)。

## 使用说明

TODO