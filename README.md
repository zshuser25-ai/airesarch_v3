# 三阶段自动科研流程

本项目从用户定义的科研问题开始执行。模型、对比基线、数据、指标、预算和硬件由本次 `preparation.md` 决定。默认入口不绑定特定模型、算法实现、码本、scales 或基线报告。

## 目录与启动

```text
airesarch_v3/
├── workflow.py
├── stage1_preparation/        # 收集四项研究定义
├── stage2_research/           # AI 研究、执行、校验、判断和恢复
├── stage3_output/             # 最终方法与主要思想
├── workflow_tests/
└── try001/                   # 真正启动准备阶段后才创建
    ├── preparation.md        # 用户原始输入
    ├── config.kvresearch.yaml
    ├── dataset_split.md      # 第二阶段固定划分后生成
    ├── runs/try001/          # 科研定义、固定数据、证据、代码及结果
    ├── reports/              # 阶段进度
    ├── memory/               # 本次研究经验
    └── final_method.md       # 研究结束后输出
```

```bash
python3 workflow.py start
python3 workflow.py run --try try001 --max-steps 1000
# 中断后使用相同 run 命令恢复；新研究才再次 start
python3 workflow.py output --try try001  # 重新导出已经结束的研究
```

`start` 从 `try001` 起递增创建目录，不覆盖已有研究。所有 Agent 使用 `gpt-6.1-sol / xhigh / fast`，研究 worker 使用隔离上下文 `fork_turns=none`。

## 第一阶段：预准备

依次填写四项必填内容，每项支持多行，以单独一行 `.` 结束：

1. **科研目标与对比基线**：目标、比较方法及可执行代码或获取入口、指标和资源预算。
2. **测试使用模型**：名称、版本、权重与 Tokenizer 地址，相关运行要求。
3. **数据集地址**：原始评测数据入口、任务与格式、使用哪一个官方 split，以及必要的评测说明。
4. **可用硬件**：GPU 或 CPU、数量与内存、节点连接方式、可用 Python 环境。

全部输入完成后才创建 `try**/preparation.md`。无需事先手工生成开发、正式和补测数据，也无需提供某个固定算法的材料。准备阶段只记录用户输入。

也可使用 `start --goal-baseline '...' --model '...' --datasets '...' --hardware '...'` 提供内容，缺失项仍会提示。模板见 [stage1_preparation/template.md](stage1_preparation/template.md)。

## 第二阶段：研究执行

默认配置为 `research.engine: preparation`。首次 `run` 创建本次配置和运行目录，按以下顺序执行：

1. **解析研究定义**：主 Agent 阅读准备材料并检查入口，生成 `research_context.json`，固定模型、基线、数据入口、硬件、评分器、比较协议、指标方向与预算约束。确实缺少必要信息时返回 `NEEDS_INPUT` 和缺项说明。
2. **固定数据**：宿主程序读取上述入口，为每个数据集生成样本 ID，以固定种子将总量的 **5% 用于开发、30% 用于正式评测、剩余 65% 用于补测**。三部分互斥、并集等于全量。开发与正式数量向上取整，补测取精确补集；实际数量写入 `dataset_split.md`，ID 与哈希写入 `datasets.lock.json`。后续研究复用这一划分。
3. **研究与选题**：检索文献、综合证据、诊断问题；三个独立 worker 各提出一个可选科研方案，三个 reviewer 评审选择。没有预设算法方向；用户基线独立评测，不占候选名额。
4. **实现与验证**：制定实验计划，在本轮隔离工作区实现候选和真实基线，检查代码与构建、硬件、逐任务非评分冒烟。技术故障保持同一候选修复，不作为科研失败。
5. **开发评测**：在固定 5% 上成对运行基线与候选，校验逐题 ID、结果完整性、指标与预算。满足晋级条件则 `PROMOTE`，否则根据实测结果 `REFINE / PIVOT / ARCHIVE` 并积累经验。
6. **正式与补测**：晋级后保留同一方法，在固定 30% 上正式评测；正式通过后运行剩余 65%，并用最终实现重跑固定开发 5%。按逐题结果汇总全量，避免混入旧实现的分数。正式未通过不运行剩余分区。
7. **最终判断**：以本次定义的指标方向、成功规则和资源约束裁决，目标达成则 `STOP`；预算耗尽也会如实结束并标注未达成目标。

所有 AI 步骤都能直接读取本次准备材料。每个控制产物绑定输入哈希，恢复时验证固定输入；修改已锁定的科研定义应新建 try。资源暂不可用则 `WAIT`，检查资源后设置 `KVRESEARCH_RESUME_REMOTE_ONCE=1` 再运行同一条命令恢复检查。

执行细节见 [stage2_research/README.md](stage2_research/README.md)，当前协议见 [RESEARCH_WORKFLOW.md](stage2_research/RESEARCH_WORKFLOW.md)。目录中旧领域适配器及其回归夹具用于兼容旧测试；默认入口使用新的准备材料驱动流程。

## 第三阶段：输出

有效终态产生后自动写入 `try**/final_method.md`：最终方法、主要思想与因果链、资源与预算设计、实现要点、本次指标的基线与候选实测结果、结束状态及证据入口。目标未达成会明确标记。报告沿评测计划引用的原始选择查找方案，支持正式阶段直接继承开发候选。

缺失终态、候选来源或证据哈希不一致时，不生成新的最终报告。

## 安装与验证

框架需要 Python >= 3.10、PyYAML 和 jsonschema。实际模型、评分器依赖由本次任务决定；读取 Parquet 或 Hugging Face 数据需要 `datasets`。

```bash
python3 -m pip install -e stage2_research
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s workflow_tests -v
cd stage2_research
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
