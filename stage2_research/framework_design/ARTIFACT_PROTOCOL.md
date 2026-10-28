> 历史领域适配流程参考，仅供兼容回归；当前默认流程不读取下文的固定模型、材料、候选或划分协议。当前执行说明见 stage2_research/RESEARCH_WORKFLOW.md。

# KV Cache 自动研究框架：产物协议

状态：已实现。本文保留产物协议设计；实际执行与门禁以 `kvresearch/` 源码及 `artifact_catalog.yaml` 为准。干净副本不包含数据、基线结果或运行记录。

参考底座见项目根目录的 `AUTO_RESEARCH_FRAMEWORK_REFERENCE.md`。本协议采用 ResearchStudio 的 artifact-driven 导航方式，并使用 AutoResearchClaw 的 StageContract 与 VerifiedRegistry 思路。

## 1. 基本原则

- 磁盘产物是真实状态源，不维护独立的可变状态数据库。
- `next` 只读、幂等，每次只输出一个动作。
- 不能仅凭文件存在判定成功；文件必须通过 Schema、来源和完整性校验。
- 根输入不可变；研究轮次使用递增 `cycle`；执行重试使用递增 `attempt`。
- 失败、超限和退化结果必须保留，禁止被新结果覆盖。
- JSON 用于控制和判定，JSONL 用于逐条记录，Markdown 仅用于人类阅读。
- 控制文件先写临时文件，再原子重命名；日志允许流式追加。
- 大型内容只通过绝对文件路径交给研究 Agent，不复制进编排上下文。

## 2. 运行目录

```text
runs/<run_id>/
├── request.md
├── run_manifest.json
├── inputs.lock.json
├── evidence/
│   ├── literature.jsonl
│   └── synthesis.json
├── cycles/
│   └── 000/
│       ├── problem_diagnosis.json
│       ├── candidates.jsonl
│       ├── selection.json
│       ├── experiment_plan.json
│       ├── implementation_manifest.json
│       ├── validation.json
│       ├── attempts/
│       │   └── 000/
│       │       ├── preflight.json
│       │       ├── run_manifest.json
│       │       ├── stdout.log
│       │       ├── stderr.log
│       │       ├── predictions/
│       │       │   └── <task>.jsonl
│       │       ├── storage_trace.jsonl
│       │       ├── runtime_trace.jsonl
│       │       └── exit.json
│       ├── metrics.json
│       ├── verified_registry.json
│       ├── experiment_diagnosis.json
│       ├── decision.json
│       └── lessons.jsonl
└── terminal.json
```

`cycle` 和 `attempt` 均从 `000` 开始，只能追加新编号。不得通过覆盖旧目录实现 REFINE 或 PIVOT。

## 3. 全局字段约定

所有控制类 JSON 必须包含：

| 字段 | 含义 |
|---|---|
| `schema_version` | 产物 Schema 版本 |
| `artifact_type` | 产物类型，与 catalog ID 一致 |
| `run_id` | 所属研究运行 |
| `created_at` | UTC 时间 |
| `producer` | 产生该文件的工具或 Agent |
| `inputs` | 输入文件的绝对路径和 SHA-256 |
| `status` | `complete`、`failed` 或 `partial` |

只有 `status: complete` 的控制产物可以触发正常前进。`partial` 只能恢复当前动作，`failed` 必须进入诊断或终止分支。

## 4. 根级不可变产物

### `request.md`

保存用户研究目标原文及已经确认的约束。后续 Agent 必须读取该文件，不能用摘要代替原始意图。

### `run_manifest.json`

记录运行身份：`run_id`、创建时间、父运行、框架版本、研究模式、项目根目录和 artifact catalog 哈希。创建后不可修改。

### `inputs.lock.json`

锁定所有外部输入：

- Llama 3.1 8B Instruct 模型路径和模型文件哈希。
- VecInfer 基线报告路径和哈希。
- `/nfs/AE/zsh/vecinfer` 的参考 Git 提交。
- `/nfs/AE/zsh/vecinfer_v1` 的目录快照哈希。
- 固定小样本 `manifest.json` 及五个 JSONL 的哈希。
- Python、PyTorch、Transformers、CUDA、驱动和 GPU 型号。
- 评测脚本、提示模板和解码配置的哈希。

恢复运行时任何锁定输入不一致，都必须停止并生成输入漂移错误，不能静默继续。

## 5. 研究证据产物

### `evidence/literature.jsonl`

每行是一篇真实来源记录，至少包含标题、来源 URL、标识符、检索时间、与 1-bit KV Cache 的关系、支持或反对的机制，以及证据来源。无法验证的模型记忆必须标为 `unverified`，不得作为 novelty 结论的唯一依据。

### `evidence/synthesis.json`

将证据整理为已知方法、未解决瓶颈、可借鉴机制和已被否定方向。首次研究运行和 PIVOT 到新机制族时必须存在；普通参数 REFINE 不重复检索文献。

## 6. 每轮研究产物

### `problem_diagnosis.json`

描述本轮要解决的一个主要瓶颈，必须包含：

- 观察到的 VecInfer 失败现象。
- 证据文件路径。
- 可能的量化误差来源。
- 受影响任务、层、Key/Value 或 token 区间。
- 可证伪的诊断。
- 本轮不处理的范围。

诊断没有证据路径或不可证伪时，不得生成候选。

### `candidates.jsonl`

每行一个候选，必须包含：

- `candidate_id` 和机制名称。
- 对应瓶颈。
- 机制描述与因果链。
- 理论存储设计。
- 预计影响的任务。
- 最小实现改动。
- 证伪条件。
- novelty 风险与最近相关工作。
- 与历史失败方向的差异。

候选必须具有机制差异，不能只是同一参数的多个取值。

### `selection.json`

记录被选候选、排序结果、淘汰理由和选择证据。选择维度至少覆盖：可证伪性、实现成本、预计精度收益、存储风险、与历史失败的差异。

### `experiment_plan.json`

记录候选实现与评测契约：

- 候选 ID 和代码基线。
- 唯一允许修改的代码范围。
- 开发或正式评测层级。
- 数据集、样本 manifest 和解码配置。
- 需要输出的逐题记录、存储记录和运行时记录。
- 成功、失败、REFINE 和 PIVOT 条件。
- 超时、显存和重试预算。
- 实际 KV 存储与理论位宽的核算口径。

开发评测包含 Qasper、LCC、GovReport、GSM8K 和 MATH500，每项固定 32 条。开发结果用于候选筛选，不单独构成“超过完整 VecInfer 基线”的最终结论。

正式评测包含全部 13 个 LongBench 任务以及 GSM8K、MATH500，但每个任务只使用其全量数据的固定 30%。计划必须声明 `sample_fraction=0.30`、`execution.gpu_count=8` 和 `execution.batch_size_schedule=[16,8,4,2,1]`。

30% 门禁要求至少一项严格超过 VecInfer 且三项均不低于 VecInfer。通过后才运行 `formal_remaining_70pct_testset/`，它必须是 30% manifest 的严格补集；最终按逐题记录合并两部分，而不是对 30% 与 70% 的百分比直接平均。

### `implementation_manifest.json`

记录独立候选工作区、来源代码哈希、修改文件、补丁哈希、构建命令、运行入口和依赖变化。禁止把 `/nfs/AE/zsh/vecinfer` 或 `/nfs/AE/zsh/vecinfer_v1` 当作可写工作区。

### `validation.json`

汇总运行前验证：语法、导入、接口、模型加载 smoke test、数据哈希、提示与解码一致性、理论存储预算和预估实际存储组成。所有硬检查通过后才允许分配 GPU。

## 7. 每次执行尝试的原始产物

### `preflight.json`

记录远端节点、GPU index、GPU UUID、空闲显存、已有进程、利用率、温度、驱动、CUDA、Python 包版本、磁盘余量和采集时间。GPU index 仅是当次映射，GPU UUID 才是设备身份。正式评测只有在 8 张 GPU 全部满足空闲度和显存阈值时才获准启动，并记录全部 `approved_gpus`。

### `run_manifest.json`

记录实际执行命令、工作目录、环境变量白名单、输入哈希、随机种子、任务分片、开始时间、超时和进程 ID。不得保存密码、令牌或完整环境变量。

### `predictions/<task>.jsonl`

每行至少包含：样本 ID、任务、输入哈希、生成文本、参考答案、逐题得分、输入 token 数、输出 token 数、结束原因、异常状态和生成耗时。

开发评测必须与固定 manifest 的 32 个样本 ID 完全一致；多出、缺失或重复均视为失败。

正式评测必须与 30% manifest 中每个任务的样本 ID 完全一致。执行器在 8 张卡上各维持最多一个模型进程；任务按卡分片并行执行。单进程从当前环境已有记录的最高并发 batch size 16 开始，只有发生 OOM 时依次降为 8、4、2、1，所有尝试均进入独立日志。

### `storage_trace.jsonl`

每个请求至少记录：序列长度、层数、KV 元素数、量化 payload、索引、scale、zero-point、metadata、残差 token、padding、alignment、共享码本、逻辑总字节、实际分配字节、理论平均位宽，以及同一请求的 VecInfer 参考字节。

理论约束与实际约束分别判定，不得用理论值代替真实字节记录。共享开销必须说明是逐请求计入、跨请求摊销还是模型级常驻。

### `runtime_trace.jsonl`

记录采样时刻、GPU UUID、显存占用、利用率、CPU 内存和阶段标签。吞吐比较只有在无竞争负载且环境相同时才具有正式效力。

### `exit.json`

记录退出码、结束时间、耗时、是否超时、是否 OOM、完成任务、缺失任务及日志哈希。它是 attempt 是否结束的唯一标志；PID 文件和日志末尾文字不能代替它。

## 8. 评测、验证与决策产物

### `metrics.json`

只能由逐题预测和 trace 聚合生成，至少包含：

- 每个任务的样本数、有效样本数和任务指标。
- LongBench 开发聚合结果。
- GSM8K 和 MATH500 分别的正确数与准确率。
- 理论位宽是否通过。
- 每请求实际 KV 字节是否通过。
- 峰值显存、运行耗时和失败计数。
- 与历史 VecInfer 结果的比较方式和可比性等级。

不得把 LongBench 与数学压成一个可以掩盖单侧退化的主分数。

### `verified_registry.json`

这是唯一可被 `decision.json` 引用的数值注册表。每个数值必须包含值、单位、方向、来源文件、JSON 路径或行范围、来源 SHA-256 和验证状态。

注册器必须拒绝：非有限数、模型自由文本中的数字、找不到原始来源的聚合值、样本数不匹配的正式结果，以及未声明口径的 KV 字节数。

### `experiment_diagnosis.json`

对失败或不足分类，至少区分：环境故障、代码故障、评测故障、存储超限、质量退化和假设被证伪。每项包含严重程度、原始证据、影响范围、是否可修复和建议动作。

### `decision.json`

动作只能是：

- `KEEP`：本轮结果有效，成为当前最佳候选。
- `REFINE`：机制仍成立，只做定向修复或参数调整。
- `PIVOT`：证据否定机制或瓶颈，需要生成新机制。
- `PROMOTE`：开发结果满足晋级条件，进入正式评测。
- `ARCHIVE`：结果有效但不再投入资源。
- `STOP`：目标达成或命中停止条件。

必须引用 `verified_registry.json` 中的条目 ID，并给出下一 cycle 的约束。存储硬约束未通过时禁止 `KEEP` 和 `PROMOTE`。

远端环境和超时属于无效测量，最多自动新建 3 个 attempt；OOM 已先执行 batch 自动降档。重试仍失败、代码异常、评测异常或样本映射异常时进入 AI 修复，不得登记为算法精度失败，也不得累积同族小收益次数。

每个候选必须声明稳定的 `mechanism_family`。相邻同族实验连续两次相对上一同族结果均未达到 LongBench +0.5 分或任一数学任务多答对 1 题，则阶段 17 输出 `PIVOT` 和两轮 cooldown；阶段 18归档该状态；阶段 19及候选选择硬性排除仍在冷却的族。

### `lessons.jsonl`

每行记录一个可复用经验：触发证据、适用范围、失败机制、有效修复、置信度和来源 cycle。经验只能约束未来候选，不能作为实验结果。

### `terminal.json`

记录终止类型、最终候选、最终证据、未解决问题及完整复现入口。只有 `STOP` 决策或不可恢复错误可以产生该文件。

## 9. 比较口径

- 开发层：固定小样本用于同轮候选排序、错误定位和快速淘汰。
- 正式 LongBench：13 个任务分别固定抽取全量的 30%，报告 13 个任务的宏平均分。
- 正式数学：GSM8K 和 MATH500 分别固定抽取各自全量的 30%，分别报告 accuracy。
- VecInfer 1-bit 的 LongBench 与 MATH500 基准由已有完整逐题结果按相同 source index 重算；GSM8K 按项目已确认口径记为 0，并明确标注没有完整配对历史输出。
- 三项指标分别比较，不合成可互相抵消的总分。三项均未超过基准时 `ARCHIVE`；部分超过时 `REFINE`；三项均超过且存储硬约束通过时 `STOP`。

## 10. 导航器判定规则

`next` 按以下原则工作：

1. 校验根级不可变产物和输入哈希。
2. 找到编号最大的 cycle。
3. 按 artifact catalog 顺序找到第一个缺失或无效的必需产物。
4. 如果 attempt 没有 `exit.json`，检查 heartbeat 与真实远端进程，再决定监控或恢复。
5. 如果产物 `failed`，依据错误码路由到同 cycle 新 attempt、下一 cycle REFINE/PIVOT 或终止。
6. 输出一个动作以及完整输入、输出路径，不同时推进多个控制步骤。
7. `terminal.json` 有效时只返回终态，不再产生实验动作。

## 11. 协议版本与迁移

- Schema 只允许向前增加可选字段；删除字段或改变含义必须提升主版本。
- 旧运行不原地迁移，迁移工具必须写入新运行目录并保留来源哈希。
- artifact catalog 自身参与 `inputs.lock.json` 哈希，保证恢复时使用相同判定规则。
