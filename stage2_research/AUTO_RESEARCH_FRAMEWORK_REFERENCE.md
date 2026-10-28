> 历史领域适配流程参考，仅供兼容回归；当前默认流程不读取下文的固定模型、材料、候选或划分协议。当前执行说明见 stage2_research/RESEARCH_WORKFLOW.md。

# 自动研究框架：源码级参考设计

## 已确认的技术底座

采用精简组合设计：

- ResearchStudio 的 artifact-driven `next` 导航器作为控制外壳。
- AutoResearchClaw 的实验契约、执行、诊断修复、证据验证与经验记忆作为实验内核。
- 不直接安装或照搬两个项目的完整流水线。

## 固定上游版本

| 项目 | 仓库 | 参考提交 | License |
|---|---|---|---|
| ResearchStudio | `microsoft/ResearchStudio` | `6cd5b6b6485c05c45a5c875027f87d7ab97d5834` | MIT |
| AutoResearchClaw | `aiming-lab/AutoResearchClaw` | `be4ba4755bf1b52220f25e13b2293b5956590070` | MIT |

后续若复制或改写上游源码，必须记录来源文件和提交，并保留相应 MIT License 声明。

## 从 ResearchStudio 直接参考

### 1. 磁盘产物驱动的 `next` 导航器

来源：`ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py`

参考行为：

- 导航器只读取运行目录，不修改实验产物。
- 根据文件是否存在、是否合法来确定状态。
- 每次只返回一个下一步动作。
- 重复调用保持幂等。
- 进程退出或 SSH 中断后，通过已有产物恢复。
- 明确区分正常完成、失败终止和重试耗尽。

本项目映射：实现 `kvresearch next --run <run_dir>`，检查 KV Cache 研究产物并输出唯一下一步任务。

### 2. 命令路由与 Agent 交接

来源：`ResearchStudio-Idea/skills/idea_spark/scripts/run.py`

参考行为：

- 一个命令入口承载所有步骤。
- 确定性步骤直接返回可执行命令。
- 需要模型推理的步骤返回结构化任务说明、输入路径和输出路径。
- 大产物通过文件传递，不塞入对话上下文。
- 每个模型推理步骤使用隔离上下文。

本项目映射：提供 `next`、`agent-task`、`validate-cycle`、`monitor` 与 `auto` 等命令；研究 Agent 只接收当前步骤需要的文件路径。

### 3. 严格产物 Schema 与路由回归测试

来源：

- `ResearchStudio-Idea/skills/idea_spark/references/schemas.md`
- `ResearchStudio-Idea/skills/idea_spark/scripts/selftest_routing.py`
- `ResearchStudio-Idea/skills/idea_spark/scripts/regression_check.py`

本项目映射：

- 每个 JSON 产物必须先通过 Schema 校验，失败时不得进入下一步。
- 用伪造运行目录覆盖成功、缺文件、坏文件、实验失败、重试和恢复等路由分支。
- 导航规则修改后必须重放全部路由测试。

### 4. 证据驱动的想法生成

来源：`ResearchStudio-Idea/skills/idea_spark/SKILL.md`

只保留以下方法：

- 文献证据整理。
- 瓶颈识别。
- 基于模式生成候选方案。
- prior-art collision 检查。
- 可证伪性和可实现性审计。
- 失败重试必须携带新的约束信息，禁止无信息重复生成。

本项目不引入 IdeaSpark 的卡片渲染、PDF 和展示流程。

## 从 AutoResearchClaw 直接参考

### 1. StageContract

来源：`researchclaw/pipeline/contracts.py`

每个 KV Cache 步骤声明：

- 输入文件。
- 输出文件。
- Definition of Done。
- 唯一错误码。
- 最大重试次数。

本项目不会照搬原有 23 个阶段，只复用这种接口形式。

### 2. 实验迭代器

来源：`researchclaw/experiment/runner.py`

参考行为：

- 运行候选实现。
- 从真实输出解析主指标和辅助指标。
- 与当前最佳结果比较。
- 保留有效候选，淘汰无效候选。
- 保存完整实验历史。

本项目需要替换其通用单指标逻辑，改成 LongBench、GSM8K、MATH500、理论位宽和实际 KV 字节数的多约束判定。

### 3. 结构化诊断与定向修复

来源：

- `researchclaw/pipeline/experiment_diagnosis.py`
- `researchclaw/pipeline/experiment_repair.py`

直接参考其“先分类、后局部修复、再执行”的闭环。保留 OOM、依赖缺失、代码崩溃、数据不可用和指标缺失等分类；增加量化专属分类：

- KV 存储预算超限。
- 码本或张量维度不匹配。
- 长上下文退化。
- Key 或 Value 单侧严重失真。
- 特定层、头或 token 区间异常。
- 输出格式或评测协议漂移。

修复只能针对诊断项，不允许每次从头重写方案。

### 4. VerifiedRegistry

来源：`researchclaw/pipeline/verified_registry.py`

本项目只允许以下数据进入结论：

- 评测脚本产生的原始预测和逐题得分。
- 存储审计器产生的字节统计。
- 运行时采集的显存、耗时和环境记录。
- 可追溯到具体文件与哈希的聚合指标。

模型在分析文字中自行声称的数字不得注册为实验事实。

### 5. 候选竞争、版本隔离和恢复

来源：

- `researchclaw/pipeline/tournament.py`
- `researchclaw/experiment/git_manager.py`
- `researchclaw/pipeline/runner.py`

本项目映射：

- 先生成多个有差异的候选，每个 cycle 选择一个候选执行；失败后按证据 REFINE 或 PIVOT。
- 每个候选使用独立工作区，不修改 `/nfs/AE/zsh/vecinfer` 和 `/nfs/AE/zsh/vecinfer_v1`。
- 每一步完成后原子写入 checkpoint。
- 保存不可变运行清单、追加日志、运行时采样和 exit 产物，便于识别进程中断。
- REFINE 或 PIVOT 前归档旧产物，不能覆盖失败证据。

实现阶段在每个新的候选代码目录初始化独立 Git 仓库，基线目录继续只读。

### 6. 实验记忆和失败经验

来源：

- `researchclaw/memory/experiment_memory.py`
- `researchclaw/evolution.py`

本项目记录：候选机制、配置、数据范围、指标、失败类型、修复动作及修复后结果。后续生成候选时读取相关经验，避免重复已经证伪的方向。

## 明确不引入的部分

- AutoResearchClaw 的完整 23 阶段流水线。
- 人工确认与 HITL 子系统。
- 论文写作、LaTeX、引用验证、发布和前端。
- 通用领域适配器、通用训练数据集和默认 Docker 环境。
- ResearchStudio 的海报、视频、PDF 卡片和展示页面。
- 每轮实验都重新进行全量文献搜索。

文献与 novelty 检查仅在首次建立研究空间、产生全新机制或执行 PIVOT 时触发。

## 面向本项目的组合运行方式

```text
kvresearch next
  -> 读取运行目录中的真实产物
  -> 返回唯一下一步动作
  -> 确定性工具或隔离研究 Agent 执行动作
  -> 输出结构化产物并校验
  -> kvresearch next
```

实验执行内部使用：

```text
候选假设
  -> 实验契约
  -> 隔离实现
  -> 静态与存储检查
  -> POD 执行
  -> 原始结果注册
  -> 诊断
  -> KEEP / REFINE / PIVOT / ARCHIVE
```

## 实现状态

上述产物协议、只读导航器、Agent 交接、隔离工作区、POD 预检、VecInfer 适配、固定数据映射、存储与运行时审计、指标验证、诊断决策和经验归档均已在 `kvresearch/` 中实现。机器可读契约位于 `framework_design/artifact_catalog.yaml`，运行目录由 `init --run-id <run_id>` 创建于配置的 `project.runs_dir`；干净副本不包含既有运行。
