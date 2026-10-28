# 当前执行协议与关键产物

## 输入来源与锁定

唯一科研定义来源为本次 `try**/preparation.md`。初始化只创建研究请求、配置引用与输入锁，不预设模型、算法、数据列表或预算。

第一步 `research_intake` 输出 `research_context.json`：`goal`、`model`、`baseline`、`datasets`、`hardware`、`metrics`、`evaluation_protocol`、`constraints`，以及可选 `resources` 和 `success_rule`。需要补充信息时输出 `ready:false` 与 `missing_inputs`，流程返回 `NEEDS_INPUT`，不启动评测。准备材料若需修改，创建新 try 保留旧记录。

指标记录稳定 ID、方向（maximize/minimize）、任务范围与 macro/micro 聚合；评分器、预处理、解码、随机种子和比较检查点在实验前固定。预算约束支持绝对阈值或相对基线比例；没有额外限制时使用空数组。

额外材料只按本次方法需要声明。例如模型权重、基线代码、校准数据等，并不固定要求某类码本或现成分数。禁止将评测样本用于校准或调参。

## 数据固定

支持 JSONL、JSON（可指定 records_key）、CSV、Parquet、Hugging Face 本地数据和 Hub repo/split。原始记录放入标准包装的 `data` 字段，宿主生成 `sample_id / task / source_index`。重复内容仍具有不同样本 ID。

在文献检索和候选探索前，对每个数据集总量 N 固定开发 `ceil(0.05N)`、正式 `ceil(0.30N)`、补测为精确补集；三者互斥，合计 N。固定种子为配置中的 seed，默认 42。数据哈希与身份写入 `datasets.lock.json`，摘要写入 `try**/dataset_split.md`。恢复只校验复用，不重新划分；原始数据或分区被修改会拒绝继续。

## 候选与执行

三个独立候选均为 `research_proposal`，方向来自本次问题与证据。字段包含 `candidate_id`、`worker_id`、`mechanism_family`、`mechanism`、`causal_chain`、`budget_design`、`minimal_code_change`、`falsification`。基线通过独立执行 arm 比较，不在候选列表中。

实验计划的 `execution.command` 是 argv 数组，必须含 `{arm}`、`{dataset}`、`{output}`、`{task_id}`。可选 `{workspace}`、`{context}`、`{run}`、`{phase}`、`{python}`。工作区为当前 cycle/workspace。执行输出、日志和缓存应写入运行目录，不修改锁定的数据或实现文件。

每题返回：

```json
{"sample_id":"宿主传入的原样ID","status":"complete","metrics":{"本次指标ID":0.8},"constraints":{"本次约束ID":1.0}}
```

只有属于该任务的指标需要填写；所有声明的约束都要完整测量。数值必须有限。实际执行需运行真实模型、真实基线和科学有效的评分器，禁止写死或伪造分数。

宿主先检查代码、工作区、文件快照与构建，按本次硬件进行预检，然后每任务取固定开发集第一条做非评分冒烟。构建产物会纳入实现哈希。每次正式执行按计划、实现和数据锁记录来源，技术修复后新建 attempt。

## 晋级、补测与结束

开发使用固定 5%。有效开发结果取得严格收益且无指标退化、满足资源约束后晋级正式评测。正式继承原来的方案与选择证据，不重做候选竞赛。

正式先使用固定 30%，通过本次成功规则和约束后再运行剩余 65%，并用同一最终实现重跑固定开发 5%。全量分数从这些逐题结果按本次聚合协议计算。正式未通过不启动补测。全量满足成功规则则 STOP，否则继续研究。

技术错误和不完整测量进入同候选修复；真实测量的资源超限属于科研结果。有效但尚未成功时 REFINE，连续三次未晋级或同族连续小收益进入 PIVOT；正式没有收益可 ARCHIVE。冷却机制族在规定轮次内不能被选择。

输入被修改返回 INVALID；资源不足返回 WAIT，可用 `KVRESEARCH_RESUME_REMOTE_ONCE=1 python3 workflow.py run --try try001` 重新检查。步骤上限是本次命令的工作量，可恢复；周期预算耗尽产生注明未达成目标的终态。

## 产物与输出来源

| 位置 | 内容 |
|---|---|
| preparation.md | 用户原始四项输入 |
| runs/try**/request.md | 原文研究请求 |
| research_context.json | 从用户输入解析并固定的科研定义 |
| inputs.lock.json / datasets.lock.json | 定义引用与数据身份、互斥划分、文件哈希 |
| evidence/ | 文献与综合证据 |
| agent_tasks/ | AI 任务包、worker 证据与执行记录 |
| cycles/000/candidates.jsonl / selection.json | 三个方案及评审选择 |
| experiment_plan.json / implementation_manifest.json | 计划、实现和快照 |
| validation.json / attempts/ | 验证、硬件预检、逐题结果和执行日志 |
| metrics.json / verified_registry.json | 按本次指标汇总并绑定来源的数值 |
| experiment_diagnosis.json / decision.json / lessons.jsonl | 有效性、科研判断与经验 |
| terminal.json | 最终候选、结束原因、来源引用 |
| try**/final_method.md | 方法、主要思想、实测对比、局限与证据 |

第三阶段只整理终态所引用的方法。正式阶段沿计划指向开发阶段的选择与候选，因此不会要求重复生成一份同名选择。报告使用本次动态指标，区分全量成功、部分测量与预算耗尽。
