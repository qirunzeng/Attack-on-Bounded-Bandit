# 实验与论文一致性审计（2026-09-07）

范围：论文的实验设定、UCB/TS 阈值构造、模拟与统计口径、所引用基线的实现及图表。
这不等于对论文全部定理和证明的形式化验证。

| 项目 | 审计结论与处置 |
|---|---|
| 原始数据 | 原工作区仅有 ml-1m，但有标为 ml-25m 的历史 CSV；仅凭目录名无法确认来源。重新下载官方 25M ZIP，核对官方 MD5、ZIP CRC，使用真实 ratings.csv 重跑。旧 CSV 不作为新图数据源。 |
| 实验时域 | 依用户选择恢复原网格 100k、250k、400k、550k、700k、850k、1M。25M 是评分条数，不是时域。 |
| 最严重问题：成功率 | 原 ml-25m 时域 CSV 的 400 条离线攻击记录标记为 certificate。原函数直接把 H 次目标拉臂填入结果。修为无模拟就没有观测计数；新管线逐轮运行 UCB/TS。 |
| 干净日志 | 离线构造每臂 5 次有放回采样，重跑间独立；在线基线没有离线日志，保存全零 clean_sum_json。 |
| 时间与比例 | 离线构造 T0=5K+sum(n)，H=T-T0；在线基线 T0=0、H=T，两种目标比例均为 N_K^on/T。初始化拉臂属于在线时域。 |
| UCB 决策 | Ours/Clipped 与正文一致；Xu 单独使用原文 sqrt(log(T)/N)。本论文学习器为：empirical_mean + 3 sigma sqrt(log(t)/N)，sigma=0.5，第一轮 t=T0+1。 |
| UCB 注入 | g_i 中的减项为 3 sigma sqrt(log(T)/T)，目标阈值 T(z-lower_mu)/(1-z)，非目标向上取整、目标额外加 1，与附录一致；逐次检查分离条件。 |
| TS 决策 | Ours/Clipped 使用本论文的 Gaussian N(empirical_mean,1/N)；Xu 使用原文 Beta–Bernoulli TS。 |
| TS 注入 | gamma(T) 与附录一致。目标下包络检查两个端点及区间内驻点 m=(2x(1-lower_mu)/gamma)^2-x；向上取整、目标额外加 1。 |
| 数值优化 | Threshold 用网格定位加黄金分割求连续阈值；不是整数全局最优证明。Direct 原搜索保留供内部诊断；全部实验图均不展示 Direct，不声称全局最优。 |
| 边界扫描（修正） | 原管线把未知随机目标的下置信界套到了已知确定性反馈上，忽略了该实验可用的信息。现按用户确认，仅在 gap 扫描中使用精确 lower_mu=mu_K；我们的每个真实目标反馈及干净目标样本均等于 mu_K；Xu TS 例外，按原文使用 Bernoulli(mu_K)。随机 MovieLens 主实验继续使用原置信下界。 |
| Jun UCB 基线 | 按目标均值减 2 beta 与 margin=0.01 确定抑制量，再截到合法奖励区间。是 bounded adaptation，不继承原无界攻击保证。 |
| Zuo TS 基线 | 原指数阈值在 [0,1] 下恒为负，截断后的反馈恒为 0。新模拟直接计算完全等价的截断反馈，不需要 exp(min(N,20)) 来制造伪“请求成本”。 |
| Xu 基线 | 当前直接使用原文学习器与附录 A.2/A.4 预算。Xu UCB 保留固定真实目标反馈；Xu TS 改为同均值 Bernoulli 反馈及 Beta(1,1) 先验。撤回上一版自推的适配预算及独立证明。 |
| 排除规则 | mu_K=0 预算无定义；C1+C2>=T 无停止攻击后评估窗口；TS 原式的第一阶段计数下界非正时不适用。保留原因与记录，不绘制这些点，不据此声称所有攻击不可能成功。 |
| 成本公平性 | 离线注入样本数、在线非目标抑制轮数（包含 0→0）、两阶段计划腐化轮数不同。正文、图注、CSV 均明确单位；不能从图直接得出相同威胁模型下的最优性结论。 |
| 统计 | 每设置 10 次真实模拟，均值±样本标准差；保留失败/非满成功的真实比例。没有把失败轨迹过滤掉或当测试错误丢弃。 |
| 可复现性 | 新 CSV 与 manifest 配对，图生成器拒绝数据混用、哈希不符、缺失重复、不完整网格和 smoke 数据。 |

一手基线参考：

- Jun et al. (2018), [Adversarial Attacks on Stochastic Bandits](https://proceedings.neurips.cc/paper_files/paper/2018/file/85f007f8c50dd25f5a45fca73cad64bd-Paper.pdf)：UCB 与在线奖励抑制机制。
- Zuo (2024), [Near Optimal Adversarial Attacks on Stochastic Bandits and Defenses with Smoothed Responses](https://proceedings.mlr.press/v238/zuo24a/zuo24a.pdf)：Algorithm 4 的 Gaussian TS、Algorithm 5 的指数抑制阈值。
- Xu et al. (2021), [Observation-Free Attacks on Stochastic Bandits](https://proceedings.neurips.cc/paper/2021/file/be315e7f05e9f13629031915fe87ad44-Paper.pdf)：Algorithm 1 的两阶段机制、Section 5.3 的 Beta posterior 设定。

仍需谨慎的科研结论：

- 10 次重复足以复现实验曲线，但不能验证 95% 或 90% 高概率定理。
- 从 MovieLens 全体电影挑选极低正均值目标属于困难实例构造，不能解释为典型电影推荐表现；电影 ID、评分数与均值随结果保存。
- 有限 T 下低成本和满成功率支持构造有效性；不能单独证明渐近 Theta(S_T) 最优性。
- Direct 的非凸阈值网格搜索仍是数值比较，不声称整数全局最优。确定性 gap 实验中使用已知目标均值，不意味着随机 MovieLens 主实验可访问真实均值。

初次重跑与验证结果（gap 部分已归档并由确定性目标实验替代）：

- 原始评分 25,000,095 条（不含 CSV 表头）。目标电影为 ID 4775，Glitter (2001)，669 条评分中 11 条为正，均值 0.01644245142。
- 共 1,400 条结果：1,300 条逐轮模拟，100 条 Two-phase 预算不可行。每个参数设置 10 次重复。
- 560 条阈值攻击轨迹均观测到在线目标比例为 1；这不等于从 10 次重复证明高概率定理。
- UCB 在 T=1M：平均成本 124,870.3，Clipped Suppression 为 357,313.0，后者在线目标比例约 0.522。
- TS 在 T=1M：平均成本 182,870.2，Clipped Suppression 为 157,216.2，后者在线目标比例约 0.790；不能声称 TS 在所有配置下成本更低。
- 四项回归测试通过，涵盖三种攻击模式下的独立 NumPy/Numba 轨迹对照、阈值合法性与时钟、证书模式无伪观测值、绘图拒绝未模拟或重复不完整结果。
- 输入/代码/结果哈希校验通过，所有模拟结果均通过成本分解、总时域、在线计数和两种比例的逐行校验。
- 六图曲线数与标签契约检查通过；论文与实验目录的 TeX 一致。LaTeX 编译成功，无 overfull box 或未定义引用；已视觉检查全部六图。

2026-09-07：确定性目标与实线配色修正

- 用户明确确认 gap 实验采用已知的恒定真实目标反馈。此时真实目标经验均值始终等于 mu_K，无须减去用于未知随机目标的统计置信半径。保留 N_i^0=5，不通过增加样本量改变实验。
- `known_target_mean` 是显式参数，默认关闭；UCB/TS 的 Threshold 与 Direct 均使用同一个精确目标下界。目标均值与干净日志不一致时拒绝计算。
- 在修正前绕过缓存，以实际输入分别计算扫描两端，确认四个原始分配函数确实返回相同分配；记录见 history/conservative-gap-20260907/gap_diagnostic.json。这排除了缓存导致平线，但不能为错误的信息设定辩护。
- 所有六幅图改用实线，并以圆圈、方框、三角等标记区分。按用户随后给定的六种 RGB 定义选色：总成本/总比例用 myblue，基线用 myred，在线比例用 mypurple，Direct 用 mybrown，Two-phase 用 myorange；成本分解的 Target/Non-target Avg. 分别用 myred/mypurple。mygreen 保留定义但不与蓝色同图使用。
- Direct 使用相同的阈值网格；TS 下包络通过已证明单峰的整数搜索加速，最终仍由原证书检查函数验证。
- 修正后完整重跑 10 次重复。全部 560 条非 gap 记录与归档结果逐字段相同；gap 中的四条成本曲线均在每个重复内随目标均值增大而下降。
- gap 从 0 增至 6 个边界尺度时，UCB 平均 Total/Direct 成本分别从 41861.2/36847.7 降至 997.0/996.0；TS 分别从 64785.4/53588.0 降至 3375.4/3240.0。
- 6 项自动回归测试通过；另在 UCB/TS 各三个代表性目标均值处对照原 Direct 搜索，注入分配逐元素完全一致。结果文件 `direct_validation.json` 记录了该项对照。

2026-09-07：图 1(b)/4(b) 基线成本非单调性复核（历史口径，已被下文替代）

- `diagnose_baseline_k.py` 使用独立的归零抑制实现、原始 MovieLens 奖励数组及已保存的干净日志/随机种子，重放 K=30、40、50 的 UCB/TS 各 10 次轨迹。全部 60 条逐臂计数和成本与论文 CSV 完全一致；UCB 在这些轨迹的每个非目标轮次上，抑制阈值均为负，故归零实现等价。TS 的区间截断已恒等于归零。
- 当前基线成本为实际修改次数：非目标奖励 1→0 计 1，0→0 计 0。在这些二元归零轨迹中，这也等于实际 L1 奖励改变量，但不等于所有非目标拉臂次数。
- UCB 的 K=30/40/50 非目标拉臂均值为 188765.9/192091.0/193732.1，确实增加；但按拉臂次数加权的非目标真实均值为 0.72726/0.69062/0.68775，导致实际修改次数为 137287.4/132660.9/133195.8。
- TS 对应的非目标拉臂次数为 182867.8/188534.5/191834.9，加权均值为 0.72759/0.69105/0.68818，修改次数为 133044.8/130320.7/132022.8。
- 臂集合按评分数量选择，随 K 扩大而加入不同奖励均值的电影；该扫描同时改变臂数与奖励组成，不能解释为其他条件相同下的纯 K 效应。较低成本也伴随更低目标选臂比例，不表示攻击效率改善。保持原始结果，不为单调性修改数据或临时更换成本口径。
- 每次重放与汇总保存在 `results/ml-25m/paper/baseline_k_diagnostic.json`，其绑定的原结果 SHA256 随文件保存。


2026-09-07：按用户要求将 0→0 纳入抑制成本（当前口径）

- Clipped Suppression 成本统一为全部在线非目标抑制轮数，即 sum(N_i^on, i<K)，不再以奖励数值是否变化为计数条件。主运行器、CSV、绘图校验、图注、正文及 TS 附录一致更新。
- 用已保存的逐臂在线计数精确重计 280 条基线记录；仅更改 cost/cost_definition 两列，全部轨迹、种子、比例和其他方法的数据保持一致。旧 CSV/manifest/诊断归档于 history/changed-observation-cost-20260907/；新 manifest 保留原轨迹生成源哈希，并记录重计脚本、原结果哈希及当前复现源码哈希。这是统计口径重计，不声称重新生成全部轨迹。
- UCB 在 K=30/40/50 的成本均值为 188765.9/192091.0/193732.1；TS 为 182867.8/188534.5/191834.9。
- T=1M 时，UCB 的 Ours/baseline 为 124870.3/477911.3，TS 为 182870.2/210318.8。按当前口径，两个 learner 在两组网格上均为 Ours 平均成本较低；历史的 TS 成本比较表述已撤换。
- 新增全零反馈回归检查：即使所有反馈均为 0，UCB/TS baseline 的非目标抑制动作仍产生正成本；共 7 项测试通过。绘图入口逐条检查基线成本等于非目标在线计数，拒绝旧口径数据。


2026-09-07：按用户要求，图 3(a) 及对应 TS 图 6(a) 移除 Direct 曲线与图例，仅显示 Total 和 Xu et al. (2021)。正文及图注同步删除 Direct 比较说明；原始数值结果留存供审计。


2026-09-07：移除 Xu/Clipped 的离线 warm start（当前初始化规则）

- 用户明确要求两类基线全程在线。现在 Clipped Suppression 与 Xu et al. (2021) 的初始计数及奖励和为零，没有每臂 5 条干净日志；全部 T 轮都在线运行。
- UCB 和 Gaussian TS 都在前 K 个在线轮次逐臂初始化：目标先行，随后依次选择其余臂，使 Clipped 的目标统计量有定义。攻击从第 1 轮生效；初始化不是额外的干净日志，也不在 T 外。Clipped 的非目标初始化轮次计费，Xu 的两阶段日程包含初始化，预算可行性以 C1+C2<T 判断。
- 使用原配置与种子重新模拟/评估全部 560 条基线记录（含预算不可行记录），840 条离线 Ours/Direct 记录逐字段保持一致。归档见 history/baseline-warm-start-20260907/；manifest 保存新基线源码哈希及原结果来源。
- 两类基线的 H=T、T0=0，N_K/T=N_K^on/T。绘图入口逐条检查初始日志、时钟和分母，拒绝仍带 warm-start 的基线数据。
- T=1M 时，UCB 的 Ours/Clipped 平均成本为 124870.3/476527.1，Clipped 目标比例 0.523473；TS 为 182870.2/206534.3，Clipped 目标比例 0.793466。正文及 TS 附录同步更新。
- 加入无日志 UCB/TS 的独立参考轨迹对照，覆盖第 1 轮、初始化完成、Xu 两阶段边界以及全程计数；另验证预算介于 T-5K 与 T 之间的 Xu 实验现在可行。

- 当前 9 项回归测试全部通过；另用独立归零实现重放 K=30/40/50 的 60 条新 Clipped 轨迹，成本与逐臂计数完全一致。论文图表已重新生成并通过编译及视觉核对。


2026-09-07：K/Gap 扫描固定时域改为 T=1,000,000

- 按用户要求，主 UCB 与附录 TS 的臂数扫描、目标 gap 扫描统一固定 T=1M；时域扫描仍用原 100k–1M 网格。
- Gap 图保留归一化横轴 0–2，并按新的 T 重新计算 S_T/T 和确定性目标奖励 mu_K；在线基线仍无离线日志，Clipped 的 0→0 抑制仍计费。
- 图 1/2/4/5 的 (b) 标签及图 3/6 图注中的固定时域统一从 manifest 读取，不再硬编码 200k。
- 原 T=200k 结果与诊断归档在 history/fixed-T-200k-20260907/。新结果由 paper_runner.py 完整执行十次重复生成。

- 十次重复已完成：1400 条记录中 1320 条实际模拟，80 条 Xu 预算不可行。所有 280 条时域扫描记录与前一版本逐字段一致；K=10、T=1M 的共同配置在两个扫描中的所有数据字段（除 sweep 标记）完全一致。
- UCB 成本分解中，K=5→50 的平均目标成本为 58360.7→146065.0，非目标平均成本为 8565.375→1804.16735，正文同步更新。
- 新网格出现 TS 的 K=5 例外：Ours 平均成本 138205.1，Clipped 为 95594.0，后者目标比例 0.904406。TS 仍在整个时域扫描及 K>=10 时平均成本较低，但不能继续声称两组网格处处较低；附录已改正。UCB 在两组网格上均保持更低平均成本。
- 9 项回归测试通过。60 条 K=30/40/50 基线轨迹由独立归零实现重放，逐臂计数与成本完全一致；所有离线分配在运行时经过证书条件检查。旧 T=200k 的 Direct 独立搜索对照仅保留于历史归档，不当作新时域的独立验证。


2026-09-07：按 Xu 的证明思路适配当前学习器和确定性目标反馈（历史版本，已按用户要求撤回）

- 保留 UCB = mean + 1.5 sqrt(log(t)/N)、Gaussian TS = N(mean,1/N)、Gap 实验固定的真实 mu_K、全程在线且无离线 warm start。
- 两阶段仍是预先固定的 0 向量、目标单位向量；阶段长度与任何已观测行为无关。新增 `xu_budget.py` 给出显式充分预算，附录 `sections/xu_budget_proof.tex` 给出推导。
- UCB：m=ceil(2.25 log(T)/mu_K^2)，C1=K*m，C2=K-1+ceil(mu_K*m/(1-mu_K))。利用探索项上界与确定性反馈证明停止攻击后永远选目标。
- Gaussian TS：delta=0.05 分为三个失败事件；先用零均值高斯最大值概率排序与指数势函数证明第一阶段计数界，再用条件选择概率下界确保第二阶段提升足够，最后控制首次停止攻击后的误选。明确不是直接套用原 Beta–Bernoulli 定理。
- 预算不是经验拟合或最低成功成本；mu_K=0 时公式无定义，C1+C2>=T 时无停止攻击后的评估窗口。不能据此断言所有 observation-free 方法不可能成功。正文原有“requires Theta”已改为对特定充分构造的比较。
- `simulation.py` 支持不同阶段长度，保留旧等长调用兼容性，保存实际阶段一、阶段二、停止攻击后的逐臂计数及后者目标比例。0→0 的计划干预仍计费。
- 原等长 Xu 版本归档于 `history/equal-phase-xu-20260907/`；完整重跑十次重复，1400 条结果中 1300 条实际模拟、100 条充分预算不可行。180 条可行 Xu 轨迹的阶段条件全部满足，停止攻击后的目标比例全部为 1；这不等于用 10 次重复统计验证 95% 定理。
- 所有 1120 条非 Xu 记录与归档逐字段完全一致。`validate_xu_run.py` 校验源哈希、预算版本、所有阶段计数、预算条件和逐行稳定性，报告见 `xu_budget_validation.json`。
- 图示 0–2 网格上，UCB 从 0.4 可行、TS 从 0.75 可行。Gap=1 时 UCB 的 Ours/Xu 成本为 69063.8/116964，TS 为 125151.4/357349。Gap=2 时为 24130.9/29425 和 67314.3/108189。全时域比例与停止攻击后的比例分开报告。
- 11 项回归检查通过，包括不等长阶段的独立 NumPy 轨迹核对、UCB/TS 充分条件与停止攻击后行为、Gaussian 第一阶段最大值概率排序的独立数值积分核对，以及既有数据/时钟/抑制成本检查。
- 六图已重新生成；24 页论文编译完成，无溢出或未定义引用。已逐页视觉核对全部实验图、更新正文及新预算证明。旧诊断和旧运行日志移入上述历史归档；当前结果由新验证报告标识。最终 PDF 已同步至 output/pdf/0-main.pdf。


2026-09-07：直接使用 Xu 原文设置，不新增证明（当前版本）

- 用户要求：能直接用则直接用，否则在 Xu 自己的 settings 下运行。当前 UCB 的探索项与 Gaussian TS 均不直接匹配 Xu 原定理，因此 Xu 基线单独采用原文学习器；Ours/Clipped 保持原设置。
- 删除论文中的独立预算证明及所有对应引用、95% 适配保证、适配预算结果；旧版本完整归档在 history/derived-xu-budgets-20260907/。
- Xu UCB：mean+sqrt(log(T)/N)，按索引逐臂在线初始化；原 bounded-reward 设置允许目标恒等于已知 mu_K。直接使用附录 A.2：C1=K*ceil(log(T)/mu_K²)，C2=ceil(mu_K*C1/(1-mu_K))。
- Xu TS：Beta(sum+1,N-sum+1)，从 Beta(1,1) 先验直接在线开始，无强制逐臂初始化；目标反馈 Bernoulli(mu_K)，非目标仍为 MovieLens 校准的 Bernoulli 臂。预算直接抄自附录 A.4，包含 n1/n2、beta、n3 和 K 相关等待项，阶段长度向上取整。已视觉核对原 PDF 的指数为 1-K。
- 原 TS 有限参数公式在 K=10 下不必随 gap 单调下降：Gap=2 时原式预算为 5,343,203，超过 T=1M；不存在人为修补常数、截断预算或改用新的证明。数据保留明确不可行原因。
- 比较保持共同 K、T、目标均值和非目标 MovieLens 校准参数；明确不同学习器以及 TS 目标反馈模型，不称为相同 learner 下的攻击对比。成本按用户约定继续计算所有计划干预，包括 0→0。
- 完整重跑十次重复：1400 条记录中 1300 条模拟、100 条原式预算不可行；所有 1120 条非 Xu 结果与撤回版本逐字段一致。180 条可行 Xu 轨迹在停止攻击后均只选目标。
- 10 项测试通过，包含原 UCB 与 Beta TS 的独立 NumPy 逐轮参考对照。另运行原文 Table 1 的 K=2、T=50000、均值(0.9,0.8)、C1=34、C2=66 各 10 次实现核对，阶段计数与时域一致；详细结果记录在 xu_budget_validation.json。
- Gap=1 时 UCB 原设置成本为 54519，低于 Ours 的 69063.8；TS 为 146125，高于 Ours 的 125151.4。Gap=0.75 的 TS 原设置成本为 96319，低于 Ours。因此不声称我们的成本在所有 gap 上更低。
- 按用户后续要求，正文删除关于 Xu 最小成功成本的讨论，只保留设置、读图说明及实际比较。21 页最终 PDF 编译无溢出或未定义引用；全部六图及更新正文通过视觉检查，输出已同步。


2026-09-08：新增与 near-boundary 理论对齐的随机反馈实验

- 新增 theory_alignment_runner.py 和 generate_theory_figures.py；原 paper/ 数据、六图和预算构造保留。目标 clean/online 同为 Bernoulli((11/669)*sqrt(100000/T))，攻击只见日志下界。
- 固定 K=10 与增长 K~T^(1/4)，T 扩展到 10^7；每配置十次重复。200 条绘图记录含共享首点，实际运行 180 条顺序轨迹，全部零次非目标在线拉取。
- 两张新图展示各算法自己的 C/Lambda 和 n_K/C，参照为 1 与 2/3 的渐近值；TS 不声称 allocation necessity。UCB 的有限时域比值略低于 1 如实保留。
- 23 项回归与独立轨迹/绘图验证测试通过；默认模拟随机流不变。设置、结果和条件详见 ../output/reviews/theory-aligned-experiments-20260908.md，完整数据与 provenance 位于 results/ml-25m/theory-alignment/。


2026-09-11: Unified zero-feedback Clipped Suppression

- Current UCB, TS, and epsilon-greedy baselines return zero on every non-target online pull and preserve genuine target rewards. They have no offline log; suppression includes initialization. Costs count all non-target pulls, including zero-to-zero replacements.
- Removed the Jun threshold and margin parameters from the active simulator. Prior Jun/Zuo implementations remain only in the historical runners identified in README. The paper now states the explicit zero-feedback rule.
- Replayed all 14 UCB configurations with the original data, seeds, and 50 repeats. The shared configuration across sweeps gives 650 distinct trajectories for 700 CSV records. Every replayed non-target feedback sum is zero.
- All 700 records match the previous results exactly; the entire paper_results.csv SHA-256 is unchanged. The other 6300 records and all TS/EG/near-boundary data are preserved. Regenerated UCB figures consequently have unchanged numerical curves.
- Provenance distinguishes original generation-source hashes from current reproduction-source hashes and records the scoped replay. Snapshot, replay script, and feedback diagnostics: ../output/reviews/zero-suppression-baseline-20260911/.
