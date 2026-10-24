# 从 Flow4DGS 开展静态一致性与动态归属研究

## 基线与本轮范围

- 2026-10-24 检查：本仓库 `main` 与 `wangys16/Flow4DGS-SLAM` 的 `main` 均为
  `ecb4e94c4a3dc3b06430bfb73c528393b3403178`，文件树无差异。
- 研究分支：`research/static-consistency`。基线 main 保持不变。
- 研究假设：区分运动证据、地图失配与未知观测，能够减少错误排除和错误地图分配。
  这是待验证假设，不能由 UP-SLAM 的实验直接推出。
- 当前实现是第一阶段的诊断和跟踪筛选原型，不是 UP-SLAM 复现。
- 当前主线保留语义掩码、光流和原位姿初始化，研究置信度驱动的静态/动态归属与纠错；暂不开展去语义实验。

## 代码中确认的接口

| 位置 | 含义与接入点 |
| --- | --- |
| `utils/dataset.py` | TUM/Bonn 读取 YOLO 与外部 `render_mask`，合并后取反。 |
| `utils/slam_frontend.py:FrontEnd.tracking` | 光流运动分解、位姿初始化、静态高斯渲染和位姿优化。 |
| `utils/slam_backend.py:segment_motion_parametric` | 返回动态掩码、刚体光流、像素残差与 twist。拟合使用输入静态掩码。 |
| `utils/slam_utils.py:get_loss_tracking_rgbd` | 接收额外 `mask`，可单独影响跟踪，无须覆盖相机的地图分配掩码。 |
| `utils/slam_backend.py` 与高斯模型 | 后续建图、动态插入和掩码损失仍消费原来的 `motion_mask`。 |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | 深度为 `sum(z * alpha * T)`，没有除以累积不透明度。 |

注意：`Camera.motion_mask == True` 表示静态/保留；运动分解结果
`results['mask_bool'] == True` 表示动态，两者方向相反。

## 第一阶段：目前已实现

`utils/static_consistency.py` 不依赖光栅化扩展，可在 CPU PyTorch 上测试。

1. 使用历史静态高斯的渲染，按累积不透明度归一化深度，仅用于新检测器。
   不改变原始跟踪损失的深度定义。
2. 用有效深度、有限值与高不透明度建立可比较区域。该区域仅代表可见性代理，
   尚未包含观测次数、几何置信度或位姿协方差，不能等同于可信静态地图。
3. 深度阈值采用 `0.05 m + 0.02 * observed_depth`，颜色阈值采用平均绝对误差
   `0.15`；这些是待调参的初始值，不是从论文搬来的最优值。
4. 区分前景遮挡候选、远处背景显露候选、地图失配及未知区域。
5. 光流异常与地图证据组合成三态诊断：0=未知，1=静态候选，2=动态候选。
   已知地图前后不一致不能单独证明动态；背景显露不直接分配为动态。
   地图无覆盖时可以记录光流动态候选，但其置信度并未校准。
6. `tracking` 模式在初步位姿优化后，仅从原始跟踪支持中排除有效地图上的
   RGB-D 异常；不使用三态结果修改动态高斯归属。
7. 若剩余支持少于 256 像素，或少于原支持的 25%，回退到原始跟踪掩码。
   这是支持数量检查，不保证空间分布、位姿可观测性或恢复成功。

三种模式：

| 模式 | 行为 |
| --- | --- |
| `off` | 默认；不计算新证据，保留基线跟踪路径。 |
| `observe` | 跟踪完成后额外渲染静态地图，记录证据；不修改掩码、损失或收敛条件。 |
| `tracking` | 默认先执行 10 次原始位姿优化，再固定额外筛选掩码用于剩余迭代；最终记录证据。 |

`observe` 增加渲染、同步和磁盘开销，不能用于报告原始基线速度，异步调度也可能受影响。
`tracking` 会在提前收敛时继续到筛选迭代，因此既改变筛选也可能增加迭代数。
正式论文实验应另设相同迭代预算控制组，不能将所有收益直接归因于掩码。
这个版本不循环刷新额外掩码，不做时序积累，不实现高斯状态迁移。

## 环境与首次运行

需要按原 README 准备 GPU/CUDA 环境、两个本地扩展、RAFT 权重与数据。
原 README 指定的 Torch 1.13/CUDA 11.7 与当前 CPU 测试环境不同，完整兼容性尚待验证。

- `pretrained/raft-things.pth`
- `pretrained/yolov9e-seg.pt`（启用 YOLO 时需要）
- `datasets/tum/rgbd_dataset_freiburg3_walking_xyz/`

已有场景 YAML 指向缺失的 `base_config.yaml`，原训练脚本通过 `--inherit_from` 覆盖。
本次新配置直接继承现有 `configs/rgbd/tum/config.yaml`，避免这个问题。
所有命令从仓库根目录执行。先在新配置中设置实际数据路径；测试短序列可在
`Dataset.Calibration` 中设置 `start: 0`、`end: 100`。短序列只用于冒烟测试，不用于比较最终指标。

```bash
# B0：原始语义+光流基线
python slam.py --config configs/experiments/consistency_tum.yaml --consistency-mode off --yolo 1 --seed 0 --exp_name B0

# B1：仅观察，检查残差是否在正确的位置出现
python slam.py --config configs/experiments/consistency_tum.yaml --consistency-mode observe --yolo 1 --seed 0 --exp_name B1

# B2：增加地图失配筛选，仍保留原始语义与光流前端
python slam.py --config configs/experiments/consistency_tum.yaml --consistency-mode tracking --yolo 1 --seed 0 --exp_name B2

```

原型保留了 `--no-semantic-masks` 开关，但不属于当前实验计划，请勿在本轮启用。
首轮在 YAML 中设置 `Results.save_results: false`、`eval_rendering: false`，不传 `--eval`。
当前 `--save_results` 仅在 `--eval` 分支生效；启用完整评估会额外执行渲染评估与颜色精化。

`--seed` 记录并设置主进程种子；不是 CUDA/多进程完全确定性的保证。
正式对比至少运行多个种子，报告均值、离散程度、失败次数与运行环境。
上面 B0 的文件读取路径与新配置保持相同，但不声称比特级复现原论文结果。

## 诊断输出

每次运行的结果目录内新增 `static_consistency/`：

- `summary.jsonl`：逐帧有效地图比例、失配比例、三态占比与筛选是否回退。
  `mask_iteration`、`baseline_support_pixels`、`retained_support_pixels` 来自中途筛选，
  其余占比来自最终位姿下重渲染；两者不是同一个时刻，不能直接逐项相减。
- 默认每 30 帧保存一份 `000030.npz`：三态标签、有效区、深度/颜色归一化残差、
  遮挡候选、背景显露候选、原始静态掩码及可用的光流证据。
- 无效位置的残差保存为 NaN；绘图或统计时必须用 `map_valid` 筛选。
- 诊断与原先 `--save_results` 独立控制。`StaticConsistency.save_every: 0` 关闭密集文件，
  仍写逐帧 JSONL。吞吐量实验应关闭整个新模块或单独统计诊断耗时。

检查重点：新出现背景是否大量误检？位姿偏差是否触发整幅失配？深度边缘是否形成假动态？
掩码改进是否伴随跟踪退化？光流判为静态而地图失配的区域，属于漏检运动、背景显露还是地图误差？

## 第二阶段：时序可信度与证据解释

保留原语义掩码、光流与位姿初始化。只有 B1 的诊断符合物理预期后，才继续：

1. 加入地图观测次数和残差历史，区分地图不成熟与真实场景变化。
2. 加入光流前后向一致性、越界和遮挡检查，明确运动证据是否可靠。
3. 以对应/重投影传播状态，禁止在固定图像坐标直接累计置信度。
4. 设计未知区域的验证与晋升规则，避免未知区永远无法建图。
5. 区分位姿误差引起的全局失配与物体运动引起的局部失配。

此阶段比较原始基线与加入置信度后的结果，检验异常观测识别、跟踪稳定性及地图污染，
不以移除语义模型为目标。

## 第三阶段：跟踪权重与地图归属分离

增加独立字段，而不是改变 `motion_mask` 的布尔语义：

- 跟踪可靠性：决定观测是否参与相机优化。
- 动态归属置信度：决定动态高斯插入与动态监督。
- 地图未知状态：不参与动态正负标签监督，待验证后加入地图。

接着设计静态到动态的迁移、暂时停止时的身份保持、错误动态插入的回收。
状态转换需要时间边界和迟滞机制，不能破坏历史关键帧的动态重建。
此阶段才是“置信度驱动的静态/动态归属与纠错”的主要研究贡献。

## 实验与决策

先用 TUM walking_xyz 与一个 sitting 序列检查跟踪/误检，再加入 Bonn balloon2、
移动箱子和物体离开后背景显露的序列。每次只增加一个机制。

跟踪报告 ATE/RPE、失败率；分割在有独立标注时报告精确率/召回率及背景误检；
重建分别评估静态背景和动态区域，还要统计错误动态高斯插入、状态翻转和恢复时间。
最终同时报告质量、端到端耗时、显存和额外计算量，不能只看全图 PSNR。

注意 UP-SLAM 的干净静态背景评估与 Flow4DGS 的动态重建目标不同，不能直接混用其指标。

## 本轮验证记录

- CPU PyTorch 2.8.0：15 个数值/逻辑测试通过。
- 测试覆盖不透明度归一化、地图空洞、背景显露、无效观测、冲突证据和支持不足回退。
- 本地无可用 CUDA，克隆目录无数据、权重；尚未运行完整 SLAM、GPU 光栅化或轨迹/渲染评估。
- 未复现论文指标，未证明方法优于基线。下一步是有 GPU 环境下先跑 B0/B1 短序列。

```bash
python -m unittest discover -s tests -v
```
