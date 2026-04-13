# Human Pose Estimation Reading List — CVPR / ICCV / ECCV (2023–2024)

> 整理自 CVPR 2023、ICCV 2023、ECCV 2024、CVPR 2024，按**实现方法**分类。  
> 范围：**人体姿态估计**（body pose estimation），不含物体 6DoF 姿态。  
> 每篇论文附 CVF Open Access / arXiv 链接。  
> 「多人支持」列：✅ 原生多人 | ⚠️ Top-down（依赖外部检测器）| ❌ 仅单人

---

## 修订说明（v3）

**v2 → v3 新增论文（二次系统检索后补充）：**

- **新增** MotionBERT（ICCV 2023）：预训练动作编码基础模型，→ 第 1 节 Transformer
- **新增** PoseFormerV2（CVPR 2023）：频域表示提升 3D lifting 效率，→ 第 1 节 Transformer
- **新增** HopFIR（ICCV 2023）：跳跃式 GraphFormer + 组内关节细化，→ 第 1 节 Transformer
- **新增** GFPose（CVPR 2023）：梯度场 score-based pose 先验，→ 第 10 节概率估计
- **新增** Group Pose（ICCV 2023）：端到端多人 One-stage baseline，→ 第 5 节多人
- **新增** Co-Evolution of Pose and Mesh（ICCV 2023）：视频中 pose 与 mesh 联合优化，→ 第 6 节 HMR
- **新增** EgoPoseFormer（ECCV 2024）：立体自我中心 3D pose 简单 baseline，→ 第 4 节多视角
- **新增** EMDB（ICCV 2023）：IMU + 视频全局 3D pose 数据集，→ 第 12 节数据集
- **新增** FreeMan（CVPR 2024）：真实场景多视角 3D pose 基准，→ 第 12 节数据集
- **移动** IPMAN（CVPR 2023）：从「GCN」→「HMR」（物理约束增强 HMR，基于 SMPL 回归，无 GCN）

**v1 → v2 改动：** 删除非人体论文（CheckerPose、HiPose）、删除手部姿态节、修正分类错误（GLA-GCN、PhaseMP、GATOR、PostureHMR、Multi-agent Forecasting）、整合跨节重复。

---

## 目录

- [1. Transformer-based 方法](#1-transformer-based-方法)
- [2. 基于扩散模型 (Diffusion Model) 的方法](#2-基于扩散模型-diffusion-model-的方法)
- [3. 基于 GCN / 图神经网络的方法](#3-基于-gcn--图神经网络的方法)
- [4. 多视角 3D 姿态估计](#4-多视角-3d-姿态估计)
- [5. 多人姿态估计（Top-down / Bottom-up / One-stage）](#5-多人姿态估计top-down--bottom-up--one-stage)
- [6. 体网格恢复 (Human Mesh Recovery, HMR)](#6-体网格恢复-human-mesh-recovery-hmr)
- [7. 视频时序姿态估计](#7-视频时序姿态估计)
- [8. 实时 / 轻量化方法](#8-实时--轻量化方法)
- [9. 领域自适应与跨域泛化](#9-领域自适应与跨域泛化)
- [10. 概率姿态估计](#10-概率姿态估计)
- [11. 全身姿态估计 (Whole-body)](#11-全身姿态估计-whole-body)
- [12. 数据集 / Benchmark 类论文](#12-数据集--benchmark-类论文)
- [多人支持汇总](#多人支持汇总)

---

## 1. Transformer-based 方法

> 以 Transformer / Attention 机制为核心，建模关节空间或时序依赖。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | ❌ 单人 | 高效 Video Pose Transformer，Token 剪枝，→ 亦见第 8 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/NationalGAILab/HoT) |
| **KTPFormer** | CVPR 2024 | ❌ 单人 | 运动学先验 + Transformer，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_KTPFormer_Kinematics_and_Trajectory_Prior_Knowledge-Enhanced_Transformer_for_3D_Human_Pose_CVPR_2024_paper.html) |
| **Video-Based Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | ⚠️ Top-down | 解耦空间-时间聚合，视频回归，→ 亦见第 7 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **3D Human Pose with Spatio-Temporal Criss-Cross Attention** | CVPR 2023 | ❌ 单人 | 十字注意力机制，时空联合建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose Estimation** | ICCV 2023 | ✅ 原生多人 | 鲁棒多人 3D 姿态，Transformer lifting，→ 亦见第 7 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |
| **PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation** | CVPR 2023 | ❌ 单人 | 频域紧凑表示骨架序列，兼顾长序列效率与噪声鲁棒性，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_PoseFormerV2_Exploring_Frequency_Domain_for_Efficient_and_Robust_3D_Human_Pose_Estimation_CVPR_2023_paper.html) · [GitHub](https://github.com/QitaoZhao/PoseFormerV2) |
| **MotionBERT: A Unified Perspective on Learning Human Motion Representations** | ICCV 2023 | ❌ 单人（预训练后可迁移） | BERT 式预训练动作编码器，统一下游任务（3D pose / HMR / 动作识别），→ 亦见第 6 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html) · [GitHub](https://github.com/Walter0807/MotionBERT) |
| **HopFIR: Hop-wise GraphFormer with Intragroup Joint Refinement for 3D Human Pose Estimation** | ICCV 2023 | ❌ 单人 | 跳跃式图感知 Transformer，组内关节细化，Graph-Transformer 混合 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_HopFIR_Hop-Wise_GraphFormer_with_Intragroup_Joint_Refinement_for_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |

> 💡 **规律**：大多数 2D-to-3D lifting 类 Transformer 方法以单人为主，输入已裁剪的单人关节序列；若需多人，需在前置 pipeline 中加入检测+追踪。**MotionBERT** 是当前最重要的通用预训练运动表示，建议优先阅读。

---

## 2. 基于扩散模型 (Diffusion Model) 的方法

> 将 DDPM 去噪扩散过程引入姿态估计，实现概率建模或不确定性量化。  
> ⚠️ 注意：并非所有使用「先验」或「多步过程」的方法都属于扩散模型，需确认是否基于 DDPM。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **DiffPose: Toward More Reliable 3D Pose Estimation** | CVPR 2023 | ❌ 单人 | 扩散模型，3D 姿态，不确定性建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gong_DiffPose_Toward_More_Reliable_3D_Pose_Estimation_CVPR_2023_paper.html) |
| **DiffPose: SpatioTemporal Diffusion for Video-Based Pose** | ICCV 2023 | ⚠️ Top-down | 时空扩散，视频 2D 姿态估计，→ 亦见第 7 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **DiffPose: Multi-hypothesis Pose using Diffusion** | ICCV 2023 | ❌ 单人 | 多假设 3D 姿态，Embedding Transformer 条件，→ 亦见第 10 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **FinePOSE** | CVPR 2024 | ❌ 单人 | 细粒度文本 Prompt，扩散模型 3D 姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_FinePOSE_Fine-Grained_Prompt-Driven_3D_Human_Pose_Estimation_via_Diffusion_Models_CVPR_2024_paper.html) |
| **DiffusionRegPose** | CVPR 2024 | ✅ 原生多人 | 扩散 + 端到端回归，原生多人，→ 亦见第 5 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |

> 💡 **规律**：扩散模型方法多以单人为主；**DiffusionRegPose** 是目前少数原生支持多人的扩散姿态方法。推理速度是此类方法的主要瓶颈（多步去噪），不适合实时部署。

---

## 3. 基于 GCN / 图神经网络的方法

> 将人体骨架显式建模为图结构，用图卷积捕捉关节拓扑依赖。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **GLA-GCN: Global-local Adaptive Graph Convolutional Network** | ICCV 2023 | ❌ 单人 | 全局/局部自适应 GCN，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html) |

> 💡 **规律**：独立 GCN 方法在 2024 年顶会中明显减少，逐渐被融合进 Transformer 架构（如 GATOR）。GCN 建模单个骨架图，多人场景需外部检测器逐人处理。

---

## 4. 多视角 3D 姿态估计

> 利用多相机几何约束进行更精确的 3D 关节位置估计。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **MVGFormer: Multiple View Geometry Transformers** | CVPR 2024 | ✅ 原生多人 | 多视角几何 Transformer，三角测量，外观+几何双模块 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/XunshanMan/MVGFormer) |
| **Probabilistic Triangulation for Uncalibrated Multi-View 3D Pose** | ICCV 2023 | ✅ 原生多人 | 无标定多视角，概率三角化，蒙特卡洛后验，→ 亦见第 10 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **3D Human Pose Perception from Egocentric Stereo Videos** | CVPR 2024 | ❌ 单人 | 自我中心立体视频，双目几何，第一人称 3D 姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_3D_Human_Pose_Perception_from_Egocentric_Stereo_Videos_CVPR_2024_paper.html) |
| **EgoPoseFormer: A Simple Baseline for Stereo Egocentric 3D Human Pose Estimation** | ECCV 2024 | ❌ 单人 | 立体自我中心 3D pose 简单 Transformer baseline，UnrealEgo 数据集 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0966_ECCV_2024_paper.php) · [GitHub](https://github.com/ChenhongyiYang/egoposeformer) |

> 💡 **规律**：多视角方法在精度上有优势，但推理速度受视角数量和跨视角注意力复杂度影响显著（O(N²) 瓶颈）。无标定方法（如 Probabilistic Triangulation）泛化性更强但精度略低。

---

## 5. 多人姿态估计（Top-down / Bottom-up / One-stage）

> 以同时估计场景中多人关节为核心目标。

| 论文 | 会议 | 多人支持 | 范式 | 链接 |
|------|------|:------:|------|------|
| **RTMO: High-Performance One-Stage Real-Time Multi-Person Pose** | CVPR 2024 | ✅ 原生多人 | One-stage，无需检测器，→ 亦见第 8 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |
| **DiffusionRegPose** | CVPR 2024 | ✅ 原生多人 | 扩散 + 端到端，One-stage，→ 亦见第 2 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |
| **BUCTD: Rethinking Pose Estimation in Crowds** | ICCV 2023 | ✅ 原生多人 | Bottom-up 检测器 + Conditional Top-down，拥挤场景 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.html) |
| **ED-Pose: Explicit Box Detection Unifies End-to-End Multi-Person Pose** | ICCV 2023 | ✅ 原生多人 | 端到端，显式框检测，One-stage | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_ED-Pose_Explicit_Box_Detection_Unifies_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/ED-Pose) |
| **Group Pose: A Simple Baseline for End-to-End Multi-Person Pose Estimation** | ICCV 2023 | ✅ 原生多人 | 端到端多人 pose，组查询设计，与 ED-Pose 同期 baseline | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Group_Pose_A_Simple_Baseline_for_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html) |

> 💡 **趋势**：2024 年 One-stage 方法（RTMO、DiffusionRegPose）明显增多，核心动机是消除对外部检测器的依赖，实现真正的端到端训练。

---

## 6. 体网格恢复 (Human Mesh Recovery, HMR)

> 从 2D 图像恢复 SMPL/SMPL-X 参数或 3D 体表面网格（6890 顶点），比关节点估计包含更多体型信息。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **TokenHMR: Advancing HMR with a Tokenized Pose Representation** | CVPR 2024 | ⚠️ Top-down | Tokenized 姿态表示，SMPL 参数回归稳定性 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a_Tokenized_Pose_Representation_CVPR_2024_paper.html) · [GitHub](https://github.com/saidwivedi/TokenHMR) |
| **ScoreHypo: Probabilistic Human Mesh Estimation with Hypothesis Scoring** | CVPR 2024 | ⚠️ Top-down | 多假设生成 + 评分选优，概率 HMR，→ 亦见第 10 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **PostureHMR: Posture Transformation for 3D Human Mesh Recovery** | CVPR 2024 | ❌ 单人 | 扩散式运动学正向过程（kinematics-based），SMPL T-pose→目标姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_PostureHMR_Posture_Transformation_for_3D_Human_Mesh_Recovery_CVPR_2024_paper.html) |
| **Multi-HMR: Multi-Person Whole-Body HMR in a Single Shot** | ECCV 2024 | ✅ 原生多人 | 单次全身多人 SMPL-X，ViT + Cross-Attention HPH，→ 亦见第 11 节 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **Multi-RoI HMR with Camera Consistency and Contrastive Losses** | ECCV 2024 | ⚠️ Top-down | 多 RoI 相机一致性约束，对比学习缓解深度模糊 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_25) |
| **VQ-HPS: HMR in a Vector-Quantized Latent Space** | ECCV 2024 | ❌ 单人 | VQ-VAE 离散潜空间，将 HMR 转为分类问题 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0692_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2312.08291) |
| **GATOR: Graph-Aware Transformer with Motion-Disentangled Regression** | ECCV 2024 | ❌ 单人 | 图感知 Transformer，运动解耦回归，图结构作为归纳偏置 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3025_ECCV_2024_paper.php) |
| **Co-Evolution of Pose and Mesh for 3D Human Body Estimation from Video** | ICCV 2023 | ⚠️ Top-down | 视频中 pose 与 mesh 协同优化，迭代细化，时序一致性 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Co-Evolution_of_Pose_and_Mesh_for_3D_Human_Body_Estimation_from_Video_ICCV_2023_paper.html) |
| **POTTER: Pooling Attention Transformer for Efficient HMR** | CVPR 2023 | ⚠️ Top-down | 高效池化注意力 Transformer，降低 HMR 计算量 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_POTTER_Pooling_Attention_Transformer_for_Efficient_Human_Mesh_Recovery_CVPR_2023_paper.html) |
| **3D Human Mesh Estimation from Virtual Markers** | CVPR 2023 | ❌ 单人 | 虚拟标记点中间表示，非参数化网格估计 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ma_3D_Human_Mesh_Estimation_From_Virtual_Markers_CVPR_2023_paper.html) |
| **IPMAN: 3D Human Pose Estimation via Intuitive Physics** | CVPR 2023 | ❌ 单人 | 物理约束增强 HMR，压力热图 + 质心/压力中心（CoP/CoM）可微分约束，SMPL 参数回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tripathi_3D_Human_Pose_Estimation_via_Intuitive_Physics_CVPR_2023_paper.html) · [GitHub](https://github.com/sha2nkt/ipman-r) |

> 💡 **规律**：HMR 是近两年论文数量最多的子任务。**Multi-HMR** 是目前最强的原生多人全身方法；大多数 HMR 方法依赖 Top-down 管线。SMPL-X（含手+面部）正在成为新标准。

---

## 7. 视频时序姿态估计

> 利用视频帧序列的时序信息提升 2D/3D 姿态精度，或对未来姿态进行预测。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Mutual Information-Based Temporal Difference Learning** | CVPR 2023 | ✅ 原生多人 | 互信息时序差分学习，视频多人 2D 姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.html) |
| **DiffPose: SpatioTemporal Diffusion for Video-Based Pose** | ICCV 2023 | ⚠️ Top-down | 时空扩散，视频 2D 姿态，→ 亦见第 2 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose from Monocular Video** | ICCV 2023 | ✅ 原生多人 | 鲁棒平滑多人视频 3D 姿态，→ 亦见第 1 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |
| **Video-Based Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | ⚠️ Top-down | 空间-时间解耦聚合，视频回归，→ 亦见第 1 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **PhaseMP: Robust 3D Pose via Phase-conditioned Human Motion Prior** | ICCV 2023 | ❌ 单人 | 相位条件运动先验（非扩散模型），IMU 稀疏传感器融合 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_PhaseMP_Robust_3D_Pose_Estimation_via_Phase-conditioned_Human_Motion_Prior_ICCV_2023_paper.html) |
| **Multi-agent Long-term 3D Human Pose Forecasting** | CVPR 2024 | ✅ 原生多人 | **姿态预测任务**（Forecasting），多智能体交互，轨迹条件 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Vendrow_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html) |
| **HDG-ODE: A Hierarchical Continuous-Time Model for Human Pose Forecasting** | ICCV 2023 | ✅ 原生多人 | **姿态预测任务**（Forecasting），层次动态图 + Neural ODE 连续时间建模，多人遮挡鲁棒 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Xing_HDG-ODE_A_Hierarchical_Continuous-Time_Model_for_Human_Pose_Forecasting_ICCV_2023_paper.html) · [PapersWithCode](https://paperswithcode.com/paper/hdg-ode-a-hierarchical-continuous-time-model) |

> 💡 注意：Multi-agent Forecasting 和 HDG-ODE 的任务均为**预测未来帧姿态**（Pose Forecasting），输入为历史骨架序列而非图像，与其他从图像估计姿态的方法在任务定义上有本质差异，阅读时注意区分。HDG-ODE 用 Neural ODE 替代离散时间步建模，理论上可在任意时间点查询姿态。

---

## 8. 实时 / 轻量化方法

> 面向部署的高效推理，关注 FPS、参数量与精度的平衡。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose** | arXiv 2023 | ⚠️ Top-down | SimCC 坐标编码，实时，跨平台部署（CPU/GPU/移动端） | [arXiv](https://arxiv.org/abs/2303.07399) · [GitHub](https://github.com/open-mmlab/mmpose) |
| **RTMO: High-Performance One-Stage Real-Time Multi-Person Pose** | CVPR 2024 | ✅ 原生多人 | 单阶段，无检测器，高实时性，→ 亦见第 5 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) |
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | ❌ 单人 | Token 剪枝，-50% FLOPs，plug-and-play 框架，→ 亦见第 1 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/NationalGAILab/HoT) |
| **HMD-Poser: On-Device Real-time Human Motion Tracking** | CVPR 2024 | ❌ 单人 | 端侧实时，稀疏 IMU 传感器，可扩展稀疏观测 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_HMD-Poser_On-Device_Real-time_Human_Motion_Tracking_from_Scalable_Sparse_Observations_CVPR_2024_paper.html) |
| **DWPose: Effective Whole-body Pose with Two-Stage Distillation** | ICCV 2023 | ⚠️ Top-down | 两阶段知识蒸馏，全身高效推理，→ 亦见第 11 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |

> 💡 **选型对比**：实时多人场景首选 **RTMO**（无检测器，One-stage）；对精度要求高且可接受两阶段，**RTMPose + 检测器** 是工业主流方案；3D 单人需要压缩 FLOPs 可用 **HoT** 作 plug-in 插件。

---

## 9. 领域自适应与跨域泛化

> 在无源域数据或目标域标注不足时，提升姿态估计模型的跨域泛化能力。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **POST: Prior-guided Source-free Domain Adaptation for Human Pose** | ICCV 2023 | ⚠️ Top-down | 无源域自适应，先验自训练回归框架（POST） | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Raychaudhuri_Prior-guided_Source-free_Domain_Adaptation_for_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **UPose3D: Uncertainty-Aware 3D Human Pose with Cross-View Cues** | ECCV 2024 | ⚠️ Top-down | 不确定性感知，跨视角时序线索，泛化性增强 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0241_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2404.14634) |

> 💡 该方向论文数量相对少，但与跨主体泛化问题密切相关。POST 的「无源域自适应」范式和跨主体的迁移学习有方法论上的相通之处。

---

## 10. 概率姿态估计

> 从单一图像生成多个合理的 3D 姿态假设，显式建模深度模糊性和不确定性。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **GFPose: Learning 3D Human Pose Prior with Gradient Fields** | CVPR 2023 | ❌ 单人 | Score-based 梯度场建模 pose 分布，层次条件掩码策略，支持多种下游任务 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ci_GFPose_Learning_3D_Human_Pose_Prior_With_Gradient_Fields_CVPR_2023_paper.html) |
| **DiffPose: Multi-hypothesis Pose using Diffusion** | ICCV 2023 | ❌ 单人 | 多假设生成，Embedding Transformer 条件，→ 主类见第 2 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **Normalizing Flows on SO(3) Manifolds for Probabilistic Pose Modeling** | CVPR 2024 | ❌ 单人 | SO(3) 流形上的归一化流，旋转空间概率分布建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jaini_Normalizing_Flows_on_the_Product_Space_of_SO3_Manifolds_for_Probabilistic_Human_Pose_Modeling_CVPR_2024_paper.html) |
| **ScoreHypo: Probabilistic HMR with Hypothesis Scoring** | CVPR 2024 | ⚠️ Top-down | 假设评分选优，概率 HMR，→ 主类见第 6 节 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **Probabilistic Triangulation for Uncalibrated Multi-View 3D Pose** | ICCV 2023 | ✅ 原生多人 | 概率三角化，贝叶斯相机位姿分布，→ 主类见第 4 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |

---

## 11. 全身姿态估计 (Whole-body)

> 同时估计身体、面部关键点的整体姿态（133+ 关键点或 SMPL-X 参数）。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **DWPose: Effective Whole-body Pose with Two-Stage Distillation** | ICCV 2023 | ⚠️ Top-down | 两阶段蒸馏，全身 133 关键点，轻量高效，→ 亦见第 8 节 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |
| **Multi-HMR: Multi-Person Whole-Body HMR in a Single Shot** | ECCV 2024 | ✅ 原生多人 | 单次全身多人 SMPL-X，ViT + HPH 模块，→ 主类见第 6 节 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **One-Stage 3D Whole-Body Mesh with Component Aware Transformer (CAT)** | CVPR 2023 | ❌ 单人 | 单阶段全身网格，组件感知 Transformer，身体/手/面部统一 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_with_Component_Aware_Transformer_CVPR_2023_paper.html) |
| **ViTPose+: Vision Transformer Foundation Model for Generic Body Pose** | ICCV 2023 | ⚠️ Top-down | 通用体态估计基础模型，支持异构关键点集，多任务 | [arXiv](https://arxiv.org/abs/2212.04246) · [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |

---

## 12. 数据集 / Benchmark 类论文

> 提出新数据集或评估基准，直接推动该领域的量化进展。

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Human-Art: A Versatile Human-Centric Dataset** | CVPR 2023 | ✅ 多人标注 | 艺术风格人体图像，自然场景+艺术场景联合基准 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/HumanArt) |
| **EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild** | ICCV 2023 | ❌ 单人 | IMU + 单目视频全局 3D pose 数据集，真实场景全局轨迹标注 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kaufmann_EMDB_The_Electromagnetic_Database_of_Global_3D_Human_Pose_and_Shape_in_ICCV_2023_paper.html) |
| **EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams** | CVPR 2024 | ❌ 单人 | 事件相机新模态，自我中心运动捕获数据集 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Millerdurai_EventEgo3D_3D_Human_Motion_Capture_from_Egocentric_Event_Streams_CVPR_2024_paper.html) |
| **FreeMan: Towards Benchmarking 3D Human Pose Estimation under Real-World Conditions** | CVPR 2024 | ✅ 多人标注 | 大规模室外多视角 3D pose 基准，真实复杂场景，多视角同步 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_FreeMan_Towards_Benchmarking_3D_Human_Pose_Estimation_under_Real-World_Conditions_CVPR_2024_paper.html) |
| **Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose with Wi-Fi** | CVPR 2024 | ✅ 原生多人 | WiFi 信号作为新型传感模态，多人 3D 姿态数据集 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html) |

---

## 多人支持汇总

> 快速选型参考。

| 多人能力 | 论文 | 适用场景建议 |
|---------|------|------------|
| ✅ **原生多人** | RTMO, ED-Pose, BUCTD, DiffusionRegPose, MVGFormer, Probabilistic Triangulation, Multi-HMR, Mutual Info Temporal, Person-in-WiFi 3D | 直接输出所有人姿态，无外部检测依赖，适合拥挤场景或端到端部署 |
| ⚠️ **Top-down**（依赖检测器） | RTMPose, DWPose, ViTPose+, TokenHMR, POTTER, ScoreHypo, POST, UPose3D, Multi-RoI HMR, DiffPose(video) | 先检测再逐人估计；灵活可替换检测器，精度通常更高 |
| ❌ **仅单人** | HoT, KTPFormer, GLA-GCN, DiffPose(3D lifting), FinePOSE, VQ-HPS, PostureHMR, PhaseMP, HMD-Poser, Normalizing Flows | 研究单人精度/概率建模/轻量化/特殊传感器场景 |

---

## 参考资源

- [CVF Open Access (CVPR/ICCV)](https://openaccess.thecvf.com/)
- [ECCV 2024 论文集 (ECVA)](https://www.ecva.net/eccv2024.php)
- [52CV CVPR-2024-Papers](https://github.com/52CV/CVPR-2024-Papers)
- [52CV ICCV-2023-Papers](https://github.com/52CV/ICCV-2023-Papers)
- [52CV ECCV-2024-Papers](https://github.com/52CV/ECCV-2024-Papers)
- [MMPose](https://github.com/open-mmlab/mmpose)

---

*Last updated: 2025-04 | v3 | 覆盖 CVPR 2023 · ICCV 2023 · CVPR 2024 · ECCV 2024*  
*范围：人体姿态估计（body pose），不含物体 6DoF 姿态 / 手部单独估计*
