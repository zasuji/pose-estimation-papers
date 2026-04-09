# Pose Estimation Reading List — CVPR / ICCV / ECCV (2023–2024)

> 整理自 CVPR 2023、ICCV 2023、ECCV 2024、CVPR 2024，按**实现方法**分类。  
> 每篇论文附 CVF Open Access / arXiv 链接，方便直接下载 PDF。  
> 「多人支持」列说明：✅ 原生支持多人 | ⚠️ Top-down 管线（依赖外部检测器）| ❌ 仅单人

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
- [12. 手部姿态估计](#12-手部姿态估计)
- [13. 数据集 / Benchmark 类论文](#13-数据集--benchmark-类论文)
- [多人支持汇总](#多人支持汇总)

---

## 多人支持图例

| 标记 | 含义 |
|------|------|
| ✅ 原生多人 | 模型本身直接输出场景中所有人的姿态，无需外部检测器 |
| ⚠️ Top-down | 先用外部检测器裁剪单人 RoI，再逐人估计；可处理多人，但受检测质量影响 |
| ❌ 单人 | 仅针对单一人体设计，不支持多人场景 |

---

## 1. Transformer-based 方法

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | ❌ 单人 | 高效 Video Pose Transformer，Token 剪枝 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/NationalGAILab/HoT) |
| **KTPFormer** | CVPR 2024 | ❌ 单人 | 运动学先验 + Transformer，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_KTPFormer_Kinematics_and_Trajectory_Prior_Knowledge-Enhanced_Transformer_for_3D_Human_Pose_CVPR_2024_paper.html) |
| **Video-Based Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | ⚠️ Top-down | 解耦空间-时间聚合，视频回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **3D Human Pose with Spatio-Temporal Criss-Cross Attention** | CVPR 2023 | ❌ 单人 | 十字注意力机制，时空联合建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html) |
| **GLA-GCN** | ICCV 2023 | ❌ 单人 | 全局/局部自适应图卷积，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose Estimation** | ICCV 2023 | ✅ 原生多人 | 鲁棒多人 3D 姿态，Transformer lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |

> 💡 **规律**：大多数 2D-to-3D lifting 类 Transformer 方法以单人为主（输入已裁剪的单人序列）；若需多人，需在前置 pipeline 中加入检测+追踪。

---

## 2. 基于扩散模型 (Diffusion Model) 的方法

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **DiffPose: Toward More Reliable 3D Pose Estimation** | CVPR 2023 | ❌ 单人 | 扩散模型，3D 姿态，不确定性建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gong_DiffPose_Toward_More_Reliable_3D_Pose_Estimation_CVPR_2023_paper.html) |
| **DiffPose: SpatioTemporal Diffusion for Video-Based Pose** | ICCV 2023 | ⚠️ Top-down | 时空扩散，视频 2D 姿态估计 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **DiffPose: Multi-hypothesis Pose using Diffusion** | ICCV 2023 | ❌ 单人 | 多假设 3D 姿态，Embedding Transformer 条件 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **FinePOSE** | CVPR 2024 | ❌ 单人 | 细粒度文本 Prompt，扩散模型 3D 姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_FinePOSE_Fine-Grained_Prompt-Driven_3D_Human_Pose_Estimation_via_Diffusion_Models_CVPR_2024_paper.html) |
| **DiffusionRegPose** | CVPR 2024 | ✅ 原生多人 | 扩散 + 端到端回归，原生多人输出 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |
| **PostureHMR** | CVPR 2024 | ❌ 单人 | 扩散式运动学正向过程，SMPL mesh | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_PostureHMR_Posture_Transformation_for_3D_Human_Mesh_Recovery_CVPR_2024_paper.html) |
| **PhaseMP** | ICCV 2023 | ❌ 单人 | Phase 条件运动先验，IMU 融合 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_PhaseMP_Robust_3D_Pose_Estimation_via_Phase-conditioned_Human_Motion_Prior_ICCV_2023_paper.html) |

> 💡 **规律**：扩散方法多以单人为主，因为生成过程对输入边界框敏感；**DiffusionRegPose** 是目前少数原生支持多人的扩散姿态方法。

---

## 3. 基于 GCN / 图神经网络的方法

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **GLA-GCN** | ICCV 2023 | ❌ 单人 | 全局局部自适应 GCN，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html) |
| **CheckerPose** | ICCV 2023 | ❌ 单目标 | 渐进式密集关键点 + GNN，**目标物体**姿态 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lian_CheckerPose_Progressive_Dense_Keypoint_Localization_for_Object_Pose_Estimation_with_Graph_ICCV_2023_paper.html) |
| **HiPose** | CVPR 2024 | ❌ 单目标 | 分层 GCN 对应关系剪枝，**6DoF 目标**姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_HiPose_Hierarchical_Binary_Surface_Encoding_and_Correspondence_Pruning_for_RGB-D_CVPR_2024_paper.html) |
| **3D Human Pose via Intuitive Physics** | CVPR 2023 | ❌ 单人 | 物理约束图模型，身体力学先验 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tripathi_3D_Human_Pose_Estimation_via_Intuitive_Physics_CVPR_2023_paper.html) |
| **GATOR** | ECCV 2024 | ❌ 单人 | 图感知 Transformer，运动解耦回归，HMR | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3025_ECCV_2024_paper.php) |

> 💡 **规律**：GCN 方法建模的是**单个人体骨架图**，天然不包含多人之间的关联；多人场景需外部检测器裁剪后逐个处理。

---

## 4. 多视角 3D 姿态估计

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **MVGFormer** | CVPR 2024 | ✅ 原生多人 | 多视角几何 Transformer，三角测量 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/XunshanMan/MVGFormer) |
| **Probabilistic Triangulation for Uncalibrated Multi-View 3D Pose** | ICCV 2023 | ✅ 原生多人 | 无标定多视角，概率三角化，蒙特卡洛 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Single-to-Dual-View for Egocentric 3D Hand Pose** | CVPR 2024 | ❌ 单人手部 | 单→双视角自适应，第一人称手部 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **3D Human Pose from Egocentric Stereo Videos** | CVPR 2024 | ❌ 单人 | 自我中心立体视频，3D 姿态感知 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_3D_Human_Pose_Perception_from_Egocentric_Stereo_Videos_CVPR_2024_paper.html) |
| **Multi-agent Long-term 3D Human Pose Forecasting** | CVPR 2024 | ✅ 原生多人 | 多智能体长期姿态预测，交互轨迹条件 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Vendrow_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html) |

---

## 5. 多人姿态估计（Top-down / Bottom-up / One-stage）

> 本节所有论文均以多人为核心目标。

| 论文 | 会议 | 多人支持 | 范式 | 链接 |
|------|------|:------:|------|------|
| **RTMO** | CVPR 2024 | ✅ 原生多人 | One-stage，无需检测器 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |
| **DiffusionRegPose** | CVPR 2024 | ✅ 原生多人 | 扩散 + 端到端，One-stage | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |
| **BUCTD (Rethinking Pose in Crowds)** | ICCV 2023 | ✅ 原生多人 | Bottom-up 检测器 + Conditional Top-down | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.html) |
| **ED-Pose** | ICCV 2023 | ✅ 原生多人 | 端到端，显式框检测，One-stage | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_ED-Pose_Explicit_Box_Detection_Unifies_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/ED-Pose) |
| **Mutual Information Temporal Difference Learning** | CVPR 2023 | ✅ 原生多人 | Top-down 多人视频姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.html) |
| **Multi-HMR** | ECCV 2024 | ✅ 原生多人 | 单次全身多人网格，ViT + Cross-Attention | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |

---

## 6. 体网格恢复 (Human Mesh Recovery, HMR)

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **TokenHMR** | CVPR 2024 | ⚠️ Top-down | Tokenized 姿态表示，SMPL | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a_Tokenized_Pose_Representation_CVPR_2024_paper.html) · [GitHub](https://github.com/saidwivedi/TokenHMR) |
| **ScoreHypo** | CVPR 2024 | ⚠️ Top-down | 概率假设评分，HMR 不确定性 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **PostureHMR** | CVPR 2024 | ❌ 单人 | 扩散式运动学正向过程，SMPL mesh | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_PostureHMR_Posture_Transformation_for_3D_Human_Mesh_Recovery_CVPR_2024_paper.html) |
| **Multi-HMR** | ECCV 2024 | ✅ 原生多人 | 单次全身多人 SMPL-X，ViT + HPH | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **Multi-RoI HMR** | ECCV 2024 | ⚠️ Top-down | 多 RoI 相机一致性约束，对比学习 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_25) |
| **VQ-HPS** | ECCV 2024 | ❌ 单人 | VQ-VAE 离散潜空间，分类式 HMR | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0692_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2312.08291) |
| **HaMeR** | CVPR 2024 | ⚠️ Top-down | 全 Transformer 手部网格恢复 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html) · [GitHub](https://github.com/geopavlakos/hamer) |
| **POTTER** | CVPR 2023 | ⚠️ Top-down | 高效池化注意力 Transformer，HMR | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_POTTER_Pooling_Attention_Transformer_for_Efficient_Human_Mesh_Recovery_CVPR_2023_paper.html) |
| **3D Human Mesh from Virtual Markers** | CVPR 2023 | ❌ 单人 | 虚拟标记点，非参数网格估计 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ma_3D_Human_Mesh_Estimation_From_Virtual_Markers_CVPR_2023_paper.html) |

> 💡 **规律**：HMR 方法中，**Multi-HMR** 是目前最强的原生多人全身方法；其他大多数 HMR 方法依赖外部检测器做 Top-down 处理。

---

## 7. 视频时序姿态估计

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | ❌ 单人 | 视频 Pose Transformer，高效 Token 管理 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) |
| **Video-Based Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | ⚠️ Top-down | 空间-时间解耦聚合，视频回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **DiffPose: SpatioTemporal Diffusion** | ICCV 2023 | ⚠️ Top-down | 时空扩散过程，视频 2D 多人 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose** | ICCV 2023 | ✅ 原生多人 | 鲁棒平滑 3D 多人视频姿态 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |
| **Mutual Information Temporal Difference Learning** | CVPR 2023 | ✅ 原生多人 | 互信息时序差分，视频姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.html) |

---

## 8. 实时 / 轻量化方法

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **RTMPose** | arXiv 2023 | ⚠️ Top-down | SimCC，实时，跨平台部署 | [arXiv](https://arxiv.org/abs/2303.07399) · [GitHub](https://github.com/open-mmlab/mmpose) |
| **RTMO** | CVPR 2024 | ✅ 原生多人 | 单阶段，无检测器，高实时性 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) |
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | ❌ 单人 | -50% FLOPs，plug-and-play | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) |
| **HMD-Poser** | CVPR 2024 | ❌ 单人 | 端侧实时，稀疏传感器 IMU | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_HMD-Poser_On-Device_Real-time_Human_Motion_Tracking_from_Scalable_Sparse_Observations_CVPR_2024_paper.html) |
| **DWPose** | ICCV 2023 | ⚠️ Top-down | 两阶段知识蒸馏，全身高效推理 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |

> 💡 **对比**：需要实时多人场景，**RTMO** 是首选（无外部检测器）；对精度要求更高但可接受两阶段，**RTMPose + 检测器** 是工业界主流方案。

---

## 9. 领域自适应与跨域泛化

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **POST: Prior-guided Source-free Domain Adaptation** | ICCV 2023 | ⚠️ Top-down | 无源域自适应，先验自训练 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Raychaudhuri_Prior-guided_Source-free_Domain_Adaptation_for_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Single-to-Dual-View for Egocentric Hand Pose** | CVPR 2024 | ❌ 单人手部 | 单→双视图自适应，手部姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **UPose3D: Uncertainty-Aware 3D Human Pose** | ECCV 2024 | ⚠️ Top-down | 不确定性感知，跨视角时序线索 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0241_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2404.14634) |

---

## 10. 概率姿态估计

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **DiffPose: Multi-hypothesis Pose using Diffusion** | ICCV 2023 | ❌ 单人 | 多假设生成，Embedding Transformer | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **ScoreHypo** | CVPR 2024 | ⚠️ Top-down | 假设评分，概率 HMR | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **Normalizing Flows on SO(3) Manifolds** | CVPR 2024 | ❌ 单人 | SO(3) 流形归一化流，概率分布 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jaini_Normalizing_Flows_on_the_Product_Space_of_SO3_Manifolds_for_Probabilistic_Human_Pose_Modeling_CVPR_2024_paper.html) |
| **Probabilistic Triangulation for Multi-View 3D Pose** | ICCV 2023 | ✅ 原生多人 | 概率三角化，无标定多视角 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |

---

## 11. 全身姿态估计 (Whole-body)

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **DWPose** | ICCV 2023 | ⚠️ Top-down | 两阶段蒸馏，全身 133 关键点 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |
| **Multi-HMR** | ECCV 2024 | ✅ 原生多人 | 单次多人全身 SMPL-X，ViT + HPH | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **One-Stage 3D Whole-Body Mesh with CAT** | CVPR 2023 | ❌ 单人 | 单阶段全身网格，组件感知 Transformer | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_with_Component_Aware_Transformer_CVPR_2023_paper.html) |
| **ViTPose+** | ICCV 2023 | ⚠️ Top-down | 通用体态估计，ViT 基础模型，异构关键点 | [arXiv](https://arxiv.org/abs/2212.04246) · [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |

---

## 12. 手部姿态估计

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **HaMeR** | CVPR 2024 | ⚠️ Top-down | 全 Transformer，3D 手部网格，in-the-wild | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html) · [GitHub](https://github.com/geopavlakos/hamer) |
| **Single-to-Dual-View for Egocentric Hand Pose** | CVPR 2024 | ❌ 单人手部 | 单→双视图，第一视角 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **Attention-Propagation for Egocentric Heatmap to 3D Lifting** | CVPR 2024 | ❌ 单人手部 | 注意力传播，热图→3D | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kang_Attention-Propagation_Network_for_Egocentric_Heatmap_to_3D_Pose_Lifting_CVPR_2024_paper.html) |

---

## 13. 数据集 / Benchmark 类论文

| 论文 | 会议 | 多人支持 | 关键词 | 链接 |
|------|------|:------:|--------|------|
| **Human-Art** | CVPR 2023 | ✅ 多人数据 | 艺术风格人体图像，多场景泛化基准 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/HumanArt) |
| **EventEgo3D** | CVPR 2024 | ❌ 单人 | 事件相机，自我中心运动捕获 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Millerdurai_EventEgo3D_3D_Human_Motion_Capture_from_Egocentric_Event_Streams_CVPR_2024_paper.html) |
| **Person-in-WiFi 3D** | CVPR 2024 | ✅ 原生多人 | WiFi 信号多人 3D 姿态，新模态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html) |

---

## 多人支持汇总

> 快速选型参考表。

| 多人能力 | 论文 | 适用场景建议 |
|---------|------|------------|
| ✅ **原生多人** | RTMO, ED-Pose, BUCTD, Multi-HMR, DiffusionRegPose, MVGFormer, Probabilistic Triangulation, Multi-agent Forecasting, Person-in-WiFi 3D | 直接输出所有人姿态，无外部检测依赖，适合拥挤场景或端到端部署 |
| ⚠️ **Top-down**（依赖检测器） | RTMPose, DWPose, ViTPose+, TokenHMR, HaMeR, POTTER, ScoreHypo, POST, UPose3D, Multi-RoI HMR, 视频扩散DiffPose | 先检测人体框再逐人估计；灵活可替换检测器，精度通常更高 |
| ❌ **仅单人** | HoT, KTPFormer, GLA-GCN, DiffPose系列(3D lifting), FinePOSE, VQ-HPS, PostureHMR, PhaseMP, HMD-Poser, 手部姿态系列 | 研究单人精度/概率建模/轻量化/特殊传感器场景 |

---

## 参考资源

- [CVF Open Access (CVPR/ICCV)](https://openaccess.thecvf.com/)
- [ECCV 2024 论文集 (ECVA)](https://www.ecva.net/eccv2024.php)
- [52CV CVPR-2024-Papers](https://github.com/52CV/CVPR-2024-Papers)
- [52CV ICCV-2023-Papers](https://github.com/52CV/ICCV-2023-Papers)
- [52CV ECCV-2024-Papers](https://github.com/52CV/ECCV-2024-Papers)
- [MMPose](https://github.com/open-mmlab/mmpose)

---

*Last updated: 2025-04 | 覆盖 CVPR 2023 · ICCV 2023 · CVPR 2024 · ECCV 2024*
