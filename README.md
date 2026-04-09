# Pose Estimation Reading List — CVPR / ICCV / ECCV (2023–2024)

> 整理自 CVPR 2023、ICCV 2023、ECCV 2024、CVPR 2024，按**实现方法**分类。  
> 每篇论文附 CVF Open Access / arXiv 链接，方便直接下载 PDF。  
> 持续更新中，欢迎 PR。

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

---

## 1. Transformer-based 方法

> 以 Transformer / Attention 机制为核心架构，建模关节空间/时序依赖。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **Hourglass Tokenizer (HoT): Efficient Transformer-Based 3D Human Pose Estimation** | CVPR 2024 | 高效 Video Pose Transformer，Token 剪枝 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/NationalGAILab/HoT) |
| **KTPFormer: Kinematics and Trajectory Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation** | CVPR 2024 | 运动学先验 + Transformer，时序建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_KTPFormer_Kinematics_and_Trajectory_Prior_Knowledge-Enhanced_Transformer_for_3D_Human_Pose_CVPR_2024_paper.html) |
| **Video-Based Human Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | 解耦空间-时间聚合，视频回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **3D Human Pose Estimation with Spatio-Temporal Criss-Cross Attention** | CVPR 2023 | 十字注意力机制，时空联合建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html) |
| **GLA-GCN: Global-local Adaptive Graph Convolutional Network for 3D Human Pose Estimation** | ICCV 2023 | 全局/局部自适应图卷积 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Video** | ICCV 2023 | 鲁棒多人 3D 姿态，Transformer lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |

---

## 2. 基于扩散模型 (Diffusion Model) 的方法

> 将去噪扩散过程引入姿态估计，实现概率建模或生成式姿态估计。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **DiffPose: Toward More Reliable 3D Pose Estimation** | CVPR 2023 | 扩散模型，3D 姿态估计，不确定性建模 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gong_DiffPose_Toward_More_Reliable_3D_Pose_Estimation_CVPR_2023_paper.html) |
| **DiffPose: SpatioTemporal Diffusion Model for Video-Based Human Pose Estimation** | ICCV 2023 | 时空扩散，视频 2D 姿态估计 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion Models** | ICCV 2023 | 多假设 3D 姿态，Embedding Transformer 条件 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **FinePOSE: Fine-Grained Prompt-Driven 3D Human Pose Estimation via Diffusion Models** | CVPR 2024 | 细粒度文本 Prompt，扩散模型 3D 姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_FinePOSE_Fine-Grained_Prompt-Driven_3D_Human_Pose_Estimation_via_Diffusion_Models_CVPR_2024_paper.html) |
| **DiffusionRegPose: Enhancing Multi-Person Pose Estimation using a Diffusion-Based End-to-End Regression** | CVPR 2024 | 扩散 + 端到端回归，多人姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |
| **PostureHMR: Posture Transformation for 3D Human Mesh Recovery** | CVPR 2024 | 扩散式多步姿态变换，体网格恢复 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_PostureHMR_Posture_Transformation_for_3D_Human_Mesh_Recovery_CVPR_2024_paper.html) |
| **PhaseMP: Robust 3D Pose Estimation via Phase-conditioned Human Motion Prior** | ICCV 2023 | Phase 条件运动先验，IMU 融合 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_PhaseMP_Robust_3D_Pose_Estimation_via_Phase-conditioned_Human_Motion_Prior_ICCV_2023_paper.html) |

---

## 3. 基于 GCN / 图神经网络的方法

> 将人体骨架建模为图结构，利用 GCN 显式捕捉关节拓扑依赖。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **GLA-GCN: Global-local Adaptive Graph Convolutional Network for 3D Human Pose Estimation** | ICCV 2023 | 全局局部自适应 GCN，2D-to-3D lifting | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html) |
| **HiPose: Hierarchical Binary Surface Encoding and Correspondence Pruning for RGB-D 6DoF Object Pose Estimation** | CVPR 2024 | 分层 GCN 对应关系剪枝，6DoF 目标姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_HiPose_Hierarchical_Binary_Surface_Encoding_and_Correspondence_Pruning_for_RGB-D_CVPR_2024_paper.html) |
| **CheckerPose: Progressive Dense Keypoint Localization for Object Pose Estimation with GNN** | ICCV 2023 | 渐进式密集关键点 + 图神经网络，目标姿态 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lian_CheckerPose_Progressive_Dense_Keypoint_Localization_for_Object_Pose_Estimation_with_Graph_ICCV_2023_paper.html) |
| **3D Human Pose Estimation via Intuitive Physics** | CVPR 2023 | 物理约束图模型，身体力学先验 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tripathi_3D_Human_Pose_Estimation_via_Intuitive_Physics_CVPR_2023_paper.html) |
| **GATOR: Graph-Aware Transformer with Motion-Disentangled Regression for Human Mesh Recovery** | ECCV 2024 | 图感知 Transformer，运动解耦回归 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3025_ECCV_2024_paper.php) |

---

## 4. 多视角 3D 姿态估计

> 利用多摄像头几何约束进行更精确的 3D 位置估计。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **MVGFormer: Multiple View Geometry Transformers for 3D Human Pose Estimation** | CVPR 2024 | 多视角几何 Transformer，三角测量 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/XunshanMan/MVGFormer) |
| **Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation** | ICCV 2023 | 无标定多视角，概率三角化，蒙特卡洛 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation** | CVPR 2024 | 单视到双视自适应，第一人称手部姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **3D Human Pose Perception from Egocentric Stereo Videos** | CVPR 2024 | 自我中心立体视频，3D 姿态感知 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_3D_Human_Pose_Perception_from_Egocentric_Stereo_Videos_CVPR_2024_paper.html) |
| **Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning** | CVPR 2024 | 多智能体长期姿态预测，交互轨迹条件 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Vendrow_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html) |

---

## 5. 多人姿态估计（Top-down / Bottom-up / One-stage）

> 同时检测场景中多个人体的关节点位置。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation** | CVPR 2024 | One-stage，实时多人，无需检测器 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) · [GitHub](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |
| **DiffusionRegPose: Multi-Person Pose via Diffusion End-to-End Regression** | CVPR 2024 | 扩散端到端多人姿态回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ci_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html) |
| **Rethinking Pose Estimation in Crowds: Overcoming the Detection Information Bottleneck (BUCTD)** | ICCV 2023 | Bottom-up + Top-down 混合，拥挤场景 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.html) |
| **ED-Pose: Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation** | ICCV 2023 | 端到端多人姿态，显式框检测 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_ED-Pose_Explicit_Box_Detection_Unifies_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/ED-Pose) |
| **Mutual Information-Based Temporal Difference Learning for Human Pose Estimation in Video** | CVPR 2023 | 互信息时序差分学习，视频多人姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.html) |
| **Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot** | ECCV 2024 | 单次全身多人网格恢复，ViT backbone | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |

---

## 6. 体网格恢复 (Human Mesh Recovery, HMR)

> 从 2D 图像恢复 SMPL 参数或 3D 体表面网格。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation** | CVPR 2024 | Tokenized 姿态表示，HMR | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a_Tokenized_Pose_Representation_CVPR_2024_paper.html) · [GitHub](https://github.com/saidwivedi/TokenHMR) |
| **ScoreHypo: Probabilistic Human Mesh Estimation with Hypothesis Scoring** | CVPR 2024 | 概率假设评分，HMR 不确定性 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **PostureHMR: Posture Transformation for 3D Human Mesh Recovery** | CVPR 2024 | 扩散式运动学正向过程，SMPL mesh | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_PostureHMR_Posture_Transformation_for_3D_Human_Mesh_Recovery_CVPR_2024_paper.html) |
| **Multi-HMR: Multi-Person Whole-Body HMR in a Single Shot** | ECCV 2024 | 单次多人全身网格，ViT + Cross-Attention | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses** | ECCV 2024 | 多 RoI 相机一致性约束，对比学习 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_25) |
| **VQ-HPS: Human Pose and Shape Estimation in a Vector-Quantized Latent Space** | ECCV 2024 | VQ-VAE 离散潜空间，分类式 HMR | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0692_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2312.08291) |
| **HaMeR: Recovering 3D Hand Mesh via a Transformer-based Approach** | CVPR 2024 | 全 Transformer 手部网格恢复 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html) · [GitHub](https://github.com/geopavlakos/hamer) |
| **POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery** | CVPR 2023 | 高效池化注意力，HMR | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_POTTER_Pooling_Attention_Transformer_for_Efficient_Human_Mesh_Recovery_CVPR_2023_paper.html) |
| **3D Human Mesh Estimation from Virtual Markers** | CVPR 2023 | 虚拟标记点，非参数网格估计 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ma_3D_Human_Mesh_Estimation_From_Virtual_Markers_CVPR_2023_paper.html) |

---

## 7. 视频时序姿态估计

> 利用视频帧序列的时序信息提升 2D/3D 姿态估计精度与平滑性。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **Hourglass Tokenizer (HoT)** | CVPR 2024 | 视频 Pose Transformer，高效 Token 管理 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) |
| **Video-Based Human Pose Regression via Decoupled Space-Time Aggregation** | CVPR 2024 | 空间-时间解耦聚合，视频回归 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html) |
| **DiffPose: SpatioTemporal Diffusion for Video-Based Pose Estimation** | ICCV 2023 | 时空扩散过程，视频 2D 多人 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_DiffPose_SpatioTemporal_Diffusion_Model_for_Video-Based_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Video** | ICCV 2023 | 鲁棒平滑 3D 多人视频姿态 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.html) |
| **Mutual Information-Based Temporal Difference Learning** | CVPR 2023 | 互信息时序差分，视频姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.html) |

---

## 8. 实时 / 轻量化方法

> 面向部署的高效推理，关注 FPS、参数量与精度的平衡。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose** | arXiv 2023 | SimCC，实时，跨平台部署 | [arXiv](https://arxiv.org/abs/2303.07399) · [GitHub](https://github.com/open-mmlab/mmpose) |
| **RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation** | CVPR 2024 | 单阶段，无检测器，高实时性 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html) |
| **Hourglass Tokenizer for Efficient Transformer-Based 3D Pose Estimation** | CVPR 2024 | -50% FLOPs，plug-and-play | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hourglass_Tokenizer_for_Efficient_Transformer-Based_3D_Human_Pose_Estimation_CVPR_2024_paper.html) |
| **HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations** | CVPR 2024 | 端侧实时，稀疏传感器 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_HMD-Poser_On-Device_Real-time_Human_Motion_Tracking_from_Scalable_Sparse_Observations_CVPR_2024_paper.html) |
| **DWPose: Effective Whole-body Pose Estimation with Two-Stage Distillation** | ICCV 2023 | 两阶段知识蒸馏，全身姿态高效推理 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |

---

## 9. 领域自适应与跨域泛化

> 在无源域数据或少量目标域标注下提升姿态估计模型的跨域泛化能力。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **Prior-guided Source-free Domain Adaptation for Human Pose Estimation (POST)** | ICCV 2023 | 无源域自适应，先验自训练回归 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Raychaudhuri_Prior-guided_Source-free_Domain_Adaptation_for_Human_Pose_Estimation_ICCV_2023_paper.html) |
| **Unsupervised Domain Adaptation for Monocular 3D Object Detection via Self-Training** | ICCV 2023 | 自训练，无监督领域自适应 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Unsupervised_Domain_Adaptation_for_Monocular_3D_Object_Detection_via_Self-Training_ICCV_2023_paper.html) |
| **Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation** | CVPR 2024 | 单→双视角自适应，手部姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues** | ECCV 2024 | 不确定性感知，跨视角时序线索 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0241_ECCV_2024_paper.php) · [arXiv](https://arxiv.org/abs/2404.14634) |

---

## 10. 概率姿态估计

> 从单一图像产生多个合理 3D 姿态假设，显式建模深度模糊性。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion Models** | ICCV 2023 | 多假设生成，Embedding Transformer | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html) |
| **ScoreHypo: Probabilistic Human Mesh Estimation with Hypothesis Scoring** | CVPR 2024 | 假设评分，概率 HMR | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ScoreHypo_Probabilistic_Human_Mesh_Estimation_with_Hypothesis_Scoring_CVPR_2024_paper.html) |
| **Normalizing Flows on the Product Space of SO(3) Manifolds for Human Pose Modeling** | CVPR 2024 | SO(3) 流形上的归一化流，概率分布 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jaini_Normalizing_Flows_on_the_Product_Space_of_SO3_Manifolds_for_Probabilistic_Human_Pose_Modeling_CVPR_2024_paper.html) |
| **Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose** | ICCV 2023 | 概率三角化，无标定多视角 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) |

---

## 11. 全身姿态估计 (Whole-body)

> 同时估计身体、手部、面部关键点的整体姿态。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **DWPose: Effective Whole-body Pose Estimation with Two-Stage Distillation** | ICCV 2023 | 两阶段蒸馏，全身 17+133 关键点 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Effective_Whole-body_Pose_Estimation_with_Two-stage_Distillation_ICCV_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/DWPose) |
| **Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot** | ECCV 2024 | 单次全身多人 SMPL-X，ViT + HPH | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/0336_ECCV_2024_paper.php) · [GitHub](https://github.com/naver/multi-hmr) |
| **One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer** | CVPR 2023 | 单阶段全身网格，组件感知 Transformer | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_with_Component_Aware_Transformer_CVPR_2023_paper.html) |
| **ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation** | ICCV 2023 (Journal) | 通用体态估计，ViT 基础模型，异构关键点 | [arXiv](https://arxiv.org/abs/2212.04246) · [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |

---

## 12. 手部姿态估计

> 专注于手部关键点/网格的高精度估计。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **HaMeR: Reconstructing Hands in 3D with Transformers** | CVPR 2024 | 全 Transformer，3D 手部网格，in-the-wild | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html) · [GitHub](https://github.com/geopavlakos/hamer) |
| **Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation** | CVPR 2024 | 单→双视图，第一视角手部姿态 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html) |
| **Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting** | CVPR 2024 | 注意力传播，自我中心热图→3D | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kang_Attention-Propagation_Network_for_Egocentric_Heatmap_to_3D_Pose_Lifting_CVPR_2024_paper.html) |

---

## 13. 数据集 / Benchmark 类论文

> 提出新数据集或评估基准，推动 pose 估计研究发展。

| 论文 | 会议 | 关键词 | 链接 |
|------|------|--------|------|
| **Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes** | CVPR 2023 | 艺术风格人体图像，多场景泛化基准 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.html) · [GitHub](https://github.com/IDEA-Research/HumanArt) |
| **EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams** | CVPR 2024 | 事件相机，自我中心运动捕获数据集 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Millerdurai_EventEgo3D_3D_Human_Motion_Capture_from_Egocentric_Event_Streams_CVPR_2024_paper.html) |
| **Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi** | CVPR 2024 | WiFi 信号多人 3D 姿态，新模态 benchmark | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html) |

---

## 参考资源

- [CVF Open Access (CVPR/ICCV)](https://openaccess.thecvf.com/)
- [ECCV 2024 论文集 (ECVA)](https://www.ecva.net/eccv2024.php)
- [52CV CVPR-2024-Papers](https://github.com/52CV/CVPR-2024-Papers)
- [52CV ICCV-2023-Papers](https://github.com/52CV/ICCV-2023-Papers)
- [52CV ECCV-2024-Papers](https://github.com/52CV/ECCV-2024-Papers)
- [DmitryRyumin CVPR-2023-24-Papers](https://github.com/DmitryRyumin/CVPR-2023-24-Papers)
- [MMPose](https://github.com/open-mmlab/mmpose)

---

*Last updated: 2025-04 | 覆盖 CVPR 2023 · ICCV 2023 · CVPR 2024 · ECCV 2024*
