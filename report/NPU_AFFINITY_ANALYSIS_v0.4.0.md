# YOLOv6 v0.4.0 NPU 亲和性分析

生成日期：2026-06-16

## 分析口径

本报告按硬件亲和性流程重刷，目标不是简单回答“能否在 NPU 上跑通”，而是拆分 YOLOv6 v0.4.0 在 Ascend NPU 上的主导路径、理论瓶颈、layout/搬运风险、同步与 host/head 开销。结论绑定如下口径：

| 项目 | 口径 |
| --- | --- |
| target_platform | Ascend 950PR |
| NPU 架构版本 | 3510 |
| 运行栈 | CANN 9.0.0，torch-npu 2.10.0 |
| 分析模型 | YOLOv6-N，P5 输入尺寸 640，batch size 32 speed/val |
| dtype | FP32 val/speed；FP16 speed |
| 阶段划分 | YOLOv6 是单帧检测模型，不适用 LLM prefill/decode；本报告拆为 preprocess、network inference、decode/NMS、COCO metric |
| 实测证据 | COCO val2017 AP@[0.50:0.95] = 0.374；FP32 inference = 0.30 ms/image；FP16 inference = 0.20 ms/image |

950 系列对应 3510 架构，AIC 与 AIV 为 1:2，具备分离的 Cube/Vector 路径、NDDMA、CV Fusion、SIMD/SIMT 混合编程和 950 特有的 L2 cacheline 结构。以下判断只对 Ascend 950PR 当前运行栈成立，不外推到 A2/A3 或 950DT。

## 总体结论

YOLOv6-N 的主体网络与 Ascend 950PR 的 Cube 路径亲和性较好。模型的主要 FLOPs 来自 fused Conv/BN、RepVGG/RepBlock、RepPAN/FPN-PAN 和 head convolution，这些负载天然映射到 Cube。完整 COCO val 跑通且 mAP 与 README 基准基本一致，说明模型主体计算、head 输出和 bbox decode 没有出现明显正确性偏移。

端到端亲和性的短板不在卷积主体，而在 PyTorch eager 后处理链路：`torchvision::nms` 在当前依赖组合中不可用，实际使用 CPU fallback NMS。常规 speed 阈值下 NMS 约 0.89-0.94 ms/image，已经高于 NPU 模型推理耗时；COCO val 的低置信度阈值下 NMS 放大到 149.31 ms/image，成为压倒性瓶颈。因此，本轮结论是“network inference 对 NPU 亲和，end-to-end detection pipeline 仍被 host/head 与 CPU NMS 限制”。

## 五路径拆解

| 子段 | 主导路径 | 压力判断 | 证据 | 亲和结论 |
| --- | --- | --- | --- | --- |
| Conv/RepBlock/SPPF/RepPAN 主体 | Cube + MTE/FixPipe | 主要 FLOPs 在卷积，适合 Cube；实际效率受层间 GM 物化、tile 利用率和 eager kernel 边界影响 | README 给出 YOLOv6-N 11.4G FLOPs、4.7M params；FP16 inference 实测 0.20 ms/image | 好，但需要 profile 确认 AIC 利用率、GM bytes 与 tile 驻留 |
| BN 融合与 activation | Vector + CV Fusion 潜力 | BN 在加载时 fuse 进 Conv；SiLU/ReLU 属于 Vector epilogue，可与卷积后处理形成融合机会 | `load_checkpoint(..., fuse=True)` 触发 model fuse；网络 forward 和 mAP 通过 | 中到好；PyTorch eager 下不保证等价于手写 CV fusion |
| Head reshape/permute/cat/bbox decode | Vector + MTE/FixPipe + head | 输出约 8400 anchors × 85 fields；reshape/permute 部分可视图化，cat 与 decode 会产生小 kernel/搬运 | forward 输出 `(1, 8400, 85)`；COCO AP 对齐 | 中；应优先折叠 layout 与 decode |
| NMS 后处理 | host/head + Vector 待替换 | 当前走 CPU fallback；低阈值候选框多时极慢 | speed 阈值 0.4 下 NMS 0.89-0.94 ms/image；val 阈值 0.03 下 NMS 149.31 ms/image | 差；当前最大端到端瓶颈 |
| COCO metric | host/head | `pycocotools` 在 CPU 上统计 AP/AR | 完成 COCO AP/AR 输出 | 功能可用，但不是 NPU 亲和路径 |
| 单卡评估 | communication | 无 collective，通信压力为零 | 本轮只使用单卡可见 `npu:0` 执行 eval/speed | 好；多卡训练/分布式未验证 |

## 理论量化初判

950PR FP16 Cube-only 峰值为 378/432 TFLOPS，封装内存带宽为 1.4/1.6 TB/s，对应 FP16 Cube 平衡点约 270 FLOP/Byte。YOLOv6-N 官方表给出 11.4G FLOPs，因此理想 Cube 计算下界约为 0.026-0.030 ms/image。本轮 FP16 speed 实测模型推理为 0.20 ms/image，约为该纯峰值下界的 6-8 倍。

这个差距不说明模型不亲和，而说明 PyTorch eager 路径下的真实瓶颈不可能只由 Cube 峰值解释。按 roofline 粗判，若有效 GM 读写超过约 42 MB/image，YOLOv6-N 就会从理想 Cube-bound 转向 MTE/GM、layout 或 head-bound。模型最小权重与输入输出字节低于这个阈值，但中间 feature map 逐层物化、concat、head cat、host/device 边界和小 kernel launch 都会增加有效 GM bytes 与 head 开销。准确分类需要 STARS/torch-npu profile 采样 AIC、AIV、MTE 与 kernel timeline，本报告将其标为 `待测`。

FP32 speed 实测 inference 为 0.30 ms/image，FP16 speed 为 0.20 ms/image。FP16 提升符合 Cube 路径更亲和的方向，但端到端 speed 仍被 NMS 拉高：在 `conf_thres=0.4` 下，NMS 约为 FP16 模型推理的 4.5 倍；在 COCO val 默认低阈值下，NMS 约为 FP32 模型推理的 466 倍。因此，后处理替换比继续只优化卷积主体更优先。

## NPU 特有亲和项

| 检查项 | YOLOv6-N 现状 | 风险/机会 |
| --- | --- | --- |
| tile 驻留与 double buffer | 卷积层本身适合分块；950PR 每 AI Core 有 L1/L0/UB 层级，Conv 主体可由后端自动 tiling | PyTorch eager 不暴露 tile 驻留；P6/大模型可能因 feature map 和 channel 增大改变 tile 利用率，需 profile |
| 32B/512B 与小包 | 950 L2 为 512B cacheline + 4×128B sector；输入/feature 主体是连续 CHW tensor | head 输出 80 类、bbox 字段和候选框筛选会形成小包与非连续访问；不能套用 A2 的 GM 512B 对齐结论 |
| repeat/mask 密度 | 卷积主体由 Cube 承担；activation、sigmoid、bbox decode 是规则向量操作 | NMS 的排序、筛选、IoU 抑制不是规则高 repeat SIMD 路径，当前直接落到 CPU |
| layout 折叠 | flatten/permute 中部分可作为 view；cat、decode、NMS prep 会物化 | 可把 head 输出长期维持为 `(N, HW, C)` 物理布局，减少在线 permute/cat |
| MTE/FixPipe/NDDMA | 950 的 NDDMA 可用于多维 reorder 与搬运折叠；FixPipe 可承接部分随路转换 | 当前 PyTorch 评估未显式利用 NDDMA/CV fusion，layout conversion 收益仍是部署路径待测项 |
| reduce/state layout | 推理主体没有大规模 reduce/state update；NMS 存在排序、top-k、IoU 抑制 | NMS 不应按普通 reduce 处理，需要 NPU NMS 或部署后处理算子专项验证 |
| 同步边界 | 单卡推理无通信同步；层间是 eager kernel 边界 | 融合收益不能只按“少 launch”估算；跨硬同步边界收益需要 profile 证明 |
| host/head 开销 | speed 文档原本不把 preprocessing/NMS 计入公平推理速度；本轮日志可分别看到 preprocess、inference、NMS | 生产链路不能忽略 host/head；COCO val 低阈值下 host/head 已经支配总耗时 |

## 不亲和段的等价重写方向

当前最不亲和的是 NMS 与 head 后处理。按“先找等价/可控近似流程，再刷新亲和结论”的原则，建议如下：

| 原流程 | 问题 | 等价/近似重写方向 | 精度约束 |
| --- | --- | --- | --- |
| `sigmoid -> reshape/permute -> cat -> dist2bbox -> CPU NMS` | 小 kernel 多、layout 物化、NMS 落 CPU | 将 head 物理输出改为 `(N, HW, C)`，把 sigmoid、bbox decode、score threshold 折叠进 head 后处理 | COCO AP@[0.50:0.95] 与原 PyTorch 输出对齐 |
| class-aware batched NMS | 当前 `torchvision::nms` 不可用，CPU fallback 支配耗时 | 使用 NPU/Ascend 后处理算子，或在 OM/AscendC 路径实现 batched NMS | max_det、iou_thres、class offset 语义与原实现一致 |
| 低阈值全候选 NMS | `conf_thres=0.03` 产生大量候选框 | 在不改变 mAP 的前提下增加 per-class top-k 或 score prefilter，再执行 NMS | 需用 COCO val2017 比较 mAP，不能只看速度 |
| 在线 layout 转换 | permute/cat 在 head 末端集中出现 | 使用 NDDMA、地址生成或持久化 physical layout 折叠转换 | 输出 tensor 语义与导出/评估接口一致 |

这些重写方向尚未上板验证，报告中只能作为适配建议，不能写成已完成优化。

## 阻塞项与待测清单

| 项目 | 当前状态 | 下一步验证 |
| --- | --- | --- |
| NPU 原生 NMS | 未完成；当前 CPU fallback | 建立 8400 anchors、80 classes、阈值 0.03/0.4 的 NMS microbench，比较语义、mAP 和耗时 |
| AIC/AIV/MTE profile | 未采样；只有端到端日志耗时 | 用 STARS/torch-npu profiler 拆 AIC、AIV、MTE、host/head 时间，验证是否 GM-bound |
| tile 与 double buffer | 后端自动 tiling，未显式观测 | 对 YOLOv6-N/S/M/L 和 P6 1280 输入分别采样，观察 tile 利用率与 HBM 压力 |
| FP16 mAP | 只跑了 FP16 speed，未跑完整 FP16 COCO mAP | 如需声明 FP16 精度，补跑 COCO val2017 |
| 全模型矩阵 | 完整复现只覆盖 YOLOv6-N | 补 YOLOv6-S/M/L、N6/S6/M6/L6 的 forward、speed、mAP |
| Ascend 原生部署 | 未执行 ONNX -> ATC/OM | 验证静态图、CV fusion、NDDMA layout folding 和 NPU 后处理是否消除 PyTorch eager 瓶颈 |
| 训练亲和性 | 未验证 | 单独分析 loss、assigner、optimizer、AMP、分布式和数据增强路径 |

## 自检结论

本版报告已绑定 Ascend 950PR、3510 架构、FP32/FP16 dtype 与 YOLOv6-N 640 推理口径；已按 Cube、Vector、MTE/FixPipe、communication、host/head 拆分压力；平衡点只用于 FP16 Cube 主体，不用于 NMS、metric 或 layout；950 的 512B L2 cacheline 未与 A2 GM 512B 对齐规则混写；所有未 profile 的 API 效率、tile 驻留、NDDMA/CV fusion 收益和 NMS 替换效果均标为 `待测`。

最终判断：YOLOv6 v0.4.0 的 NPU 亲和性应分层描述。模型主体网络对 Ascend 950PR 推理亲和，mAP 和 inference speed 已经证明可用；当前端到端检测 pipeline 的亲和性为中等偏低，主要被 CPU NMS、host/head 和 PyTorch eager layout/launch 边界限制。下一阶段优先级应是 NPU 原生 NMS 与静态图/OM 部署，而不是继续只优化卷积主体。
