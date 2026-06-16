# YOLOv6 v0.4.0 NPU 部署报告

生成日期：2026-06-16

## 任务概述

本次部署任务是在 YOLOv6 v0.4.0 代码基础上，将项目部署到 Ascend NPU 容器化环境中，并完成以 COCO val2017 精度复现和推理速度测试为核心的验证。YOLOv6 是面向目标检测任务的 YOLO 系列实现，v0.4.0 的公开基准主要围绕 COCO 数据集上的 mAP、端到端推理速度和不同模型规模展开。

版本锚点为 tag `0.4.0`，commit 为 `f2114f37ff12ca888cddb47baf6b532e04110251`。本次只验证 v0.4.0 代码与公开权重在 Ascend NPU PyTorch 路径下的可部署性、可复现性和主要性能瓶颈；未把结果与官方 T4 TensorRT FPS 直接等价。

硬件环境为 Ascend950PR，CANN 版本为 `9.0.0`，`npu-smi`/driver 版本为 `25.7.rc1`，单卡 HBM 约 114688 MB。按任务约束，本次只使用后四张物理 NPU；实测中完整精度复现使用一张空闲后段卡，FP32/FP16 speed 测试分别使用另外两张空闲后段卡。设备通过环境变量隔离后，进程内可见设备为 `npu:0`。

## 部署过程

代码以本地 tag `0.4.0` 为准同步到目标容器的挂载工作区，远端通过 git commit 确认版本一致。运行环境、数据集、权重和日志均放在挂载盘，未将大文件写入系统盘。

Python 运行环境采用独立 Python 3.12 虚拟环境，关键依赖如下：

| 组件 | 版本/状态 |
| --- | --- |
| Python | 3.12.13 |
| CANN | 9.0.0 |
| npu-smi / driver | 25.7.rc1 |
| torch | 2.10.0+cpu |
| torch-npu | 2.10.0 |
| torchvision | 0.25.0，`torchvision::nms` 在当前组合中不可用 |
| pycocotools | 已安装，用于 COCO mAP 计算 |
| opencv-python-headless | 已安装，用于图像读取和预处理 |
| thop | 已安装，用于模型信息统计 |

数据侧使用 COCO val2017 图片和 `instances_val2017.json`。YOLOv6 v0.4.0 的 dataloader 需要 YOLO 格式 label 文件，因此本次从 COCO JSON 生成了 val2017 的检测标签，共 5000 张图片和 5000 个 label 文件，之后再执行官方 `tools/eval.py` 验证。

公开权重已完成下载并放置在挂载盘，包括 `yolov6n/s/m/l` 和 `yolov6n6/s6/m6/l6`。本轮完整复现选择 YOLOv6-N，原因是它是 v0.4.0 README 中最小且最适合先验证 NPU 路径的官方模型；其余模型权重已就绪，但未纳入本轮完整 COCO 复现矩阵。

## 兼容适配

YOLOv6 v0.4.0 源码默认假设 CUDA 路径，并且官方公开权重中存在旧版 pickled class/属性结构。当前 Ascend PyTorch 运行栈还存在 `torchvision.ops.nms` 不可用的问题。为完成 NPU 复现，本次做了小范围兼容适配，重点是让官方权重能够加载、模型能够迁移到 NPU、推理和评估链路能够跑完。

| 文件 | 适配内容 | 影响 |
| --- | --- | --- |
| `yolov6/core/evaler.py` | 增加 `torch_npu` 设备识别；CUDA 不可用时选择 NPU；checkpoint 先加载到 CPU 再迁移到目标设备 | 解除评估入口的 CUDA 绑定 |
| `yolov6/utils/checkpoint.py` | 在新版 PyTorch 下显式使用兼容 checkpoint 加载参数 | 解决 PyTorch 2.6+ `weights_only` 默认行为变化 |
| `yolov6/utils/torch_utils.py` | 增加 NPU 同步计时；模型信息统计对不支持路径做降级 | 让 speed 统计可在 NPU 上输出 |
| `yolov6/utils/nms.py` | `torchvision.ops.nms` 不可用时使用 CPU fallback NMS，并把索引迁回原设备 | 保证 COCO 评估链路完整跑通 |
| `yolov6/layers/common.py` | 补充旧权重反序列化所需类名和属性兼容；增强 SPPF/CSPSPPF、RepVGGBlock 对旧结构的兼容 | 解决公开权重加载与 forward 兼容问题 |
| `yolov6/models/heads/effidehead_distill_ns.py` | 兼容旧版 detect head 的回归预测属性命名 | 解决部分公开权重 head 结构差异 |

这些适配不改变 YOLOv6-N 的检测数学定义，主要处理设备选择、权重加载、旧结构兼容和后处理 fallback。需要注意的是，CPU fallback NMS 会明显影响低置信度 COCO val 的总耗时，因此速度结论必须拆分模型推理和后处理。

## 复现方法

精度复现使用 YOLOv6 官方评估入口和官方 COCO 配置：

```bash
python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task val \
  --reproduce_640_eval
```

速度测试使用官方 speed 入口，固定 batch size 为 32，并分别验证 FP32 与 FP16：

```bash
python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task speed \
  --conf-thres 0.4

python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task speed \
  --conf-thres 0.4 \
  --half
```

官方 README 中的 T4 TensorRT FPS 属于 ONNX/TensorRT 部署路径，本次 speed 结果是 Ascend NPU 上的 PyTorch/torch-npu 路径，二者不直接横向比较。

## 功能验证结果

| 验证项 | 结果 |
| --- | --- |
| 版本锚点 | tag `0.4.0`，commit `f2114f37ff12ca888cddb47baf6b532e04110251` |
| NPU 可用性 | `torch_npu` 可导入，`torch.npu.is_available()` 可用 |
| 设备隔离 | 只使用后四张物理 NPU 中的空闲卡；单进程映射为 `npu:0` |
| 数据完整性 | COCO val2017 图片 5000 张；YOLO label 5000 个；COCO annotation 可被 `pycocotools` 读取 |
| 权重加载 | 官方 YOLOv6-N 权重可加载到 NPU 并完成 forward |
| forward smoke | 输入 `1x3x640x640`，输出形状为 `(1, 8400, 85)` |
| 完整 COCO val | YOLOv6-N 完成 5000 张 val2017 评估，并输出 COCO AP/AR |
| speed FP32 | batch 32 可完成官方 speed 入口 |
| speed FP16 | batch 32 + `--half` 可完成官方 speed 入口 |

YOLOv6-N 在 COCO val2017 上的完整复现结果如下：

| 指标 | 本次 NPU 复现 | README 参考 |
| --- | ---: | ---: |
| AP@[IoU=0.50:0.95] | 0.374 | 0.375 |
| AP@0.50 | 0.529 | 未单列 |
| AP@0.75 | 0.404 | 未单列 |
| AP small | 0.176 | 未单列 |
| AP medium | 0.414 | 未单列 |
| AP large | 0.550 | 未单列 |
| AR@1 | 0.318 | 未单列 |
| AR@10 | 0.528 | 未单列 |
| AR@100 | 0.579 | 未单列 |

精度结果与 README 中 YOLOv6-N 37.5 mAP 基准基本一致，差异为 0.1 个百分点量级，属于可复现范围。

速度测试结果如下，单位为单图平均耗时：

| 测试模式 | pre-process | inference | NMS | 说明 |
| --- | ---: | ---: | ---: | --- |
| val / FP32 / `conf_thres=0.03` | 0.04 ms | 0.32 ms | 149.31 ms | COCO 精度评估低阈值产生大量候选框，CPU fallback NMS 成为主瓶颈 |
| speed / FP32 / `conf_thres=0.4` | 0.03 ms | 0.30 ms | 0.94 ms | 更接近常规推理阈值下的 PyTorch NPU speed |
| speed / FP16 / `conf_thres=0.4` | 0.03 ms | 0.20 ms | 0.89 ms | 半精度模型推理耗时下降，但 NMS 仍在 CPU fallback |

## 部署结论

YOLOv6 v0.4.0 已在 Ascend950PR / CANN 9.0.0 / torch-npu 2.10.0 环境中完成 NPU 部署和 YOLOv6-N 官方精度复现。官方权重可加载，模型主干推理可在 NPU 上执行，COCO val2017 mAP 与 README 基准基本一致，FP16 speed 路径也可运行。

当前部署不应声明为完整 NPU 原生推理栈。主要限制是 `torchvision.ops.nms` 在当前依赖组合中不可用，评估链路使用 CPU fallback NMS；在 COCO val 的低置信度阈值下，该 fallback 会把 NMS 放大为主要耗时来源。若后续目标是稳定吞吐或端到端部署，应优先替换为 NPU 兼容 NMS，或转向 Ascend 原生推理部署路径。

本轮只完成 YOLOv6-N 的完整 COCO 精度复现和速度验证。其他 P5/P6 模型权重已就绪，但仍需按模型规模、输入尺寸和精度模式补全验证矩阵，尤其是 P6 系列需要使用更大输入尺寸进行独立评估。
