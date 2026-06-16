# YOLOv6 v0.4.0 — 目标检测 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | YOLOv6-0.4.0 |
| 任务用途 | COCO 目标检测推理、评估、训练单卡功能验证与 NPU 亲和性复现 |
| 仓库 | https://github.com/meituan/YOLOv6 |
| 版本 / commit | 0.4.0 / f2114f37 |
| 报告人 | - |
| 日期 | 2026-06-16 |
| 硬件 | Ascend 950PR ×8 / CANN 9.0.0 |
| 软件 | torch 2.10.0+cpu / torch_npu 2.10.0 / torchvision 0.25.0+cpu / Python 3.12.13 |

---

## 1. 技术栈梳理
- 主语言:Python。
- ML 框架:PyTorch。评估入口为 `tools/eval.py`,COCO 指标走 `pycocotools`。
- CUDA 依赖:原始 v0.4.0 评估与训练代码均默认 CUDA。评估侧默认 CUDA 设备与 `torchvision.ops.nms`;训练侧 `select_device`、AMP/GradScaler、loss、empty_cache、DDP、distill/fuse_ab 分支均有 CUDA 绑定。标准训练、resume、distill、fuse_ab、DDP 入口均已执行,阻塞项只记录实测失败路径。
- 自定义核(.cu / C++ 扩展):无必须编译的自定义 CUDA 扩展。当前后处理瓶颈来自 `torchvision::nms` 无 NPU dispatch kernel,不是项目自带 .cu 编译失败。
- 第三方库:opencv-python-headless、pycocotools、thop、onnx/onnxsim、tensorboard 等。`torchvision 0.25.0+cpu` 可加载 CPU custom ops,`torchvision::nms` 具备 CPU/Meta kernel,无 NPU kernel。
- 模型权重 / 来源:YOLOv6 v0.4.0 release 官方权重。完整 COCO val 复现使用 `yolov6n.pt`;P5/P6 全部官方权重已完成 speed 可运行矩阵。

## 2. 部署步骤
- [x] 依赖安装:使用独立 Python 3.12 虚拟环境,安装 torch 2.10.0+cpu、torch_npu 2.10.0、pycocotools、opencv-python-headless、thop 等依赖。
- [x] 编译 / 构建:无需编译项目本体;以源码方式运行评估入口。
- [x] 权重获取:下载 v0.4.0 官方 P5/P6 权重,完整复现选择 YOLOv6-N。
- [x] 数据准备:COCO val2017 图片 5000 张,由 `instances_val2017.json` 生成 YOLO 格式 val label 5000 个。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):评估入口支持 NPU 设备;checkpoint 先 CPU 加载再迁移;PyTorch 2.6+ checkpoint 加载显式兼容;旧权重序列化 `Conv`、`SimConv`、`Conv_C3` 等 class/属性增加 shim;NMS 因缺 NPU kernel 走 CPU fallback。
- [x] 训练单卡功能验证适配:标准 YOLOv6-N、单 NPU、FP32、2 epoch 功能验证链路通过。已覆盖 data loader、forward、loss/assigner、backward、optimizer step、checkpoint save、final eval、未 strip 中间 checkpoint resume。
- [x] profiler 采集:仓内无内置 profiler,按注入式 `torch_npu.profiler` 采集 YOLOv6-N FP16 forward 与 FP32 train step,使用 Level1 + PipeUtilization,并验收 profiler CSV/JSON 产物。
- 命令:
```bash
# COCO val 精度复现
python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task val \
  --reproduce_640_eval

# FP32 speed
python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task speed \
  --conf-thres 0.4

# FP16 speed
python tools/eval.py \
  --data data/coco.yaml \
  --batch-size 32 \
  --weights weights/yolov6n.pt \
  --task speed \
  --conf-thres 0.4 \
  --half

# 训练单卡功能验证,使用少量 COCO val 样本构造 train/val 子集
python tools/train.py \
  --data-path data/npu_train_subset.yaml \
  --conf-file configs/yolov6n.py \
  --epochs 2 \
  --batch-size 2 \
  --img-size 320 \
  --workers 0 \
  --eval-final-only \
  --output-dir runs/train_npu_supplement \
  --name standard_2epoch

# P5/P6 speed 矩阵:按模型大小调整 batch
python tools/eval.py \
  --data data/coco.yaml \
  --weights weights/yolov6s6.pt \
  --img-size 1280 \
  --batch-size 8 \
  --task speed \
  --conf-thres 0.4 \
  --half
```

## 3. 验证用例
- 输入数据:COCO val2017,共 5000 张图片;P5 输入尺寸 640,batch size 32。训练功能验证另用 8 张 train + 4 张 val 的小子集,评价范围为训练程序路径可执行性,不代表训练精度或收敛性。
- 运行命令:见第 2 节,覆盖 COCO val FP32/FP16、P5/P6 全权重 speed、标准训练 2 epoch、resume、fuse_ab、distill、DDP 启动与 profiler。
- 期望输出:README 中 YOLOv6-N mAP<sup>val 0.5:0.95</sup> 为 37.5;官方 speed 文档说明公平测速不包含 preprocess 与 NMS。
- 实测输出:
  - forward 功能验证:输入 `1x3x640x640`,输出 `(1, 8400, 85)`。
  - YOLOv6-N COCO val FP32:AP@[0.50:0.95] = 0.374,AP@0.50 = 0.529,AP@0.75 = 0.404;preprocess 0.04 ms/image,inference 0.32 ms/image,NMS 149.31 ms/image。
  - YOLOv6-N COCO val FP16:AP@[0.50:0.95] = 0.374,AP@0.50 = 0.529,AP@0.75 = 0.404;preprocess 0.03 ms/image,inference 0.19 ms/image,NMS 157.18 ms/image。
  - 原始训练入口:使用 NPU 环境直接运行会在 `select_device()` 处因 `torch.cuda.is_available()` 为 False 触发 `AssertionError`。
  - 标准训练功能验证:YOLOv6-N 单 NPU FP32,batch size 2,image size 320,workers 0,8 张 train + 4 张 val,完成 2 epoch,每 epoch 4 step,生成 `last_ckpt.pt` 与 `best_ckpt.pt`,未出现 label assignment CPU fallback。
  - resume 功能验证:对未 strip 的中间 checkpoint 加入 `weights_only=False` 兼容后可恢复并继续完成下一 epoch。训练结束后被 `strip_optimizer` 清理的导出 checkpoint 不包含 optimizer,不作为继续训练 checkpoint 使用。
  - 训练 assigner:首次 NPU 训练补丁会在 label assignment 中因 `DT_DOUBLE` 输入触发 NPU `Min` error code `161002`,随后回退 CPU;将 `loss.preprocess()` 生成的 numpy array 固定为 `float32` 后,同一训练功能验证不再出现 `CPU mode for This Batch`。
  - `fuse_ab` 单卡训练:实测失败。分支 loss 仍会生成 `DT_DOUBLE` 进入 NPU `Min`,随后 CPU fallback 路径调用 `.cuda()` 并触发 `AssertionError: Torch not compiled with CUDA enabled`。
  - `distill` 单卡训练:实测失败。`loss_distill_ns.py` 与 `fuse_ab` 同类,先触发 `DT_DOUBLE` NPU `Min` 参数错误,再在 fallback 中调用 `.cuda()` 失败。
  - DDP 启动:实测失败。`LOCAL_RANK` 分支直接调用 `torch.cuda.set_device(args.local_rank)`,当前 CPU+torch_npu torch build 下报 `_cuda_setDevice` 不存在。
  - QAT/quant 链路:上游实现面向 NVIDIA TensorRT INT8 部署,依赖 `pytorch_quantization`、QAT/校准权重与 TensorRT scale cache;源码入口仍绑定 `.cuda()`。该链路不纳入 Ascend PyTorch FP32/FP16 运行评价。
- 与 CPU/GPU 基准对比(误差/一致性):mAP 与 README 37.5 基准相差约 0.1 个百分点,可视为复现通过。README 的 T4 TensorRT FPS 属于 TensorRT 部署路径,当前结论仅覆盖 torch_npu PyTorch 路径,不做横向等价比较。

**Speed 矩阵**(`task=speed,conf_thres=0.4`;时间单位 ms/image):

| 模型 | 输入 | batch | FP32 inference | FP16 inference | NMS 范围 |
|---|---:|---:|---:|---:|---:|
| YOLOv6-N | 640 | 32 | 0.28 | 0.22 | 0.91-0.92 |
| YOLOv6-S | 640 | 32 | 0.54 | 0.34 | 0.86-0.97 |
| YOLOv6-M | 640 | 16 | 1.10 | 0.69 | 0.90-0.97 |
| YOLOv6-L | 640 | 16 | 1.91 | 1.16 | 0.94-1.21 |
| YOLOv6-N6 | 1280 | 8 | 1.22 | 1.06 | 1.19-1.29 |
| YOLOv6-S6 | 1280 | 8 | 2.19 | 1.48 | 1.20-1.48 |
| YOLOv6-M6 | 1280 | 4 | 4.60 | 3.18 | 1.33-1.37 |
| YOLOv6-L6 | 1280 | 4 | 7.71 | 5.39 | 1.10-1.70 |

## 4. NPU 亲和性
| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 推理/评估跑通;P5/P6 全权重 speed 跑通。标准 YOLOv6-N 单卡 FP32 2 epoch 训练、未 strip checkpoint resume、训练 step profiler 均跑通。`fuse_ab`、`distill`、DDP 入口实测失败,见阻塞项。 |
| NPU 利用率 (npu-smi) | npu-smi 用于确认后段卡可见与作业结束无残留进程。细粒度 AIC/AIV/MTE 使用注入式 `torch_npu.profiler` 产物判断。 |
| HBM 占用 | 单卡 HBM 114688MB。YOLOv6-N FP16 forward profiler:allocated peak 500.63 MiB,reserved peak 718.0 MiB;YOLOv6-N FP32 train-step profiler:allocated peak 188.79 MiB,reserved peak 276.0 MiB。 |
| 关键算子是否回退 CPU | 推理主体无关键 CPU 回退;NMS 回退 CPU;pycocotools 指标统计在 CPU。标准训练在 float32 assigner 修正后未观察到 label assignment CPU fallback。`fuse_ab`/`distill` 分支仍会触发 assigner fallback 并因 `.cuda()` 失败。 |
| 性能(吞吐/时延) | YOLOv6-N FP16 COCO val inference 0.19 ms/image,NMS 157.18 ms/image;speed 阈值下 YOLOv6-N FP16 inference 0.22 ms/image,NMS 0.92 ms/image。P5/P6 详见 speed 矩阵。 |

口径:Ascend 950PR、3510 架构、YOLOv6-N、P5 640、batch 32。YOLOv6 是单帧检测模型,不适用 LLM prefill/decode;这里拆为 preprocess、network inference、decode/NMS、COCO metric。FP16 Cube 平衡点约 270 FLOP/Byte,高于为算力主导,低于为访存/搬运主导。

**计算分布**(基于模型结构、speed 日志与 Level1 + PipeUtilization profiler):

| 单元 | 主要算子/路径 | 占比/压力 | 说明 |
|---|---|---:|---|
| 算力(Cube,卷积矩阵) | Conv2D、RepBlock、SPPF、RepPAN、head conv | 主体 | forward profiler 中 Conv2DV2 AI_CORE 占 53.24%,Conv2DV2 MIX_AIC 占 8.69%;FP16 speed 随模型规模线性上升,主体亲和 NPU。 |
| 向量(Vector,激活解码) | SiLU/ReLU、Sigmoid、bbox decode、训练 elementwise/optimizer | 中 | forward profiler 中 ReLU 7.82%、Concat 6.29%、Transpose 3.11%;train-step profiler 中 Mul 19.56%、Add 13.52%,说明训练 step 的向量/elementwise 比重显著。 |
| 搬运(MTE/FixPipe) | feature map 读写、concat、reshape/permute/cat | 中 | profiler 已能看到 Concat/Transpose 等布局相关开销;PyTorch eager 层间物化仍是优化空间。 |
| 通信(communication) | 无 | 0 | 单卡评估,无 collective。 |
| 调度(host/head) | preprocess、CPU NMS、pycocotools、训练保存/eval | 高 | COCO val 低阈值下 NMS 约 150ms/image,远大于 NPU inference;端到端检测瓶颈在 host 后处理。 |

**判定**:YOLOv6-N 的网络主体亲和 NPU。按 950PR FP16 Cube 峰值 378/432 TFLOPS 粗估,11.4G FLOPs 的理想 Cube 下界约 0.026-0.030ms/image;实测 FP16 speed inference 0.22ms/image、完整 val inference 0.19ms/image,差距来自真实 tile 利用率、GM 搬运、eager kernel 边界与 head 开销。端到端瓶颈明确在 CPU NMS,不是卷积主体。

- 算子回退清单:
  - `torchvision.ops.nms`:CPU/Meta kernel 可用,NPU kernel 不存在;NPU 张量调用触发 torch_npu CPU fallback。功能不阻塞,mAP 可复现;端到端性能被严重限制。
  - `pycocotools`:CPU 统计 AP/AR,不属于 NPU kernel 路径。
  - OpenCV/letterbox/preprocess:host 侧处理,当前耗时很小,不是主要瓶颈。
  - 训练 label assignment:标准 loss 中 `loss.preprocess()` 固定为 `float32` 后,单卡训练功能验证未再回退。`fuse_ab`、`distill` loss 未应用同类 dtype/device 修正,实测仍触发 `DT_DOUBLE`。
- profiler 摘要:
  - 仓内无 `enable_profiler`、`tensorboard_trace_handler` 或 `torch_npu.profiler` 内置采集入口,采用注入采集。配置为 `ProfilerLevel.Level1 + AiCMetrics.PipeUtilization`,循环结束额外 `prof.step()` 收尾。
  - FP16 forward profiler:输出 `(32, 8400, 85)`,产物 `kernel_details.csv` 49 列、2260 行,`op_statistic.csv`、`api_statistic.csv`、`operator_details.csv`、`step_trace_time.csv` 均存在,`trace_view.json` 合法。
  - FP32 train-step profiler:采 2 个 train step,产物 `kernel_details.csv` 49 列、7164 行,CSV/JSON 验收通过。Top op 为 `Mul` 19.56%、`Conv3DBackpropFilterV2` 16.03%、`Add` 13.52%、`Conv3DBackpropInputV2` 10.29%、`Conv2DV2` 7.93%。

## 5. 阻塞项
| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| NMS CPU fallback | `torchvision::nms` CPU/Meta kernel 可用,但无 NPU dispatch kernel;torch_npu 对 NPU 张量调用回退 CPU | 对功能不是硬阻塞;对全 NPU 端到端是硬阻塞 | 使用 Ascend 后处理算子、OM 后处理或 AscendC batched NMS;保留 class-aware 语义 | CPU NMS,可复现 mAP 但低阈值极慢 |
| 原始训练入口 CUDA 绑定 | `select_device()` 强制 CUDA;训练 loop 使用 CUDA AMP/GradScaler/empty_cache;loss 多处 `.cuda()` | 对原始 v0.4.0 NPU 训练入口是硬阻塞 | 建立统一 accelerator abstraction;NPU 默认 FP32 或 torch_npu AMP 路径 | 标准 YOLOv6-N 单卡 FP32 已通过适配验证 |
| `fuse_ab` / `distill` loss 分支 | 分支 loss 仍产生 `DT_DOUBLE` 进入 NPU `Min`,fallback 继续调用 `.cuda()` | 对这两个训练模式是硬阻塞 | 将 `loss_fuseab.py`、`loss_distill*.py` 按标准 loss 同步 dtype/device 修正,所有 fallback tensor 回到当前 accelerator | 不启用 `fuse_ab` / `distill` 时标准训练可运行 |
| DDP 多卡训练入口 | `LOCAL_RANK` 分支直接调用 `torch.cuda.set_device` 并设置 CUDA device,未接 HCCL/NPU device | 对多卡 NPU 训练是硬阻塞 | 将 DDP 初始化改为 `torch.npu.set_device`、NPU device、HCCL backend,并修正 DDP device_ids | 单卡 NPU 训练 |

## 6. 结论
- 运行方案(NPU / NPU+CPU / CPU):NPU+CPU。YOLOv6-N 推理主体在 NPU 上跑通且亲和,COCO FP32/FP16 mAP 均与 README 基准对齐;P5/P6 全权重 speed 矩阵已跑通;NMS 与 COCO metric 当前在 CPU。适配后,标准 YOLOv6-N 单 NPU FP32 2 epoch、未 strip checkpoint resume、训练 step profiler 均跑通。
- 待办 / 风险:
  - 优先替换 CPU NMS。只优化卷积主体不能解决端到端检测耗时。
  - 原始训练入口不支持 NPU;`fuse_ab`、`distill`、DDP 多卡训练是实测硬阻塞。完整训练模式需先修 `fuse_ab` / `distill` loss 的 dtype/device 与 fallback,再修 DDP/HCCL 入口。
  - QAT/quant 为 NVIDIA TensorRT INT8 链路,不纳入 Ascend PyTorch FP32/FP16 训练结论。
  - 若目标是生产部署,应走 ONNX -> ATC/OM 或 Ascend 原生后处理路径,验证 CV fusion、NDDMA layout folding 与 NPU NMS。
