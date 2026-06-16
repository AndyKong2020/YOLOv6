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
| 软件 | torch 2.10.0+cpu / torch_npu 2.10.0 / torchvision 0.25.0 / Python 3.12.13 |

---

## 1. 技术栈梳理
- 主语言:Python。
- ML 框架:PyTorch。评估入口为 `tools/eval.py`,COCO 指标走 `pycocotools`。
- CUDA 依赖:原始 v0.4.0 评估与训练代码均默认 CUDA。评估侧默认 CUDA 设备与 `torchvision.ops.nms`;训练侧 `select_device`、AMP/GradScaler、loss、empty_cache、DDP、distill/fuse_ab/QAT 分支均有 CUDA 绑定。
- 自定义核(.cu / C++ 扩展):无必须编译的自定义 CUDA 扩展。当前阻塞来自 `torchvision::nms` 在本 torch/torchvision/torch_npu 组合中不可用,不是项目自带 .cu 编译失败。
- 第三方库:opencv-python-headless、pycocotools、thop、onnx/onnxsim、tensorboard 等。`torchvision` 可安装但 NMS op 不可用。
- 模型权重 / 来源:YOLOv6 v0.4.0 release 官方权重,本轮完整复现使用 `yolov6n.pt`;`yolov6s/m/l` 与 `yolov6n6/s6/m6/l6` 权重已就绪但未跑完整矩阵。

## 2. 部署步骤
- [x] 依赖安装:使用独立 Python 3.12 虚拟环境,安装 torch 2.10.0+cpu、torch_npu 2.10.0、pycocotools、opencv-python-headless、thop 等依赖。
- [x] 编译 / 构建:无需编译项目本体;以源码方式运行评估入口。
- [x] 权重获取:下载 v0.4.0 官方 P5/P6 权重,完整复现选择 YOLOv6-N。
- [x] 数据准备:COCO val2017 图片 5000 张,由 `instances_val2017.json` 生成 YOLO 格式 val label 5000 个。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):评估入口支持 NPU 设备;checkpoint 先 CPU 加载再迁移;PyTorch 2.6+ checkpoint 加载显式兼容;旧权重序列化 class/属性增加 shim;NMS 在 `torchvision.ops.nms` 不可用时走 CPU fallback。
- [x] 训练单卡功能验证适配:本轮验证通过的训练范围限定为标准 YOLOv6-N、单 NPU、FP32、1 epoch 功能验证链路。已覆盖 data loader、forward、loss/assigner、backward、optimizer step、checkpoint save、final eval;未覆盖 AMP、DDP/HCCL、多 epoch 收敛、resume、distill、fuse_ab、QAT/quant 和全量 COCO 训练。
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
  --data-path data/npu_train_smoke.yaml \
  --conf-file configs/yolov6n.py \
  --epochs 1 \
  --batch-size 2 \
  --img-size 320 \
  --workers 0 \
  --eval-final-only \
  --output-dir runs/train_npu_smoke \
  --name patched_npu_smoke_float32_assigner
```

## 3. 验证用例
- 输入数据:COCO val2017,共 5000 张图片;P5 输入尺寸 640,batch size 32。训练功能验证另用 8 张 train + 4 张 val 的小子集,只验证训练程序路径是否可执行,不代表训练精度或收敛性。
- 运行命令:见第 2 节,覆盖 COCO val、FP32 speed、FP16 speed、1 epoch 单卡训练功能验证。
- 期望输出:README 中 YOLOv6-N mAP<sup>val 0.5:0.95</sup> 为 37.5;官方 speed 文档说明公平测速不包含 preprocess 与 NMS。
- 实测输出:
  - forward 功能验证:输入 `1x3x640x640`,输出 `(1, 8400, 85)`。
  - COCO val:AP@[0.50:0.95] = 0.374,AP@0.50 = 0.529,AP@0.75 = 0.404。
  - val 日志耗时:preprocess 0.04 ms/image,inference 0.32 ms/image,NMS 149.31 ms/image。
  - speed FP32:`conf_thres=0.4`,preprocess 0.03 ms/image,inference 0.30 ms/image,NMS 0.94 ms/image。
  - speed FP16:`conf_thres=0.4`,preprocess 0.03 ms/image,inference 0.20 ms/image,NMS 0.89 ms/image。
  - 原始训练入口:使用 NPU 环境直接运行会在 `select_device()` 处因 `torch.cuda.is_available()` 为 False 触发 `AssertionError`。
  - 训练功能验证范围:标准 YOLOv6-N、单 NPU、FP32、batch size 2、image size 320、workers 0、1 epoch、8 张 train + 4 张 val。该范围只证明训练程序路径可执行,不证明全量 COCO 训练可收敛。
  - 训练功能验证结果:完成 1 epoch,4/4 step,生成 `last_ckpt.pt` 与 `best_ckpt.pt`,日志显示 `Using 1 NPU for training`、`Training completed`。已覆盖 forward、loss/assigner、backward、optimizer step、checkpoint save、final eval。
  - 训练未覆盖测试项:AMP/GradScaler、DDP/HCCL、多 epoch 稳定性、resume、distill、fuse_ab、QAT/quant、全量 COCO train2017 和训练性能 profiler 均未执行。本项是测试覆盖范围缺口,不是已确认阻塞;报告仅据此限制训练结论外推。
  - 训练 assigner:首次 NPU 训练补丁会在 label assignment 中因 `DT_DOUBLE` 输入触发 NPU `Min` error code `161002`,随后回退 CPU;将 `loss.preprocess()` 生成的 numpy array 固定为 `float32` 后,同一训练功能验证不再出现 `CPU mode for This Batch`。
- 与 CPU/GPU 基准对比(误差/一致性):mAP 与 README 37.5 基准相差约 0.1 个百分点,可视为复现通过。README 的 T4 TensorRT FPS 属于 TensorRT 部署路径,本报告只对 torch_npu PyTorch 路径负责,不做横向等价比较。

## 4. NPU 亲和性
| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 推理/评估能跑通。训练原始入口不能直接跑;适配后仅声明标准 YOLOv6-N、单 NPU、FP32、1 epoch 功能验证链路通过,覆盖 forward/backward/optimizer/save/final eval。不声明 AMP、多卡、resume、distill、QAT/quant、全量 COCO 训练和多 epoch 收敛可用。 |
| NPU 利用率 (npu-smi) | 未采集稳定利用率曲线;npu-smi 仅用于确认设备可见性与无残留进程。AIC/AIV/MTE 利用率应按注入式 `torch_npu.profiler` 补测。 |
| HBM 占用 | 单卡 HBM 114688MB;YOLOv6-N 峰值 HBM 未单独采样,待 profiler 补测。 |
| 关键算子是否回退 CPU | 推理主体无关键 CPU 回退;NMS 回退 CPU;pycocotools 指标统计在 CPU。训练功能验证在 float32 assigner 修正后未观察到 label assignment CPU fallback,但未用 profiler 排查全部 backward kernel。 |
| 性能(吞吐/时延) | FP32 inference 0.30 ms/image;FP16 inference 0.20 ms/image;CPU NMS 在 speed 阈值下约 0.9 ms/image,在 COCO val 低阈值下 149.31 ms/image。训练功能验证样本太小,不作为吞吐、收敛或资源占用指标。 |

口径:Ascend 950PR、3510 架构、YOLOv6-N、P5 640、batch 32。YOLOv6 是单帧检测模型,不适用 LLM prefill/decode;这里拆为 preprocess、network inference、decode/NMS、COCO metric。FP16 Cube 平衡点约 270 FLOP/Byte,高于为算力主导,低于为访存/搬运主导。

**计算分布**(基于模型结构与实测日志归类,尚未采 Level1 profiler):

| 单元 | 主要算子/路径 | 占比/压力 | 说明 |
|---|---|---:|---|
| 算力(Cube,卷积矩阵) | Conv2D、RepBlock、SPPF、RepPAN、head conv | 主体 | README 给 YOLOv6-N 11.4G FLOPs、4.7M 参数;FP16 inference 0.20ms,主体亲和 NPU。 |
| 向量(Vector,激活解码) | SiLU/ReLU、Sigmoid、bbox decode、score 计算 | 中 | 规则向量路径可跑通;head 末端小 kernel 多,优化空间在融合 decode。 |
| 搬运(MTE/FixPipe) | feature map 读写、concat、reshape/permute/cat | 中 | PyTorch eager 层间物化明显;950 上可考虑 NDDMA/layout folding,但未实测。 |
| 通信(communication) | 无 | 0 | 单卡评估,无 collective。 |
| 调度(host/head) | preprocess、CPU NMS、pycocotools、训练保存/eval | 高 | NMS 是当前端到端瓶颈。训练只验证单卡 FP32 功能验证链路,DDP/HCCL、AMP、resume、distill/fuse_ab/QAT 未验证。 |

**判定**:YOLOv6-N 的网络主体亲和 NPU。按 950PR FP16 Cube 峰值 378/432 TFLOPS 粗估,11.4G FLOPs 的理想 Cube 下界约 0.026-0.030ms/image;实测 FP16 inference 0.20ms/image,差距来自真实 tile 利用率、GM 搬运、eager kernel 边界与 head 开销。若有效 GM 读写超过约 42MB/image,就会从理想 Cube-bound 转为 MTE/GM 或 head-bound;该项需 profiler 采样确认。端到端瓶颈明确在 CPU NMS,不是卷积主体。

- 算子回退清单:
  - `torchvision.ops.nms`:当前不可用,使用 CPU fallback。功能不阻塞,mAP 可复现;端到端性能被严重限制。
  - `pycocotools`:CPU 统计 AP/AR,不属于 NPU kernel 路径。
  - OpenCV/letterbox/preprocess:host 侧处理,当前耗时很小,不是主要瓶颈。
  - 训练 label assignment:原始 NPU 补丁下因 `DT_DOUBLE` 输入触发 NPU `Min` 不支持并回退 CPU;固定 `loss.preprocess()` 为 `float32` 后,单卡训练功能验证未再回退。
- profiler 摘要:
  - 仓内未发现 `enable_profiler`、`tensorboard_trace_handler` 或 `torch_npu.profiler` 内置采集入口,因此 profiling 路由应走注入采集。采集时在目标 loop 外先做一次模型预热,再用 `torch_npu.profiler.profile` 注入推理 forward 或训练 step。
  - profiler 配置必须包含 `torch_npu.profiler._ExperimentalConfig(profiler_level=Level1, aic_metrics=PipeUtilization)`,否则 `kernel_details.csv` 只有基础 9 列,无法分析 Cube/Vector/MTE pipeline utilization。schedule 建议 `wait=0,warmup=0,active=N,repeat=1,skip_first=0`,循环中每步 `torch.npu.synchronize(); prof.step()`,循环结束额外调用一次 `prof.step()` 触发收尾。
  - 产物验收标准:生成 `ASCEND_PROFILER_OUTPUT/`;`kernel_details.csv` 约 47 列;`op_statistic.csv`、`api_statistic.csv`、`operator_details.csv`、`step_trace_time.csv` 存在;`trace_view.json` 为合法 JSON。若 CSV 缺失或 `trace_view.json` 截断,优先排查输出目录所在文件系统并换盘重采/重解析,不要把大体量产物长期留在系统盘。
  - 现有日志只提供 preprocess、inference、NMS 三段时间和训练单卡功能通过证据。AIC/AIV/MTE、tile 驻留、NDDMA/CV fusion、训练 backward/assigner 热点均为待测。

## 5. 阻塞项
| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| NMS CPU fallback | 当前 torch/torchvision/torch_npu 组合中 `torchvision::nms` 不可用 | 对功能不是硬阻塞;对全 NPU 端到端是硬阻塞 | 使用 Ascend 后处理算子、OM 后处理或 AscendC batched NMS;保留 class-aware 语义 | CPU NMS,可复现 mAP 但低阈值极慢 |
| 原始训练入口 CUDA 绑定 | `select_device()` 强制 CUDA;训练 loop 使用 CUDA AMP/GradScaler/empty_cache;loss 多处 `.cuda()` | 对原始 v0.4.0 NPU 训练入口是硬阻塞 | 建立统一 accelerator abstraction;NPU 默认 FP32 或 torch_npu AMP 路径 | 已绕过的范围:标准 YOLOv6-N、单 NPU、FP32、1 epoch、8 train + 4 val |
| 训练 assigner double tensor | `loss.preprocess()` 的 numpy array 默认为 float64,NPU `Min` 不支持 DT_DOUBLE | 对纯 NPU label assignment 是硬阻塞 | 固定 label preprocess 为 float32;后续对 assigner 做 dtype 单测 | 在上述单卡 FP32 功能验证范围内未再 CPU fallback;未证明所有训练配置下均无 fallback |
| head 末端 layout 与小 kernel | reshape/permute/cat/dist2bbox/NMS prep 在 eager 下产生多段小算子与物化 | 非硬阻塞 | 固定 `(N, HW, C)` 物理布局,用 NDDMA/address generation 折叠转换,将 decode 融入后处理 | 继续 eager 执行 |
| 旧权重序列化结构 | 官方权重保存旧 class/属性,直接用 v0.4.0 代码加载会失败 | 非硬阻塞,已用兼容 shim 解决 | 重新导出标准 state_dict 或 ONNX,减少 Python class 依赖 | 保留兼容 shim |
| 全模型矩阵未补齐 | 本轮完整 COCO val 只跑 YOLOv6-N | 非硬阻塞 | 逐个补 YOLOv6-S/M/L 与 P6 1280 输入;采集 profiler | 当前只声明 YOLOv6-N |
| HBM/利用率未 profile | npu-smi 未采稳定利用率曲线,且尚未生成 profiler CSV | 非硬阻塞 | 注入 `torch_npu.profiler` 并使用 Level1 + PipeUtilization,验收 `kernel_details.csv` 约 47 列、`op_statistic.csv`、`api_statistic.csv` 与合法 `trace_view.json` | 用日志三段耗时做初判 |

## 6. 结论
- 运行方案(NPU / NPU+CPU / CPU):NPU+CPU。YOLOv6-N 推理主体在 NPU 上跑通且亲和,COCO mAP 与 README 基准对齐;NMS 与 COCO metric 当前在 CPU。训练原始入口不支持 NPU;适配后仅证明标准 YOLOv6-N、单 NPU、FP32、1 epoch、8 train + 4 val 的功能验证链路可执行,覆盖 forward/backward/optimizer step/save/final eval,不能外推为全量训练、AMP、多卡、resume、distill、QAT/quant 或多 epoch 收敛可用。
- 待办 / 风险:
  - 优先替换 CPU NMS。只优化卷积主体不能解决端到端检测耗时。
  - 训练需要作为单独适配重点:清理 CUDA 绑定,补 torch_npu AMP 或明确 FP32 训练策略,对 assigner/loss 建单测,再补多 epoch 稳定性。
  - 按注入式 `torch_npu.profiler` 方法补采 profiler,确认 Cube/Vector/MTE/host/head 实际占比、训练 backward/assigner 热点与 HBM 峰值。
  - 补 FP16 COCO mAP,当前 FP16 只跑了 speed。
  - 补 YOLOv6-S/M/L 和 P6 1280 输入矩阵,不能把 YOLOv6-N 结果外推到全系列。
  - 补测 AMP/GradScaler、DDP/HCCL、resume、distill、fuse_ab、QAT/quant、全量 COCO train2017 和多 epoch 稳定性。本轮未执行这些训练分支,因此只能记录为未覆盖测试项,不能写成已确认阻塞。
  - 若目标是生产部署,应走 ONNX -> ATC/OM 或 Ascend 原生后处理路径,验证 CV fusion、NDDMA layout folding 与 NPU NMS。
