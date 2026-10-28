# Project Context — Efficient Retail Object Detection for Edge AI

> **Purpose:** Paste this entire file into a new ChatGPT conversation to restore the project's goals, constraints, technical decisions, and current state. Use `Plan-checklist.md` to track completed tasks. This is a project brief, not a claim that implementation is finished.

**Project name:** Efficient Retail Object Detection and Counting for Edge AI  
**Short name:** `efficient-retail-detection`  
**Prepared:** 2026-10-16  
**Status:** Planning; no implementation milestone has been confirmed complete.  
**Primary use:** Learning + a reproducible portfolio project for the Qualcomm AI Resident (Engineering) application/interviews in Hanoi.

## 1. Why this project exists

Build a real computer-vision system that **detects and counts individual products visible in retail-shelf images**. Use it to learn object detection from fundamentals, experimentally compare architectures, and optimize a model for resource-constrained inference.

The Qualcomm AI Residency Engineering track emphasizes practical AI engineering, including efficient AI and embedded engineering. This project is intended to demonstrate:

1. **AI/ML foundations:** CNNs, feature extraction, detection heads, loss functions, transfer learning, training, and evaluation.
2. **Experimental engineering:** clean data splits, reproducibility, meaningful baselines, ablations, quantitative analysis, and failure cases.
3. **Efficient inference:** exporting models, benchmarking latency and memory, and evaluating quantization and deployment trade-offs.
4. **Software engineering:** a clean PyTorch codebase, reliable preprocessing/postprocessing, tests, a usable demo, and clear documentation.

This project is **not** affiliated with or endorsed by Qualcomm.

## 2. Exact task and scope

### Problem definition

**Input:** One uploaded photo of a store shelf (no live camera required for the MVP).  
**Output:** Bounding boxes around individually detected retail products, one generic label (`product` or `object`), a **count of predicted products**, and optional inference timing.  
**Ground truth:** Bounding boxes around visible individual products in dataset images.  
**Primary target:** single-class dense object detection **plus counting**, not fine-grained product recognition.

### What is deliberately out of scope for the core MVP

- Identifying specific brands, prices, SKU IDs, or product categories from SKU-110K labels (the labels do **not** contain this information).
- Full retail inventory systems, billing, tracking across video, stock forecasting, or a production backend.
- Writing an entire production-ready YOLO/Faster R-CNN/DETR architecture from scratch.
- Requiring a dedicated Snapdragon development board or a webcam.
- Claiming a real-world frame rate without an actual device benchmark.
- Uploading/redistributing the original dataset or including proprietary images in the public repository.

### Optional extensions *after* the core system works

- A more detailed comparison with a Faster R-CNN baseline.
- Learning how DETR works; a DETR experiment only if time and data density constraints are handled.
- ONNX Runtime quantization; TensorRT or Qualcomm AI Hub profiling where feasible.
- A compact smartphone/edge deployment demonstration, if supported hardware is available.

## 3. Dataset contract

**Preferred dataset:** [SKU-110K (original authors)](https://github.com/eg4000/SKU110K_CVPR19).  
**Documentation and loading reference:** [Ultralytics SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k).

Verified dataset properties (documentation checked 2026-10-16):

- **11,743 images**: 8,219 train, 588 validation, 2,936 test.
- More than **1.7 million** annotated objects, roughly **147 products per image** on average.
- **One class only:** each bounding box labels an object, not a brand or SKU identity.
- Dataset size is substantial (Ultralytics documents an approximately **13.6 GB download**); subset-based debugging is important.
- The authors restrict it to **academic and non-commercial use**. Check licenses before publishing outputs; do not rehost data or assume the dataset is unrestricted.

**Data-integrity rules:** Use the provided train/val/test separation; initially take deterministic subsets **within** these splits. Never use test images for tuning, threshold selection, calibration, or model selection. Create calibration data from training examples only. Record sample IDs, sampling seed, and preprocessing settings.

**Important technical pitfall:** Dense shelves and small objects make detection especially hard. Original DETR configurations may allow fewer queries than the number of products in some shelf images; do not use a default DETR setup uncritically.

## 4. Model plan and rationale

| Model | Role | Priority |
|---|---|---|
| **YOLO11n (pretrained)** | Primary baseline and first working model; fine-tune for one-class shelf detection | **P0 — core** |
| **Faster R-CNN (TorchVision)** | Compare one-stage vs two-stage detection on a controlled image subset | **P1 — after core** |
| **DETR** | Study transformer-style object detection, object queries and matching; experiment only with appropriate configuration/resources | **P2 — optional** |

**Why YOLO11n?** It is a compact and practical starting point with established export paths. Fix a specific model version, configuration, library version, and checkpoint rather than constantly switching to newer YOLO families.

**Understanding-first principle:** Implement **IoU**, basic **NMS**, annotation visualization, and **counting metrics** yourself in small, tested exercises. Use mature framework implementations for full detector training. Understand what the libraries do rather than only calling `train()`.

## 5. Chosen tools and environments

- **Language:** Python 3.12 recommended for the repository (document the tested version).
- **Training:** PyTorch, TorchVision, Ultralytics, Google Colab (GPU if available).
- **Data/visualization:** NumPy, Pandas where useful, Pillow/OpenCV, Matplotlib.
- **Evaluation:** COCO-style object-detection metrics (e.g., TorchMetrics/pycocotools), plus a custom counting error metric.
- **Testing:** pytest.
- **Export/inference:** ONNX and ONNX Runtime; optional other runtime(s) if supported.
- **Demo:** a **local image-upload demo** (Gradio is a candidate), which draws boxes and displays the count.
- **Tracking:** small reproducible JSON/CSV experiment records, saved configuration, and checkpoints; external experiment-tracking services are optional.

### Available computing constraints

- **Ubuntu desktop:** Intel UHD Graphics 770; no NVIDIA GPU detected. Good for coding, tests, and CPU inference benchmarking; not a CUDA training machine.
- **Other Windows computer:** RTX 3050 with **4 GB VRAM** was previously reported; use only if current CUDA/PyTorch setup is confirmed, and expect small batch sizes.
- **Google Colab:** practical training option, but sessions/storage are temporary. Save checkpoints, config, and metrics to persistent storage and support resuming training. No need to keep the user's computer powered on for a remote Colab runtime, but Colab runtimes themselves can disconnect.
- **Hardware availability is not guaranteed.** Real Snapdragon inference is a stretch goal, not a dependency of the MVP.

## 6. Architecture of the finished system

```text
Image (file upload)
    |
    v
Decode + validate image
    |
    v
Preprocess / resize (record exact method)
    |
    v
Trained detection model
    |
    v
Decode boxes + confidence threshold + NMS/postprocessing
    |
    +-----> Bounding boxes + detection confidence
    |
    +-----> Product count = number of retained boxes
    |
    v
Annotated image + count + measured inference information
```

**Engineering requirement:** PyTorch and ONNX inference must use equivalent image preprocessing and postprocessing to make accuracy/latency comparisons meaningful.

## 7. Evaluation and experiment protocol

### Detection quality

- Primary: **mAP@0.5:0.95**; secondary: AP@0.5, precision/recall at a specified operating point.
- Counting: **MAE** over images, i.e. mean of `abs(predicted_count - true_count)`. Optionally report RMSE and count bias.
- Failure review: small items, densely packed overlaps, blurry packaging, poor lighting, edge-of-image products, false positives, missed items, and duplicate detections.

### Performance

- Measure **serialized model size**, latency (median and p95), throughput where relevant, peak RAM/VRAM where measurable, hardware/runtime, input shape, and batch size.
- Separate **model-forward latency** from **end-to-end latency** (decode/preprocess/infer/postprocess) if possible.
- Repeat benchmarks after warmup; report sample count and timing methodology.
- Compare PyTorch FP32 with ONNX Runtime FP32 **first**. Explore INT8 only after the ONNX FP32 pipeline works and matches outputs within a documented tolerance.
- Never claim that INT8 is faster merely because a model file is smaller; measure it on the specific backend/hardware.
- Never combine CPU and GPU latency numbers as if they represent a controlled speed comparison.

### Experiment questions

1. Which errors dominate product detection in densely packed images?
2. What happens to accuracy and latency as image resolution changes?
3. On identical splits, how does a compact one-stage detector compare with a two-stage detector?
4. Does ONNX export preserve prediction quality while changing speed?
5. If supported, what accuracy/memory/latency trade-off does INT8 quantization provide?

**No fabricated numbers:** All mAP, latency, memory, FPS, and model-size claims must come from executed experiments, with configuration and device details attached.

## 8. Milestones and priorities

### P0 — working portfolio MVP (target before November 12, 2026 application deadline)

1. Understand and implement IoU/NMS/counting metrics in tests.
2. Create a deterministic train/val subset; inspect boxes and class format.
3. Fine-tune YOLO11n and save a usable checkpoint.
4. Evaluate on validation data; show mAP and counting MAE, plus qualitative examples.
5. Build an image-upload demo or reliable command-line inference path.
6. Add a README, reproducibility instructions, an honest results table, and a short demo/video or screenshots made from properly usable images.

### P1 — deeper interview-quality engineering (after the core MVP)

7. Study and run a Faster R-CNN baseline on an adequately scoped, comparable subset.
8. Analyze model failure cases and the small-object issue in depth.
9. Export the chosen YOLO model to ONNX; verify output parity and benchmark CPU inference.
10. Experiment with input resolution and quantization if backend support permits.
11. Report accuracy vs latency and accuracy vs model-size trade-offs.

### P2 — stretch only

12. Explore TensorRT or Qualcomm AI Hub compilation/profiling on an actually supported target.
13. Study DETR and optionally add a correctly configured comparison.
14. Add extra features only if they improve the core experimental story.

**Schedule reference (as published on 2026-10-16):** Qualcomm Engineering Residency Batch 4 application period October 12–November 12, 2026; entrance test November 18–20; coding interviews December 7–18; technical interviews January 4–15, 2027. This schedule may change. Always prepare for coding interviews separately—this object-detection project does not replace algorithms practice.

## 9. Suggested repository shape

```text
efficient-retail-detection/
├── projec-context.md
├── Plan-checklist.md
├── README.md
├── requirements.txt         # or pinned pyproject.toml
├── .gitignore
├── configs/                 # model, data, preprocessing, evaluation
├── src/
│   ├── data/                # datasets, transformations, label adapters
│   ├── fundamentals/        # IoU, NMS and box utilities
│   ├── training/            # fine-tuning entry points
│   ├── evaluation/          # mAP and counting metrics
│   ├── inference/           # predictor and annotations
│   └── optimization/        # export, quantization, benchmarks
├── tests/
├── notebooks/               # exploration only; source code lives in src/
├── experiments/             # configs, summary CSV/JSON, figures
├── demo/
└── assets/                  # self-created or properly licensed visuals
```

- Avoid committing datasets, private keys, huge model checkpoints, Colab caches, or environment directories.
- If model weights cannot be hosted, document how to obtain/reproduce them legally.
- Keep the repo small at the beginning; add directories when needed.

## 10. Definition of done

### Core MVP is finished when

- [ ] A documented command can train/resume the baseline from the dataset configuration.
- [ ] A documented command can evaluate **validation** detection and counting metrics.
- [ ] An example image can pass end-to-end inference and produce boxes + count.
- [ ] Bounding box conversions, IoU/NMS, and counting have unit tests.
- [ ] At least one real failure case is explained and illustrated legitimately.
- [ ] Repo provides setup, run commands, dataset-license notice, experiment settings, and actual—not invented—results.

### Extended engineering project is finished when

- [ ] A controlled second-model or controlled-resolution comparison exists.
- [ ] ONNX FP32 model runs and its outputs are checked against PyTorch.
- [ ] Inference is benchmarked in a repeatable script on named hardware.
- [ ] Optimization decisions follow evidence about accuracy, latency, and memory.

## 11. How a future ChatGPT conversation should help

**When I paste this file into a new chat:**

1. Treat this file as the current project scope; **do not assume tasks are completed** unless I supply checked boxes, code, logs, or test output.
2. Look at `Plan-checklist.md` to see dependencies and the next unchecked task.
3. Explain unfamiliar theory from first principles with small numerical or PyTorch examples, especially CNN/detection/quantization topics.
4. Separate **what to learn**, **what to understand**, and **what to build**. Explain how each concept directly helps this project.
5. Prefer incremental, runnable Python examples with exact file paths, terminal commands, expected outputs, and tests.
6. Prefer a real engineering implementation over copying an entire tutorial. Show trade-offs and why a design decision is made.
7. Respect Colab interruptions, memory limits, licenses, and the distinction between CPU, GPU, and NPU performance.
8. Do not invent experiment results or completed steps. Ask for actual measurements when analysis requires them.
9. Keep MVP requirements separate from optional enhancements. If time is limited, prioritize a working, measured YOLO system over unfinished comparisons.
10. When updating the plan, preserve already checked work and give me the **next smallest actionable milestone**.

## 12. Verified reference links

- Qualcomm AI Residency official overview + published timeline: https://www.qualcomm.com/research/artificial-intelligence/ai-residency-program
- Engineering job link: https://qualcomm.wd12.myworkdayjobs.com/External/job/Hanoi-VNM/AI-Resident--Engineering-_3097473
- Original SKU-110K data and terms: https://github.com/eg4000/SKU110K_CVPR19
- SKU-110K Ultralytics documentation: https://docs.ultralytics.com/datasets/detect/sku-110k
- D2L computer vision chapter: https://d2l.ai/chapter_computer-vision/index.html
- PyTorch object detection finetuning: https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- YOLO export documentation: https://docs.ultralytics.com/modes/export/
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Qualcomm AI Hub YOLOv11 model information: https://aihub.qualcomm.com/mobile/models/yolov11_det
- Qualcomm AI Hub Android object detection sample: https://aihub.qualcomm.com/apps/object_detection_android

> **Verification note:** The official Qualcomm program page was accessible and was used for the role-track emphasis and dates. The Workday posting itself did not expose readable job text in the web retrieval, so do not treat this file as an exhaustive quotation of the job description.
