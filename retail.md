# Plan Checklist — Efficient Retail Object Detection for Edge AI

> **How to use:** Tick tasks as you complete them (`[ ]` → `[x]`). Each phase separates **Learn** (study the concept), **Understand** (explain or verify it), and **Build** (produce working code/artifacts). Do not mark a phase complete from reading alone: pass its checkpoint.

**Start date:** 2026-10-16  
**Current progress:** No items marked complete; update this file after each session.  
**Internal résumé/project release deadline:** **November 8, 2026**.  
**Target application submission:** **November 10, 2026** (ahead of the published November 12 deadline; verify the latest official notice before applying).  
**Entrance-test preparation target:** **November 17, 2026** (test window previously published for November 18–20).  
**Coding-interview preparation target:** **December 6, 2026** (interview window previously published for December 7–18).  
**Technical-interview readiness target:** **January 3, 2027** (interview window previously published for January 4–15).  
**Priorities:** `P0` = required résumé-ready MVP; `P0+` = strongly recommended *if the baseline is already working*; `P1` = advanced work for interviews; `P2` = optional/stretch.  
**Schedule disclaimer:** These are planning dates based on the recruitment timeline discussed on October 16, 2026; confirm all dates and requirements on Qualcomm's official posting.

## GitHub reference map (study only when the phase needs it)

These repositories are **references, not additional deliverables**. Their reported results are **their authors' claims**, not measurements you can use as your own. Follow your own train/validation split, run your own experiments, and check each repository's license before copying/adapting code. **Do not spend days cloning/debugging someone else's full environment**, particularly for older repositories. Sources were checked on **October 16, 2026**.

| When | Reference repository | What to read or study | Use priority |
|---|---|---|---|
| Phases 3–4 | [SKU-110K official research repository](https://github.com/eg4000/SKU110K_CVPR19) | Official annotation conventions, dataset terms, crowded-scene research problems; original RetinaNet code is a **research reference**, not your PyTorch training starter | P0 |
| Phases 4–5, 7 | [Retail Shelf Monitor](https://github.com/rickhaile/retail-shelf-monitor) | `src/train.py`, `src/evaluate.py`, model settings for dense shelves, counting metrics, reproducibility | **P0 — main baseline example** |
| Phases 4–6, 12 | [YOLOv8 / Shelf Monitoring](https://github.com/samusafe/YOLOv8) | `train.py`, `eval.py`, `pipeline.py`, `submissao/infer.py`, `submissao/app/app.py`, `tests/` — pipeline, inference and Streamlit organization | P0 — secondary implementation example |
| Phases 2, 4, 9 | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Official model implementation, train/val/predict/export APIs; cross-check with current documentation | P0 / P0+ |
| Phase 8 | [Faster R-CNN on SKU-110K](https://github.com/naviocean/faster_rcnn_sku110) | Detectron2 setups and lightweight backbones (MobileNet, ShuffleNet, EfficientNet, etc.); **older PyTorch/CUDA stack** | P1 |
| Phases 9–11 | [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) | Find `qai_hub_models.models.yolov11_det`; explore export, inference I/O, target-specific profiling and deployment limitations | P0+ / P1 / P2 |
| Phase 11 (stretch) | [Retail Shelf Recognition](https://github.com/k0202122/retail-shelf-recognition) | API/UI integration and *optional* product identity via embeddings and OCR; **beyond your one-class counting MVP** | P2 |

**How to use each repo (not a new mandatory checklist):** first read the README, then trace one entrypoint → data loading → model call → evaluation/output. Write your own code, add small tests, and note any ideas you adapt in your README. You may consult older or alternate architectures without reproducing their full dependency stacks. When short on time, prioritize **Retail Shelf Monitor + the official SKU-110K documentation**, and only consult the others when you reach their phases.

---

## Execution order at a glance

| Order | Phase | Priority | Main tangible output |
|---|---|---|---|
| 0 | Freeze scope and repo | P0 | Small runnable repository |
| 1 | CNN/PyTorch prerequisites | P0 | Shape/backprop explanations + sanity script |
| 2 | Detection foundations | P0 | Tested IoU, NMS, bounding box utilities |
| 3 | Dataset pipeline | P0 | Validated visualized SKU-110K subset |
| 4 | First YOLO baseline | P0 | Trained checkpoint + training logs |
| 5 | Evaluation and error analysis | P0 | mAP, counting MAE, failure examples |
| 6 | Inference demo + README | P0 | Upload image → boxes + count |
| 7 | Baseline benchmark + application package | P0 | Reproducible latency report + portfolio summary |
| 8 | Architecture study + Faster R-CNN | P1 | Comparable second-model evaluation |
| 9 | ONNX export, parity and benchmarks | P0+ | Working ONNX Runtime FP32 path, if baseline is ready |
| 10 | Quantization and optimization | P1 | Evidence-based trade-off table |
| 11 | Edge deployment and DETR | P2 | Optional additional deployment/research results |
| 12 | Interview preparation and release | P0/P1 | Explanation, demo, tested GitHub repository |

**Short-term execution calendar (internal targets, not guarantees):**

- **Oct 16–19:** Phases 0–2 essentials; implement and test IoU/NMS. Review Phase 1 *just in time* rather than studying every CNN topic before coding.
- **Oct 20–24:** Phase 3 dataset subset and Phase 4 pretrained YOLO inference + training smoke test.
- **Oct 25–31:** Finish first reproducible YOLO fine-tuning run and Phase 5 evaluation.
- **Nov 1–5:** Phase 6 image → boxes + count demo; Phase 7 baseline latency/size benchmark. Attempt Phase 9 (ONNX) only if the P0 pipeline is stable.
- **Nov 6–8:** Phase 12 résumé-version README, tests, demo visuals, truthful measurements; freeze the application-ready release.
- **Nov 9–10:** Finalize résumé and submit the application; **do not wait until November 12** if you can submit earlier.
- **Nov 11–20:** Prioritize entrance-test revision and timed practice; only maintenance on the project.
- **Nov 21–Dec 18:** Prioritize coding practice/interviews; optional P1 project progress if time allows.
- **Dec 19–Jan 3:** Finish interview-relevant architecture and optimization studies/experiments (Phases 8–10) and prepare project explanations.
- **Jan 4–15:** Technical interviews and targeted review; avoid disruptive refactors.

If Colab limits or debugging make the schedule unrealistic, reduce the number of training images/epochs and label results honestly as **subset/prototype** findings, not full-dataset benchmarks. **Always protect test time and application time before adding features.**

## Two project completion gates

### Gate A — Résumé-ready by November 8 [P0]

- [ ] A fresh setup can run the trained YOLO detector on a retail image and output bounding boxes + a product count.
- [ ] Document the SKU-110K dataset source, license, image IDs/splits, and whether training was on a subset or full training data.
- [ ] Complete a repeatable training run and preserve checkpoint, configuration, library versions, and seed.
- [ ] Report **real** held-out validation mAP and counting MAE with evaluation size and confidence threshold.
- [ ] Report repeatable inference latency and hardware details (and model size if possible).
- [ ] Publish clean code, tests/smoke-test instructions, README, and sample predictions you may legally share.
- [ ] Write an accurate 2–4-bullet résumé project summary, using measured numbers only where available.
- [ ] **Optional for Gate A [P0+]:** ONNX export, verified prediction parity, and same-machine latency comparison.
- [ ] Prepare the application and aim to **submit November 10**; verify the official deadline.

**Stop rule:** If training/evaluation/demo/README are incomplete by November 5, postpone ONNX, Faster R-CNN, quantization and other P1/P2 tasks.

### Gate B — Technical-interview-ready by January 3 [P1]

- [ ] Explain the YOLO detection pipeline, NMS, mAP, dense-scene failure modes, and counting MAE clearly.
- [ ] Explain Faster R-CNN vs YOLO; ideally have a controlled comparison on common image IDs.
- [ ] Validate ONNX FP32 inference and compare against PyTorch under documented conditions.
- [ ] Understand PTQ calibration, operator support, accuracy changes, and why INT8 may not accelerate a CPU.
- [ ] Attempt and report one controlled optimization experiment (resolution and/or quantization), including negative results.
- [ ] Prepare one accuracy–latency–size trade-off table and a 2-minute project presentation.
- [ ] Complete the separate selection-preparation track below; the AI project is **not** a substitute for timed coding and math practice.

**Optional throughout [P2]:** DETR training, TensorRT deployment, actual Snapdragon on-device benchmarking, SKU/product identification, full-stack application, extra models. None should delay Gate A.

## Daily time allocation (8 hours of focused study, with breaks)

| Time period | Project | Algorithms/programming | Math + ML theory | Documentation / mock practice |
|---|---:|---:|---:|---:|
| Oct 16–Nov 10 | 4.5 h | 2 h | 1 h | 0.5 h |
| Nov 11–20 | 1 h | 4 h | 2.5 h | 0.5 h |
| Nov 21–Dec 18 | 1.5 h | 4 h | 2 h | 0.5 h |
| Dec 19–Jan 15 | 3 h | 1 h | 3 h | 1 h |

Adjust if the official selection schedule changes. These are upper-bound planning blocks, not a requirement to study without rest.

---

## Phase 0 — Scope, environment and repository [P0]

### Learn
- [ ] Understand the full ML workflow: data → train → validate → test → inference → deployment.
- [ ] Learn the difference between source code, model weights, experiment artifacts, and datasets.
- [ ] Review dataset and model-code licenses before publicly sharing artifacts.

### Understand (answer these without notes)
- [ ] Define the single-class prediction target: one bounding box per visible product.
- [ ] Explain why product detection/counting is **not** brand/SKU recognition.
- [ ] State P0/P1/P2 and decide what is out of scope.
- [ ] Explain why using Colab for training and local CPU for benchmarking is reasonable, but comparing their raw latencies is not.

### Build
- [x] Initialize `efficient-retail-detection/` as a Git repository.
- [x] Add `projec-context.md`, `Plan-checklist.md`, `.gitignore`, a first `README.md`, and a requirements/environment file.
- [x] Create a minimal `src/` package and `tests/` directory; avoid empty complexity.
- [x] Verify `python --version`, installed `torch`/`torchvision`, and CPU/CUDA device availability **in each environment**.
- [ ] Run a minimal PyTorch inference/forward-pass script locally or in Colab.
- [ ] Set up durable Colab checkpoint storage; make sure no passwords/API keys are committed.
- [ ] Pin and record working library versions and Python environment.

**Checkpoint:** A fresh clone can install dependencies and run a simple script and `pytest` successfully.

---

## Phase 1 — CNN + PyTorch knowledge needed for detection [P0]

### Learn
- [x] Review tensor dimensions `N,C,H,W`, Conv2D, padding, stride, feature maps, and receptive fields.
- [x] Study ReLU, BatchNorm, residual/skip connections, and pooling.
- [x] Understand CNN backbones (ResNet and a lightweight MobileNet-style backbone).
- [x] Review `Dataset`, `DataLoader`, `model.train()`, `model.eval()`, `torch.no_grad()`, gradient flow, optimizer, and loss functions.
- [x] Study transfer learning: frozen feature extractor vs fine-tuning an entire model.

### Understand
- [x] Explain why deeper features can recognize higher-level patterns.
- [x] Explain how stride/downsampling affects the ability to detect **small products**.
- [ ] Explain how a detector can reuse a pretrained classification backbone.
- [ ] Explain training vs inference BatchNorm behavior and why `model.eval()` matters.
- [ ] Explain overfitting, data augmentation, learning rate, batch size, and image-resolution trade-offs.

### Build
- [ ] Write `src/fundamentals/tensor_shapes.py` to print shapes through a tiny CNN.
- [ ] Create one simple PyTorch data-loading/training sanity example (no need to retrain a large classifier).
- [ ] Confirm model checkpoints can be saved and reloaded reproducibly.

**Checkpoint:** Given an input image tensor and a sequence of convolutions, predict feature-map shapes and explain what information is lost during downsampling.

**Resources:** [D2L modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html) · [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).

---

## Phase 2 — Object detection fundamentals [P0]

### Learn
- [ ] Classification vs localization vs object detection vs instance segmentation.
- [ ] Bounding box formats: `xyxy`, `xywh`, normalized vs pixel coordinates.
- [ ] Intersection over Union (IoU) and what it measures.
- [ ] Confidence scores, detection thresholds, false positives, false negatives.
- [ ] Non-Maximum Suppression (NMS) and what can go wrong in crowded scenes.
- [ ] Anchors vs anchor-free detection at a conceptual level.
- [ ] Multiscale feature maps and feature pyramids.
- [ ] Precision, recall, PR curve, AP, `mAP@0.5` and `mAP@0.5:0.95`.
- [ ] Classification/objectness/localization loss components at the architectural level.

### Understand
- [ ] Calculate IoU by hand for two simple boxes.
- [ ] Explain why IoU threshold and confidence threshold solve different problems.
- [ ] Explain why NMS can mistakenly remove two adjacent overlapping products.
- [ ] Explain why mAP is not the same as simple per-image counting accuracy.
- [ ] Explain why high recall with many duplicate boxes may still produce poor counts.
- [ ] Describe how resizing and letterboxing affect bounding box coordinates.

### Build
- [ ] Implement `xyxy ↔ xywh` conversion functions in `src/fundamentals/boxes.py`.
- [ ] Implement IoU **from scratch** in `src/fundamentals/iou.py`.
- [ ] Implement a simple NMS **from scratch for learning** in `src/fundamentals/nms.py` (not necessarily the production NMS).
- [ ] Add unit tests for identical boxes, no overlap, partial overlap, invalid/degenerate boxes, score sorting, and coordinate conversion.
- [ ] Draw boxes and scores on a sample image using Matplotlib/OpenCV.

**GitHub code reference [P0]:** [Ultralytics](https://github.com/ultralytics/ultralytics) — browse prediction and postprocessing concepts **after** implementing simple IoU/NMS yourself. Treat production implementations as a comparison, not a substitute for the learning exercises.

**Checkpoint:** Explain and demonstrate a missed product and a duplicate detection. Unit tests pass for your box utilities.

**Resources:** [D2L object detection and bounding boxes](https://d2l.ai/chapter_computer-vision/bounding-box.html) · [D2L anchor boxes/IoU/NMS](https://d2l.ai/chapter_computer-vision/anchor.html) · [D2L multiscale](https://d2l.ai/chapter_computer-vision/multiscale-object-detection.html).

---

## Phase 3 — SKU-110K data exploration and data pipeline [P0]

### Learn
- [ ] Read the [original dataset documentation](https://github.com/eg4000/SKU110K_CVPR19) and its academic/non-commercial-use restriction.
- [ ] Understand SKU-110K's single-class labels and train/validation/test split.
- [ ] Study annotation coordinates and the YOLO text annotation format.
- [ ] Study object-density distribution and how input resolution affects tiny products.
- [ ] Understand train/validation/test leakage and the purpose of deterministic sampling.

### Understand
- [ ] Explain why `SKU-110K` does **not** mean 110,000 label classes.
- [ ] Explain why resizing images requires transforming the ground-truth boxes in the same coordinate system.
- [ ] Explain why model tuning and quantization calibration must not use test images.
- [ ] Explain why dense scenes can expose annotation quality issues and NMS limitations.

### Build
- [ ] Assess available disk space and data-transfer limits before downloading the full ~13.6 GB dataset.
- [ ] Obtain the dataset from an authorized official source and keep it **outside Git**.
- [ ] Inspect examples, annotation files, split definitions, and image dimensions.
- [ ] Create deterministic **small** training and validation subsets by sampling within the official splits; document IDs and seed.
- [ ] Verify all boxes have valid coordinates, nonnegative areas, and one supported class ID.
- [ ] Visualize at least 10 annotated images with varied crowding and object sizes.
- [ ] Compute basic data statistics: objects/image, object-area distribution, image dimensions, corrupt/missing files.
- [ ] Implement a loader/adaptor or use the Ultralytics dataset setup; validate conversion by overlaying annotations.
- [ ] Add basic data checks and smoke tests.

**GitHub code reference [P0]:** [SKU-110K official repo](https://github.com/eg4000/SKU110K_CVPR19) — consult the README, annotation format, official download conditions and crowded-scene examples. Its older Keras/RetinaNet environment is **not** required for this PyTorch MVP.

**Checkpoint:** A reproducible data subset loads into a detector and visual inspection confirms that bounding boxes align with actual products.

**Resource:** [Ultralytics SKU-110K setup](https://docs.ultralytics.com/datasets/detect/sku-110k).

---

## Phase 4 — First YOLO11n training baseline [P0]

### Learn
- [ ] Study one-stage detection conceptually: backbone, neck/multiscale features, detection head, decoding/NMS.
- [ ] Study the YOLO11n model architecture at a level where you can describe the feature pipeline.
- [ ] Learn transfer learning, pretrained checkpoints, model-specific augmentation, and optimizer scheduling.
- [ ] Learn training logging: train vs validation loss/metrics, checkpointing and resuming.

### Understand
- [ ] Explain what pretrained weights provide and why single-class fine-tuning is needed.
- [ ] Explain what `imgsz`, batch size, epochs, learning rate, and data config mean.
- [ ] Explain why no single training loss value is enough to establish detection quality.
- [ ] Explain likely GPU memory bottlenecks, and what to do if Colab runs out of memory.
- [ ] Explain why a shorter subset experiment is a smoke test, not a comparable benchmark to published full-dataset results.

### Build
- [ ] Run **inference first** with an unmodified pretrained YOLO11n model and save sample predictions.
- [ ] Train on a tiny SKU-110K training subset for a few iterations; confirm no data/shape errors.
- [ ] Fine-tune the one-class detector on a fixed, documented training subset with a held-out validation subset.
- [ ] Save final/best checkpoint, config, random seed, runtime library versions, and training logs to persistent storage.
- [ ] Add a `train` entrypoint/script or documented CLI invocation that works from a clean environment.
- [ ] Ensure training can resume from saved checkpoints after a Colab runtime disconnect.
- [ ] Inspect learning curves and qualitative predictions; note overfitting/underfitting.

**GitHub code references [P0]:** [Retail Shelf Monitor](https://github.com/rickhaile/retail-shelf-monitor) — start with `src/train.py`, noting data paths, `fraction`, image size, `max_det` and checkpoint settings. [Ultralytics](https://github.com/ultralytics/ultralytics) — use as the authoritative framework code/API reference. [samusafe/YOLOv8](https://github.com/samusafe/YOLOv8) — read `train.py`/`pipeline.py` for an alternative script structure; despite the repo name, its training CLI also describes YOLO11.

**Checkpoint:** A trained `best` checkpoint reloads successfully and predicts boxes on validation images. Its metrics are clearly identified as **subset** or **full-split** metrics.

**Resource:** [Ultralytics training docs](https://docs.ultralytics.com/modes/train/).

---

## Phase 5 — Evaluation, counting, and failure analysis [P0]

### Learn
- [ ] Learn COCO-style evaluation and how score thresholds change precision/recall.
- [ ] Define counting **MAE**, **RMSE** (optional), and bias.
- [ ] Study error types: false positives, false negatives, localization errors, duplicate boxes.
- [ ] Learn how to compare predictions visually against ground truth.

### Understand
- [ ] Explain why the optimal confidence threshold for counting can differ from the setting maximizing mAP.
- [ ] Explain why `predicted_count = number of retained detections` is simple but not always accurate.
- [ ] Explain how to avoid cherry-picking visualization examples.
- [ ] Explain the risks of comparing results from different splits or incompatible preprocessing.

### Build
- [ ] Implement `src/evaluation/counting.py` with `predicted_count`, per-image absolute error, and dataset MAE.
- [ ] Unit-test counting MAE with hand-computed examples.
- [ ] Evaluate YOLO on the held-out **validation** set using an established mAP implementation.
- [ ] Sweep confidence thresholds using validation data only; report the selected threshold.
- [ ] Create a CSV/JSON report with image ID, GT count, predicted count, count error, configuration, and inference version.
- [ ] Save sample qualitative success/failure cases and explain **at least 3** recurrent failure modes.
- [ ] Record actual mAP, counting MAE and evaluation set size; never manufacture missing metrics.

**GitHub code references [P0]:** [Retail Shelf Monitor](https://github.com/rickhaile/retail-shelf-monitor) — inspect `src/evaluate.py` for detection + count metrics and dense-scene settings; **recompute metrics yourself**. [samusafe/YOLOv8](https://github.com/samusafe/YOLOv8) — read `eval.py` and `tests/` for report and test organization.

**Checkpoint:** You can answer: “How accurate is my detector? How wrong are its product counts? On what images does it fail?”

---

## Phase 6 — An image-upload inference demo [P0]

### Learn
- [ ] Understand `model.eval()` and inference-mode execution.
- [ ] Learn input validation, RGB/BGR differences, letterboxing, confidence filtering, and drawing boxes.
- [ ] Study a minimal local Gradio interface **or** a reproducible Python CLI.

### Understand
- [ ] Explain each step from raw image bytes to boxes in original-image coordinates.
- [ ] Explain why inference must use the same class mapping and compatible preprocessing as evaluation.
- [ ] Explain the difference between a display label and an actual dataset class.

### Build
- [ ] Implement a single entrypoint: `predict(image_path, model_path, conf, iou) -> boxes, scores, count` or equivalent.
- [ ] Validate accepted file formats, input errors, and images with **zero detections**.
- [ ] Overlay boxes on the original image and display the count.
- [ ] Add a local upload UI or CLI output that saves an annotated image (UI optional if time is short).
- [ ] Add a minimal smoke test for the inference pipeline.
- [ ] Create README run instructions and a demo example using self-created or properly licensed imagery.

**GitHub code reference [P0]:** [samusafe/YOLOv8](https://github.com/samusafe/YOLOv8) — follow `submissao/infer.py` for a simple image → annotations/JSON CLI; study `submissao/app/app.py` **only if** you choose a Streamlit UI. Do not copy unnecessary webcam/multi-model features into Gate A.

**Checkpoint:** Another person can provide an image and reproduce boxes + count without running a training notebook.

---

## Phase 7 — Baseline benchmarking and application package [P0]

### Learn
- [ ] Difference between model-only and end-to-end latency.
- [ ] Difference between throughput, latency, FPS, and utilization.
- [ ] CPU vs GPU hardware benchmarking and the importance of warmup/repetitions.

### Understand
- [ ] Explain why a Colab T4 latency number and an Ubuntu CPU latency number are not directly comparable.
- [ ] Explain why batch size, resolution, precision and postprocessing must be recorded.
- [ ] Explain why median/p95 results are more informative than a single run.

### Build
- [ ] Write `src/optimization/benchmark.py` or similar.
- [ ] Record hardware, OS, runtime version, threads, precision, image shape, batch size, timing scope, warmup and number of timed runs.
- [ ] Benchmark the PyTorch baseline at batch size 1 and a documented resolution; report median and p95 latency.
- [ ] Record serialized model size and peak memory where measurable.
- [ ] Summarize validation metrics and baseline performance in a transparent README table.
- [ ] Prepare a 30–60-second demo or concise visual walkthrough for a portfolio application (only include results that actually exist).

**GitHub code references [P0]:** [Retail Shelf Monitor](https://github.com/rickhaile/retail-shelf-monitor) — examine how its README documents speed, hardware and counting results, but **do not use its reported speed as your own benchmark**. [samusafe/YOLOv8](https://github.com/samusafe/YOLOv8) — inspect `eval.py`/`pipeline.py` to learn how reproducible results are recorded.

**Checkpoint:** The P0 submission has working code, visible predictions, reproducible results and a benchmark; no claim of optimization is needed yet.

---

## Phase 8 — Learn detection architectures and compare Faster R-CNN [P1]

### Learn
- [ ] Understand Faster R-CNN: backbone → RPN → RoI feature extraction → class and box head.
- [ ] Understand YOLO one-stage prediction and training at an architectural level.
- [ ] Learn region proposals, foreground/background assignment and regression targets.
- [ ] Learn why dense small objects can challenge proposal selection and max detections/image settings.

### Understand
- [ ] Explain one-stage vs two-stage detectors without just listing names.
- [ ] Explain how RoI Align relates proposals to backbone features.
- [ ] Explain why defaults limiting detections/image can invalidate dense-shelf evaluation; inspect detector/evaluator limits.
- [ ] Explain why a fair benchmark needs comparable train/validation subsets, resolution and evaluation settings, while architecture-specific preprocessing can still differ.

### Build
- [ ] Implement a minimal TorchVision detection Dataset returning images and target boxes/labels.
- [ ] Fine-tune a pretrained Faster R-CNN on a manageable, **identical** image-ID subset as YOLO.
- [ ] Check class indexing, transforms, max detections/image, and evaluation configuration.
- [ ] Evaluate both models with compatible COCO-style metrics and counting MAE.
- [ ] Compare parameter count, serialized model size, latency, and qualitative failures, clearly documenting different training budgets if unavoidable.
- [ ] Write 1–2 pages of analysis on **why** the measured differences appeared.

**GitHub code reference [P1]:** [Faster R-CNN on SKU-110K](https://github.com/naviocean/faster_rcnn_sku110) — inspect its `projects/sku110/` and `datasets/sku110/` directories and lightweight-backbone experiments. It targets **PyTorch 1.4 / CUDA 10.2**, so use current TorchVision code for your own runnable baseline rather than copying its historical environment.

**Checkpoint:** You can walk through YOLO and Faster R-CNN diagrams and support an architecture choice with controlled measurements.

**Resource:** [TorchVision detection finetuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

---

## Phase 9 — ONNX export, correctness, and inference benchmarking [P0+]

**Timing:** Try before the résumé deadline *only if* Phases 4–7 and the Gate A README already work. Otherwise complete this phase after submitting the application.

### Learn
- [ ] Learn ONNX: computation graphs, opsets, model export, runtime/execution providers.
- [ ] Learn portability limits: unsupported operators, opset versions, graph preprocessing, output format differences.
- [ ] Understand NMS placement (inside vs outside graph) and how that affects parity checks.

### Understand
- [ ] Explain what ONNX **is**, and why model export is not itself a guarantee of faster inference.
- [ ] Explain model-output parity vs bounding-box parity.
- [ ] Explain why consistent preprocessing/NMS and confidence filtering matter when comparing runtimes.

### Build
- [ ] Export the trained YOLO11n model to ONNX and record opset, input shape and export settings.
- [ ] Inspect ONNX graph inputs/outputs and validate loading with ONNX Runtime.
- [ ] Test PyTorch vs ONNX on the same fixed images (numerical tolerance and postprocessed box comparison).
- [ ] Measure ONNX Runtime FP32 median/p95 CPU latency on the **same machine** as PyTorch CPU.
- [ ] Compare accuracy on the **same validation image IDs**.
- [ ] Save a reproducible benchmark script and raw results.

**GitHub code references [P0+]:** [Ultralytics](https://github.com/ultralytics/ultralytics) — follow its current export flow, then inspect the model input/output formats. [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) — locate `qai_hub_models.models.yolov11_det` and study its export/inference assumptions for later deployment. **First prove local ONNX parity; Qualcomm deployment is not required here.**

**Checkpoint:** ONNX FP32 gives comparable detections to PyTorch within a documented tolerance; speed claims are measured, not assumed.

**Resources:** [Ultralytics export](https://docs.ultralytics.com/modes/export/) · [ONNX Runtime](https://onnxruntime.ai/docs/).

---

## Phase 10 — Quantization and efficient model design [P1]

### Learn
- [ ] FP32, FP16, INT8: precision, memory footprint, numerical range.
- [ ] Scale and zero-point in linear quantization.
- [ ] Post-training static quantization (PTQ), calibration, and quantization-aware training (conceptual extension).
- [ ] CPU instruction support and execution-provider operator coverage.
- [ ] Effects of input image resolution on FLOPs, memory and small-object recall.
- [ ] Optional: pruning, operator fusion and knowledge distillation as concepts (not required implementations).

### Understand
- [ ] Explain why representative **training** data is used for calibration.
- [ ] Explain why naive INT8 export can fail or run slower on unsupported operators/hardware.
- [ ] Explain the difference between model size and runtime peak memory.
- [ ] Explain why increasing resolution may help AP but hurt latency and memory usage.

### Build
- [ ] First run a controlled resolution experiment (e.g. same model family at two documented resolutions).
- [ ] Verify that the exported ONNX model and runtime support the intended quantization path and all required operators.
- [ ] If supported, use **static PTQ** with a representative calibration subset drawn from training images only.
- [ ] Verify quantized output validity and rerun mAP/counting MAE on the same validation images.
- [ ] Benchmark ONNX FP32 vs quantized variant **on the same hardware and at the same input size**.
- [ ] Record what quantizes, what stays FP32, model sizes, median/p95 latency, memory, and measured quality changes.
- [ ] If quantization fails or degrades the model, **document why** and retain the working FP32 ONNX result as a valid outcome.
- [ ] Produce an accuracy–latency–size table and choose a justified deployment configuration.

**GitHub code reference [P1]:** [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) — learn how model input, output, conversion and device profiling are handled; do not assume it supports arbitrary fine-tuned checkpoints or provides a ready-made INT8 pipeline for your model. Pair this with the ONNX Runtime quantization documentation above.

**Checkpoint:** You can explain which tested change improves your target constraints and what detection quality it costs.

**Resource:** [ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

---

## Phase 11 — Optional advanced deployment and DETR learning [P2]

**Timing:** Do not start this phase before the application, entrance test, or coding interview at the expense of higher-priority work.

### Learn
- [ ] Understand GPU vs NPU acceleration and portability between inference backends.
- [ ] Learn TensorRT at a conceptual level; try only if compatible NVIDIA hardware/software is available.
- [ ] Learn Qualcomm AI Hub compilation/profiling and model compatibility limitations.
- [ ] Learn DETR object queries, attention, bipartite/Hungarian matching, and its difference from NMS-based models.

### Understand
- [ ] Explain why edge-device benchmarks require actual target-device measurements.
- [ ] Explain how detector limits on predictions/queries per image interact with SKU-110K density.
- [ ] Explain how to distinguish a device-profiling result from a locally measured result.

### Build
- [ ] **Optional:** test Qualcomm AI Hub tooling on a compatible model and document whether it supports this **fine-tuned** checkpoint.
- [ ] **Optional:** measure performance on actual supported Snapdragon hardware if accessible.
- [ ] **Optional:** benchmark TensorRT on the Windows NVIDIA machine with supported software.
- [ ] **Optional:** run a carefully configured DETR experiment on a less crowded subset or increase query capacity if appropriate.
- [ ] Clearly label simulations, cloud profiling, local CPU experiments, and actual on-device measurements.

**GitHub code references [P2]:** [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) — investigate deployment and actual device profiling **only if compatible hardware/service access is available**. [Retail Shelf Recognition](https://github.com/k0202122/retail-shelf-recognition) — see `api/`, `frontend/` and `src/shelf_recognition/` for possible future app expansion; product identity/OCR is not a requirement for the résumé-ready detector.

**Checkpoint:** No stretch feature is considered completed without a reproducible run or a documented supported external benchmark.

**Resources:** [Qualcomm AI Hub model example](https://aihub.qualcomm.com/mobile/models/yolov11_det) · [Qualcomm Android object detection sample](https://aihub.qualcomm.com/apps/object_detection_android) · [DETR documentation](https://huggingface.co/docs/transformers/model_doc/detr).

---

## Phase 12 — Technical interview, documentation and final release [P0/P1]

**Timing:** Complete a *minimal* README and reproducible release for the résumé by November 8 [P0]. Return for deeper model trade-offs and technical-interview rehearsals by January 3 [P1].

### Understand: practice these questions
- [ ] Why did I choose object detection rather than classification?
- [ ] What makes densely packed retail shelves a difficult detection benchmark?
- [ ] Why YOLO11n? How does its inference path work?
- [ ] What are IoU, NMS, AP/mAP, precision and recall? Can I work through a numerical example?
- [ ] Why can counting MAE disagree with a detector's mAP?
- [ ] What causes small-object misses and duplicate detections?
- [ ] What is the difference between a two-stage and a one-stage detector?
- [ ] What does ONNX change? What does ONNX **not** automatically improve?
- [ ] How do static INT8 calibration and QDQ-style quantization work at a high level?
- [ ] What are the latency/accuracy/model-size trade-offs, **based on my own measured results**?
- [ ] How would I adapt this system for a genuinely limited edge device?
- [ ] What didn't work, what changed, and what would I do with another month?

### Build
- [ ] Clean README: purpose, dataset/legal note, architecture diagram, setup, train, evaluate, infer, reproduce, results, limitations.
- [ ] Include configuration, seed, hardware, runtime, split IDs, inference thresholds and model checkpoint information.
- [ ] Add model/evaluation pipeline tests; run the full test suite.
- [ ] Ensure `git status` does not include datasets, credentials, virtual environments, or large accidental artifacts.
- [ ] Run one complete reproduction from a fresh environment or clean Colab session.
- [ ] Prepare a concise project pitch (problem → approach → experiments → limitations → impact).
- [ ] Create a short demo from images with legitimate sharing rights.
- [ ] Separately prepare algorithm/data-structure interview questions; this project is not a replacement for the coding round.

**GitHub code reference [P0/P1]:** [samusafe/YOLOv8](https://github.com/samusafe/YOLOv8) — read its README layout, `tests/`, experiment manifests and inference instructions for documentation ideas. Cite any adapted work, but present **your own** methods, splits, results, failures and limitations.

**Final checkpoint:** Another engineer can run the project, reproduce at least one reported experiment, inspect honest limitations, and understand your architectural and optimization decisions.

---

## Parallel Track — Qualcomm entrance test and interviews (independent of project phases)

> **Do not wait until the retail detector is finished.** Spend time on algorithms and theory from the first week. The subjects below are sensible preparation topics, **not a verified official exam syllabus**.

### Track A — Before November 17: entrance-test preparation [P0, parallel]

#### Learn
- [ ] Review linear algebra: matrices, dot products, matrix multiplication, vectors, eigenvalues (conceptually).
- [ ] Review probability: conditional probability, Bayes' theorem, independence.
- [ ] Review statistics: expectation, variance, covariance, standard deviation.
- [ ] Review calculus: derivatives, gradients, chain rule and partial derivatives.
- [ ] Review optimization: gradient descent, SGD, Adam and learning rates.
- [ ] Review ML basics: train/validation/test, overfitting, regularization, losses, metrics.
- [ ] Review deep learning: backpropagation, CNNs, BatchNorm and transfer learning.
- [ ] Review programming fundamentals: arrays, hashing, binary search, BFS/DFS and dynamic programming.

#### Understand
- [ ] Explain why gradient-based optimization needs derivatives and how gradients flow backward.
- [ ] Derive the variance of a sum of independent random variables in a simple example.
- [ ] Explain bias–variance trade-offs and data leakage with examples.
- [ ] Explain time and space complexity of your most common algorithm patterns.

#### Build / practice
- [ ] Solve timed programming problems each week, recording mistakes and fixes.
- [ ] Practice short written or oral explanations of math and machine learning fundamentals.
- [ ] Complete at least one timed, mixed-topic mock assessment before November 17.
- [ ] Verify the actual examination instructions, format, permitted language and logistics when officially provided.

**Checkpoint:** Able to work through mixed math/ML/programming questions under a time limit without looking up basic formulas.

### Track B — November 21–December 6: coding-interview preparation [P0, parallel]

#### Learn
- [ ] Review arrays/strings, two pointers, sliding window, hashing and binary search.
- [ ] Review heaps, trees, BFS/DFS, graph shortest paths and union-find.
- [ ] Review greedy algorithms, dynamic programming and complexity analysis.

#### Understand
- [ ] For each solution, explain a brute-force approach, optimized approach and proof/intuition.
- [ ] Explain worst-case time and auxiliary space complexity.
- [ ] Identify off-by-one, empty-input, duplicate and overflow cases without prompting.

#### Build / practice
- [ ] Solve 2–4 carefully reviewed problems on practice days; prefer learning from errors over volume.
- [ ] Conduct at least two timed coding mock interviews before December 7.
- [ ] Maintain a concise error log and revisit unsolved questions.
- [ ] Verify the official interview format and logistics when available.

**Checkpoint:** Can explain and code standard problems in a timed setting and test them independently.

### Track C — December 19–January 3: technical-interview readiness [P1, parallel]

#### Learn
- [ ] Revisit CNN, residual networks, BatchNorm, transfer learning and object detection losses.
- [ ] Review YOLO, Faster R-CNN, feature pyramids, postprocessing and evaluation.
- [ ] Review ONNX/runtime differences, inference latency, memory and quantization.
- [ ] Revisit other stated experience (e.g., your A2A coding-agent project) only to the extent you can demonstrate it.

#### Understand
- [ ] Explain *why* the chosen detector and optimizations suit densely packed retail shelves.
- [ ] Defend dataset split, subset sampling, metrics, model selection and benchmarking fairness.
- [ ] Distinguish measured CPU/GPU/cloud figures from genuine on-device benchmarks.
- [ ] Clearly state the biggest project limitation and the next controlled experiment.

#### Build / practice
- [ ] Prepare a 60-second résumé pitch and a 2-minute project deep dive.
- [ ] Rehearse drawing the end-to-end architecture without reading source code.
- [ ] Conduct at least two mock technical interviews before January 4.
- [ ] Verify the final official interview schedule and meeting information.

**Checkpoint:** Can discuss every claim on your résumé and explain both successful and failed experiments without guessing.

---

## Experiment ledger — fill in only after actual runs

| Experiment ID | Model | Training data | Eval split | Resolution | Hardware/runtime | mAP@0.5:0.95 | Counting MAE | p50 / p95 latency | Model size | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp-001` | TBD | TBD | TBD | TBD | TBD | — | — | — | — | Not run |
| `exp-002` | TBD | TBD | TBD | TBD | TBD | — | — | — | — | Not run |

Always include **units** (ms, MB), repeat counts, benchmark method, thresholds and evaluation population when recording results. Do not put a number into the table without having generated it.

## Current work session

- **Last completed phase:** None confirmed.
- **Next action:** Phase 0 — initialize the repository, verify Python/PyTorch, and create a passing test command.
- **Current blocker:** None recorded.
- **Next specific question for ChatGPT:** “Using `projec-context.md` and `Plan-checklist.md`, guide me through the next unchecked P0 task with exact Ubuntu commands. Keep the November 8 résumé release and the parallel entrance-test preparation in mind.”
