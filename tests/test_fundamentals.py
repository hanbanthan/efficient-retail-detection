import numpy as np
import pytest
from src.fundamentals.boxes import *
from src.fundamentals.iou import iou, iou_matrix
from src.fundamentals.nms import nms


def test_conversion_roundtrips():
    b = np.array([[10, 20, 50, 80], [0, 0, 5, 5]], dtype=float)
    assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(b)), b)
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)
    assert np.allclose(xyxy_to_xywh([10, 20, 50, 80]), [10, 20, 40, 60])


def test_iou_identical():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_no_overlap_and_touching_edge():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0   # shared edge only


def test_iou_partial():
    assert iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_iou_degenerate_boxes_no_nan():
    assert iou([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0         # zero area
    assert iou([2, 2, 0, 0], [0, 0, 2, 2]) == 0.0         # inverted box
    assert not np.isnan(iou_matrix([[1, 1, 1, 1]], [[1, 1, 1, 1]])).any()


def test_nms_sorts_by_score_and_suppresses_duplicate():
    boxes = [[1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10]]
    scores = [0.8, 0.7, 0.9]
    assert nms(boxes, scores, 0.5) == [2, 1]   # score order; box 0 suppressed by box 2


def test_nms_threshold_and_empty():
    boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]   # IoU = 1/3
    assert nms(boxes, [0.9, 0.8], 0.5) == [0, 1]
    assert nms(boxes, [0.9, 0.8], 0.3) == [0]
    assert nms(np.zeros((0, 4)), []) == []


def test_matches_torchvision():
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision")
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (200, 2)); wh = rng.uniform(5, 40, (200, 2))
    boxes = np.hstack([xy, xy + wh]); scores = rng.uniform(size=200)
    ref = tv.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
    assert nms(boxes, scores, 0.5) == ref