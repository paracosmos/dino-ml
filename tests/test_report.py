from src.ml.sl.report import class_distribution, confusion_matrix


def test_class_distribution_counts_and_ratios():
    d = class_distribution([0, 0, 1, 2, 2, 2], n_classes=3)
    assert d["total"] == 6
    assert d["counts"] == [2, 1, 3]
    assert abs(d["ratios"][2] - 0.5) < 1e-9


def test_class_distribution_empty():
    d = class_distribution([], n_classes=3)
    assert d["total"] == 0
    assert d["ratios"] == [0.0, 0.0, 0.0]


def test_confusion_matrix():
    cm = confusion_matrix([0, 1, 2, 2], [0, 1, 2, 1], n_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1 and cm[2, 1] == 1
