from src.ml.dino.action_spec import DinoAction


def test_action_values():
    assert (int(DinoAction.NOOP), int(DinoAction.JUMP), int(DinoAction.DUCK)) == (0, 1, 2)


def test_action_count():
    assert len(DinoAction) == 3
