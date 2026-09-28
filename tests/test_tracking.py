import numpy as np

from collision_avoidance.tracking import Detection, Tracker, iou

NO_FLOW = np.zeros((384, 512, 2), np.float32)


def det(x1, y1, x2, y2, name="car"):
    return Detection(np.array([x1, y1, x2, y2]), name, 0.9)


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.999
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_track_keeps_identity_while_moving():
    tracker = Tracker()
    for step in range(10):
        tracks = tracker.update([det(100 + 3 * step, 100, 160 + 3 * step, 150)], NO_FLOW)
    assert list(tracks) == [0]
    assert tracks[0].age == 10


def test_new_object_gets_new_track_and_lost_tracks_expire():
    tracker = Tracker(max_lost=2)
    tracker.update([det(10, 10, 50, 50)], NO_FLOW)
    tracker.update([det(300, 200, 350, 260, "person")], NO_FLOW)
    assert set(tracker.tracks) == {0, 1}
    for _ in range(3):
        tracker.update([det(300, 200, 350, 260, "person")], NO_FLOW)
    assert set(tracker.tracks) == {1}
