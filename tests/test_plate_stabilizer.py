import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.plate_stabilizer import PlateStabilizer, calculate_box_iou

def test_box_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 0, 15, 10]
    # intersection: 5x10 = 50, union: 100 + 100 - 50 = 150 -> 50/150 = 0.333
    iou = calculate_box_iou(box1, box2)
    assert abs(iou - 1.0 / 3.0) < 1e-4

def test_stabilizer_voting():
    stabilizer = PlateStabilizer(vote_window=5, min_votes=3, plate_cooldown=3.0)
    seq = ["43A12345", "43A12345", "43A12346", "43A12345", "43A12345"]
    bbox = [100, 100, 250, 180]

    outputs = []
    for idx, text in enumerate(seq, 1):
        res = stabilizer.update(
            [{"bbox": bbox, "confidence": 0.92, "plate_text": text}],
            current_time=idx * 0.1
        )
        assert len(res) == 1
        outputs.append(res[0])

    # Check frame 1: 1 vote -> not stable yet
    assert outputs[0]["is_stable"] is False
    assert outputs[0]["votes"] == 1
    assert outputs[0]["plate_text"] == "43A12345"

    # Check frame 2: 2 votes -> not stable yet
    assert outputs[1]["is_stable"] is False
    assert outputs[1]["votes"] == 2

    # Check frame 3: jitter "43A12346" -> still 2 votes for "43A12345"
    assert outputs[2]["is_stable"] is False

    # Check frame 4: 3 votes for "43A12345" -> becomes STABLE!
    assert outputs[3]["is_stable"] is True
    assert outputs[3]["plate_text"] == "43A12345"
    assert outputs[3]["votes"] == 3
    assert outputs[3]["is_new_stable_event"] is True

    # Check frame 5: 4 votes for "43A12345" -> remains STABLE, but no duplicate notification
    assert outputs[4]["is_stable"] is True
    assert outputs[4]["plate_text"] == "43A12345"
    assert outputs[4]["votes"] == 4
    assert outputs[4]["is_new_stable_event"] is False

    print("All stabilizer tests passed successfully!")

if __name__ == "__main__":
    test_box_iou()
    test_stabilizer_voting()
