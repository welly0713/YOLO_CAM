from app.detection import AlertGate, DetectionWindow


def test_two_thirds_of_fifteen_samples_qualifies():
    window = DetectionWindow(window_size=15, minimum_ratio=2 / 3)

    for _ in range(9):
        stats = window.add(False)
    assert not stats.ready

    stats = window.add(True)
    assert not stats.ready
    assert stats.positive_samples == 1
    assert not stats.qualifies

    for _ in range(9):
        stats = window.add(True)

    assert stats.positive_samples == 10
    assert stats.required_samples == 10
    assert stats.qualifies


def test_alert_gate_alerts_once_until_window_clears():
    window = DetectionWindow(window_size=3, minimum_ratio=2 / 3)
    gate = AlertGate(cooldown_seconds=300)

    stats = window.add(True)
    stats = window.add(True)
    stats = window.add(False)
    assert gate.should_alert(stats, now=0) is True
    assert gate.should_alert(stats, now=1) is False

    # A non-qualifying window clears the event latch.
    stats = window.add(False)
    assert not stats.qualifies
    assert gate.should_alert(stats, now=2) is False

    # The same event cannot alert again during the configured cooldown.
    stats = window.add(True)
    stats = window.add(True)
    assert stats.qualifies
    assert gate.should_alert(stats, now=3) is False
