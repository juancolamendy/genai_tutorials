from pipeline_example.backchannel import BackchannelProcessor


def test_backchannel_respects_min_interval():
    bp = BackchannelProcessor(min_interval_seconds=5, chance_factor=1.0)
    assert bp.can_backchannel()
    bp.record_backchannel()
    assert not bp.can_backchannel()


def test_backchannel_respects_per_minute_cap():
    bp = BackchannelProcessor(min_interval_seconds=0, max_frequency_per_minute=2, chance_factor=1.0)
    assert bp.can_backchannel()
    bp.record_backchannel()
    assert bp.can_backchannel()
    bp.record_backchannel()
    assert not bp.can_backchannel()
