import math
import pandas as pd

from pathsnap_person4.decoder import DecoderConfig, decode


def test_viterbi_repairs_unreachable_independent_prediction(tiny_network):
    # At point 0, classifier-only incorrectly prefers downstream arc 2.
    # Directed motion from arc 2 back to arc 1 is impossible in this graph.
    table = pd.DataFrame([
        {"point_idx": 0, "arc_id": 0, "positive_prob": .40},
        {"point_idx": 0, "arc_id": 2, "positive_prob": .90},
        {"point_idx": 1, "arc_id": 1, "positive_prob": .95},
        {"point_idx": 2, "arc_id": 2, "positive_prob": .95},
    ])
    result = decode(table, tiny_network)
    assert result.classifier_arc_ids == [2, 1, 2]
    assert result.hmm_arc_ids == [0, 1, 2]
    assert result.diagnostics == {"no_candidates": 0, "gps_gaps": 0, "unreachable_restarts": 0}
    lines = tiny_network.route_lines(result.hmm_arc_ids, result.breaks)
    assert len(lines) == 1
    assert len(lines[0]) == 4


def test_missing_candidate_returns_none_not_fabricated(tiny_network):
    table = pd.DataFrame([
        {"point_idx": 0, "arc_id": 0, "positive_prob": .8},
        {"point_idx": 2, "arc_id": 2, "positive_prob": .8},
    ])
    result = decode(table, tiny_network)
    assert result.hmm_arc_ids == [0, None, 2]
    assert result.diagnostics["no_candidates"] == 1
    assert len(tiny_network.route_lines(result.hmm_arc_ids, result.breaks)) == 2


def test_gps_gap_starts_new_segment(tiny_trip):
    track_path = tiny_trip / "12345678.track"
    track_path.write_text(
        "0.00005 0.00000 1\n0.00015 0.00000 2\n0.00025 0.00000 100\n",
        encoding="utf-8",
    )
    from pathsnap_person4.network import TripNetwork
    network = TripNetwork.from_directory(tiny_trip)
    table = pd.DataFrame([
        {"point_idx": i, "arc_id": i, "positive_prob": .8} for i in range(3)
    ])
    result = decode(table, network, DecoderConfig(gap_seconds=15))
    assert result.hmm_arc_ids == [0, 1, 2]
    assert result.diagnostics["gps_gaps"] == 1
    assert 2 in result.breaks


def test_directed_not_reverse(tiny_network):
    assert math.isfinite(tiny_network.distance_between(0, 0.5, 1, 0.5, 100))
    assert math.isinf(tiny_network.distance_between(2, 0.5, 1, 0.5, 100))
