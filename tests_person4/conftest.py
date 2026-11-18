from pathlib import Path
import pytest

from pathsnap_person4.network import TripNetwork


@pytest.fixture
def tiny_trip(tmp_path):
    folder = tmp_path / "12345678"
    folder.mkdir()
    (folder / "12345678.nodes").write_text(
        "0.0000 0.0000\n0.0001 0.0000\n0.0002 0.0000\n0.0003 0.0000\n",
        encoding="utf-8",
    )
    (folder / "12345678.arcs").write_text("0 1\n1 2\n2 3\n", encoding="utf-8")
    (folder / "12345678.track").write_text(
        "0.00005 0.00000 1\n0.00015 0.00000 2\n0.00025 0.00000 3\n",
        encoding="utf-8",
    )
    (folder / "12345678.route").write_text("0\n1\n2\n", encoding="utf-8")
    return folder


@pytest.fixture
def tiny_network(tiny_trip):
    return TripNetwork.from_directory(tiny_trip)
