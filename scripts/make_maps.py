"""Generate Folium maps for selected trips into reports/maps/.

  python scripts/make_maps.py                      -> default trips below
  python scripts/make_maps.py train/00000001 ...   -> any <partition>/<trip_id>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import DATA  # noqa: E402
from src.maps import make_route_map  # noqa: E402

DEFAULT = [
    "train/00000054",         # 89% uncertain labels
    "train/00000090",         # largest GPS gap (~42 h)
    "student_test/00000065",  # student_test example
]


def main():
    out = ROOT / "reports" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:] or DEFAULT:
        trip_dir = DATA / spec
        path = out / f"trip_{trip_dir.name}.html"
        path.write_text(make_route_map(trip_dir), encoding="utf-8")
        print(f"saved {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
