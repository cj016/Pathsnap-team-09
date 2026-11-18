"""PathSnap Folium maps: GPS points, nearby road network, reference route and (optional) predicted route."""
from pathlib import Path

import folium
import numpy as np
import shapely

from src.data_prep import arc_lines, label_points, load_trip, project_trip

MAX_POINTS = 3000  # draw at most this many GPS markers so the HTML stays small


def _arc_coords(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] segments for the given arc ids."""
    latlon = trip["nodes"][["lat", "lon"]].to_numpy()
    arcs = trip["arcs"][["from_node", "to_node"]].to_numpy()
    return latlon[arcs[np.asarray(arc_ids, dtype=int)]].tolist()


def make_route_map(trip_dir, predicted_arcs=None, reference_arcs=None, return_html=True, network_buffer_m=150):
    """Folium map of one trip.

    Layers: road network within `network_buffer_m` of the track (grey), reference route (blue),
    predicted route (green, only if `predicted_arcs` is given), GPS points (orange) with
    uncertain-label points (> 50 m from their label arc) in red.
    `reference_arcs` defaults to the trip's .route file.
    Returns the full HTML string if `return_html` is True, otherwise the folium.Map object.
    """
    trip_dir = Path(trip_dir)
    trip = project_trip(load_trip(trip_dir))
    if reference_arcs is not None:
        trip["route"] = np.asarray(reference_arcs, dtype=int)
    lines = arc_lines(trip)
    trip = label_points(trip, lines)  # gives label_uncertain per point
    t = trip["track"]

    track_line = shapely.linestrings(t[["x", "y"]].to_numpy())
    near = shapely.STRtree(lines).query(track_line, predicate="dwithin", distance=network_buffer_m)

    m = folium.Map(tiles="OpenStreetMap")
    layers = {
        "net": folium.FeatureGroup(name=f"Road network (within {network_buffer_m} m)"),
        "ref": folium.FeatureGroup(name="Reference route"),
        "pred": folium.FeatureGroup(name="Predicted route"),
        "gps": folium.FeatureGroup(name="GPS points"),
        "unc": folium.FeatureGroup(name="Uncertain points (> 50 m)"),
    }
    # one multi-line per layer keeps the HTML small
    folium.PolyLine(_arc_coords(trip, near), color="gray", weight=1, opacity=0.7).add_to(layers["net"])
    folium.PolyLine(_arc_coords(trip, trip["route"]), color="blue", weight=4, opacity=0.7).add_to(layers["ref"])
    if predicted_arcs is not None and len(predicted_arcs):
        folium.PolyLine(_arc_coords(trip, predicted_arcs), color="green", weight=3, dash_array="6").add_to(layers["pred"])

    step = max(1, int(np.ceil(len(t) / MAX_POINTS)))
    for row in t.iloc[::step].itertuples():
        unc = bool(row.label_uncertain)
        folium.CircleMarker([row.lat, row.lon], radius=2, color="red" if unc else "orange", fill=True,
                            tooltip=f"point {row.Index}, t={row.timestamp:.0f}s, {row.label_dist:.0f} m to label arc"
                            ).add_to(layers["unc" if unc else "gps"])
    for key, fg in layers.items():
        if key != "pred" or predicted_arcs is not None:
            fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[t["lat"].min(), t["lon"].min()], [t["lat"].max(), t["lon"].max()]])  # auto-fit zoom

    pred_row = '<span style="color:green">&#9644;</span> predicted route<br>' if predicted_arcs is not None else ""
    sampled = f"<br><i>every {step}th point shown</i>" if step > 1 else ""
    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;'
        'border:1px solid grey;font-size:13px">'
        f"<b>Trip {trip_dir.name}</b> ({len(t)} GPS points, {t['label_uncertain'].mean():.1%} uncertain)<br>"
        '<span style="color:gray">&#9644;</span> road network<br>'
        '<span style="color:blue">&#9644;</span> reference route<br>' + pred_row +
        '<span style="color:orange">&#9679;</span> GPS point<br>'
        f'<span style="color:red">&#9679;</span> uncertain label (&gt; 50 m){sampled}</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    return m.get_root().render() if return_html else m
