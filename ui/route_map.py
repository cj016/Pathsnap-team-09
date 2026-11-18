import folium
from folium.plugins import FastMarkerCluster


def route_map(result):
    points = [[p["latitude"], p["longitude"]] for p in result["observations"]]
    if not points:
        raise ValueError("Missing GPS observations")
    view = folium.Map(location=points[0], zoom_start=15, prefer_canvas=True)
    FastMarkerCluster(points, name="GPS points").add_to(view)
    route = folium.FeatureGroup(name="Predicted route").add_to(view)
    bounds = list(points)
    for line in result["predicted_route_lines"]:
        coords = [[lat, lon] for lon, lat in line]
        if len(coords) >= 2:
            folium.PolyLine(coords, color="#176442", weight=5).add_to(route)
            bounds.extend(coords)
    view.fit_bounds(
        [
            [min(p[0] for p in bounds), min(p[1] for p in bounds)],
            [max(p[0] for p in bounds), max(p[1] for p in bounds)],
        ],
        max_zoom=17,
    )
    folium.LayerControl(collapsed=False).add_to(view)
    return view.get_root().render()
