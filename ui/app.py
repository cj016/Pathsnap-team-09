import os
from math import isfinite
from urllib.parse import quote

import requests
from flask import Flask, render_template, request

app = Flask(__name__)
API = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")


def metric(value, *, percent=False):
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        return "—"
    if percent:
        return f"{value:.2%}" if value <= 1 else "—"
    return f"{value:,.0f} ms"


@app.route("/", methods=["GET", "POST"])
def index():
    trips, selected, route, scores, error = [], "", "", [], ""
    try:
        response = requests.get(f"{API}/trips", timeout=(5, 10))
        response.raise_for_status()
        trips = response.json()["trip_ids"]
        if not isinstance(trips, list) or not all(isinstance(t, str) and t for t in trips):
            trips = []
            raise ValueError("Invalid trip list")

        selected = request.form.get("trip", trips[0] if trips else "")
        if request.method == "POST":
            if selected not in trips:
                raise ValueError("Unknown trip")

            response = requests.post(
                f"{API}/predict/{quote(selected, safe='')}", timeout=(5, 180)
            )
            response.raise_for_status()
            result = response.json()

            if result["trip_id"] != selected:
                raise ValueError("Trip ID mismatch")
            if not isinstance(result["map_html"], str) or not result["map_html"].strip():
                raise ValueError("Missing map")

            m = result["metrics"]
            scores = [
                ("Classifier Top-1", metric(m["top1_accuracy"], percent=True)),
                ("Decoded accuracy", metric(m["decoded_point_accuracy"], percent=True)),
                ("Inference time", metric(m["inference_time_ms"])),
            ]
            route = result["map_html"]

    except requests.Timeout:
        error = "The API timed out."
    except requests.RequestException:
        error = "The PathSnap API request failed."
    except (KeyError, TypeError, ValueError):
        error = "The trip selection or API response was invalid."

    if error:
        app.logger.warning("Trip %r: %s", selected, error)

    return render_template(
        "index.html", trips=trips, selected=selected,
        route=route, scores=scores, error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501)
