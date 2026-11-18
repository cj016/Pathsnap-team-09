# PathSnap TEAM-9

PathSnap matches GPS observations to directed road segments. Laptop 1 builds six features for each road within 75 metres, scores the candidates with LightGBM, and decodes a route with Viterbi. Laptop 2 lists trips and displays GPS points, the decoded route, accuracy when references are available, continuity, and inference time.

