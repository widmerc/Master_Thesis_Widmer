# routing_helpers.py
import itertools as it
import numpy as np
import networkx as nx
import geopandas as gpd
import math

from shapely.geometry import LineString, Point
from shapely.ops import linemerge
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    # Fallback: provide a minimal KD-tree wrapper if scipy is not available.
    # This is slower and only used in environments missing scipy during static checks.
    class cKDTree:
        def __init__(self, data):
            self.data = list(data)

        def query(self, pt):
            # naive nearest neighbor
            best = None
            bestd = float('inf')
            for i, p in enumerate(self.data):
                d = (p[0]-pt[0])**2 + (p[1]-pt[1])**2
                if d < bestd:
                    bestd = d
                    best = i
            if best is None:
                # empty tree
                return float('inf'), 0
            return math.sqrt(bestd), best
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _endpoints(line: LineString):
    """Runde Endpunkte leicht, damit stabile Node-Keys entstehen."""
    a, b = line.coords[0], line.coords[-1]
    return (round(a[0], 6), round(a[1], 6)), (round(b[0], 6), round(b[1], 6))


def _edge_cost(length, safety, alpha=1.0, beta=1.0):
    """
    Robuste Kantenkosten:
      - safety in [0, 100] clampen, NaN -> 50 (neutral)
      - keine negativen Gewichte (für Dijkstra/Yen/… nötig)
      - Formel: alpha*L + beta*(100 - S)*(L/100)
    """
    length = float(length)
    # Abwehr: negative/NaN-Längen
    if not np.isfinite(length) or length < 0:
        length = 0.0

    # Safety säubern
    if safety is None or (isinstance(safety, float) and np.isnan(safety)):
        safety = 50.0
    safety = max(0.0, min(100.0, float(safety)))

    cost = alpha * length + beta * (100.0 - safety) * (length / 100.0)
    return max(cost, 0.0)


# -------------------------------------------------------
# Graph-Aufbau (parallel edges kollabieren)
# -------------------------------------------------------
def build_graph_simple(gdf: gpd.GeoDataFrame, alpha=1.0, beta=1.0) -> nx.Graph:
    """
    Undirektionaler Graph:
      - Knoten = Segmentendpunkte
      - Kanten = Liniensegmente
      - Parallelkanten werden durch min(cost) ersetzt
        Tie-Breaker: höhere safety_score bevorzugen
    Erwartete Spalten: geometry(LineString), length_m, safety_score
    """
    need = [c for c in ("geometry", "length_m", "safety_score") if c not in gdf.columns]
    if need:
        raise ValueError(f"Missing columns for routing: {need}")

    G = nx.Graph()

    for _, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.is_empty:
            continue

        u, v = _endpoints(geom)
        length = float(row["length_m"])
        safety = float(row["safety_score"]) if row["safety_score"] is not None else np.nan
        cost = _edge_cost(length, safety, alpha, beta)

        if u not in G:
            G.add_node(u, x=u[0], y=u[1])
        if v not in G:
            G.add_node(v, x=v[0], y=v[1])

        # saubere (geclampte) Safety auch als Attribut ablegen
        safety_clean = 50.0 if (isinstance(safety, float) and np.isnan(safety)) else max(0.0, min(100.0, float(safety)))

        attrs = dict(
            fid=row.get("fid"),
            length_m=length,
            safety_score=safety_clean,
            cost=cost,
            geom=geom
        )

        if G.has_edge(u, v):
            curr = G[u][v]
            # besser = geringere Kosten
            better = (cost < curr["cost"]) or (
                # bei Kostengleichheit: höhere Sicherheit bevorzugen
                np.isclose(cost, curr["cost"]) and safety_clean > curr["safety_score"]
            )
            if better:
                G[u][v].update(attrs)
        else:
            G.add_edge(u, v, **attrs)

    print(f"[Graph built] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (alpha={alpha}, beta={beta})")
    return G


# -------------------------------------------------------
# NodeLocator (Nearest Node)
# -------------------------------------------------------
class NodeLocator:
    def __init__(self, G: nx.Graph):
        coords = np.array([(n[0], n[1]) for n in G.nodes()])
        self.nodes = list(G.nodes())
        self.tree = cKDTree(coords)

    def nearest(self, pt: Point):
        _, idx = self.tree.query([pt.x, pt.y])
        try:
            if idx is None:
                raise ValueError("KD-tree returned no index")
            return self.nodes[int(idx)]
        except Exception as e:
            # fallback: brute-force nearest
            coords = np.array(self.nodes)
            d2 = np.sum((coords - np.array([pt.x, pt.y])) ** 2, axis=1)
            i = int(np.argmin(d2))
            return self.nodes[i]


# -------------------------------------------------------
# kompaktes Edge-Map (für Parallel-Summary)
# -------------------------------------------------------
def _make_edge_map(G: nx.Graph):
    def key(u, v): return (u, v) if u <= v else (v, u)
    emap = {}
    for u, v, data in G.edges(data=True):
        emap[key(u, v)] = {
            "length_m": float(data["length_m"]),
            "cost": float(data["cost"]),
            "safety_score": float(data["safety_score"]),
            "fid": data.get("fid"),
            "geom": data.get("geom"),
        }
    return emap


# -------------------------------------------------------
# Pfad zusammenfassen
# -------------------------------------------------------
def _summarize_path_from_map(path, emap):
    def key(u, v): return (u, v) if u <= v else (v, u)
    total_len = 0.0
    total_cost = 0.0
    min_safety = float("inf")
    worst_edge = None
    edges = []

    for u, v in zip(path[:-1], path[1:]):
        d = emap[key(u, v)]
        edges.append(d)
        total_len += d["length_m"]
        total_cost += d["cost"]
        if d["safety_score"] < min_safety:
            min_safety = d["safety_score"]
            worst_edge = d.get("fid")

    lw_mean = (sum(e["safety_score"] * e["length_m"] for e in edges) /
               max(total_len, 1e-9))

    return dict(
        nodes=path,
        edges=edges,
        total_length_m=total_len,
        total_cost=total_cost,
        safety_mean_lenweighted=lw_mean,
        safety_min_edge=min_safety,
        worst_edge_fid=worst_edge
    )


# -------------------------------------------------------
# k Alternativen (nx.shortest_simple_paths + Parallel-Summary)
# -------------------------------------------------------
def k_routes_mp(G, src_pt: Point, dst_pt: Point, k=1):
    locator = NodeLocator(G)
    s, t = locator.nearest(src_pt), locator.nearest(dst_pt)
    gen = nx.shortest_simple_paths(G, s, t, weight="cost")
    path = next(gen)  # nur der beste Pfad
    emap = _make_edge_map(G)
    return [_summarize_path_from_map(path, emap)]
