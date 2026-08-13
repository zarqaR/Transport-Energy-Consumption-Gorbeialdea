"""
build_link_table.py
────────────────────────────────────────────────────────────────────────────────
Builds a directed link table from a road network.

Inputs  : road GeoPackage (directed LineStrings) + node GeoPackage (Points)
Outputs : link_table.csv   — tabular link table (no geometry)
          link_table.gpkg  — spatial link table (merged LineString per link)

Source CRS  : EPSG:4258  (ETRS89)
Working CRS : EPSG:25830 (ETRS89 / UTM zone 30N)  — used for metric lengths

NOTE: Nodes are allowed to sit at ANY coordinate along a road segment —
      including interior vertices. The pre-processing step below splits
      each segment at every node coordinate it contains before building
      the graph.
────────────────────────────────────────────────────────────────────────────────
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import transform
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# [1]  PATHS  ←  REPLACE WITH YOUR ACTUAL FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────

ROAD_SHP   = "Gorbeialdea Main roads Final.gpkg"   # ← road GeoPackage
NODE_SHP   = "Final Node Table Gorbeialdea.gpkg"   # ← node GeoPackage

OUT_CSV    = "link_table.csv"
OUT_GPKG   = "link_table.gpkg"
GPKG_LAYER = "links"

# ──────────────────────────────────────────────────────────────────────────────
# [2]  COLUMN NAMES  ←  REPLACE WITH YOUR ACTUAL FIELD NAMES
# ──────────────────────────────────────────────────────────────────────────────

NODE_ID_COL    = "node_id"       # ← unique node identifier
IS_COUNT_COL   = "is_count"      # ← 1 if count station, else 0
STATION_ID_COL = "station_id"    # ← populated only for count station nodes
IS_ZONE_COL    = "is_zone"       # ← 1 if zone centroid, else 0
ZONE_TYPE_COL  = "zone_type"     # ← zone type (if applicable)

# ──────────────────────────────────────────────────────────────────────────────
# [3]  ROAD DIRECTIONALITY FIELD
#      Field in the road layer that indicates two-way roads.
#      Links where ANY underlying segment has this value get two_way = 1.
# ──────────────────────────────────────────────────────────────────────────────

SENTIDO_COL    = "sentidod"      # ← field name for directionality
SENTIDO_TWOWAY = "Doble"         # ← value that means two-way

# ──────────────────────────────────────────────────────────────────────────────
# [4]  COORDINATE SNAPPING TOLERANCE (metres)
# ──────────────────────────────────────────────────────────────────────────────

SNAP_TOLERANCE = 0.01   # metres

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD & REPROJECT
# ══════════════════════════════════════════════════════════════════════════════

print("Loading shapefiles …")
roads = gpd.read_file(ROAD_SHP, layer=0)
nodes = gpd.read_file(NODE_SHP, layer=0)

print(f"  Roads : {len(roads):,} segments  |  CRS: {roads.crs}")
print(f"  Nodes : {len(nodes):,} points    |  CRS: {nodes.crs}")

# Strip Z coordinates
def drop_z(geom):
    return transform(lambda x, y, *z: (x, y), geom)

roads.geometry = roads.geometry.apply(drop_z)
nodes.geometry = nodes.geometry.apply(drop_z)

TARGET_CRS = "EPSG:25830"
roads = roads.to_crs(TARGET_CRS)
nodes = nodes.to_crs(TARGET_CRS)

print(f"  Reprojected to {TARGET_CRS}.")

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD NODE LOOK-UPS
# ══════════════════════════════════════════════════════════════════════════════

COORD_ROUND = 3

def rounded_coord(geom):
    return (round(geom.x, COORD_ROUND), round(geom.y, COORD_ROUND))

nodes["_coord"] = nodes.geometry.apply(rounded_coord)

if nodes["_coord"].duplicated().any():
    dupes = nodes[nodes["_coord"].duplicated(keep=False)]
    raise ValueError(
        f"Duplicate node coordinates detected ({len(dupes)} rows). "
        "Fix your node layer before continuing.\n" + str(dupes[[NODE_ID_COL, "_coord"]])
    )

coord_to_node = dict(zip(nodes["_coord"], nodes[NODE_ID_COL]))
node_to_attrs = nodes.set_index(NODE_ID_COL)[[IS_COUNT_COL, STATION_ID_COL]].to_dict("index")

# ══════════════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING: SPLIT ROAD SEGMENTS AT INTERIOR NODE COORDINATES
# ══════════════════════════════════════════════════════════════════════════════

print("Pre-processing: splitting segments at interior node coordinates …")

def rc(x, y):
    return (round(x, COORD_ROUND), round(y, COORD_ROUND))

def coord_matches_node(x, y):
    key = rc(x, y)
    if key in coord_to_node:
        return True
    for nc in coord_to_node:
        if abs(nc[0] - x) <= SNAP_TOLERANCE and abs(nc[1] - y) <= SNAP_TOLERANCE:
            return True
    return False

def snap_coord(x, y):
    key = rc(x, y)
    if key in coord_to_node:
        return key
    for nc in coord_to_node:
        if abs(nc[0] - x) <= SNAP_TOLERANCE and abs(nc[1] - y) <= SNAP_TOLERANCE:
            return nc
    return key

def split_segment_at_nodes(coords):
    current_piece = [coords[0]]
    for pt in coords[1:]:
        current_piece.append(pt)
        if len(current_piece) >= 2 and coord_matches_node(pt[0], pt[1]):
            yield current_piece
            current_piece = [pt]
    if len(current_piece) >= 2:
        yield current_piece

skipped_segments = 0
split_sub_segments = []   # list of (start_rc, end_rc, LineString, sentidod)

for _, row in roads.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        skipped_segments += 1
        continue

    # Get sentidod value for this road segment
    sentidod_val = row.get(SENTIDO_COL, None)

    # Handle both LineString and MultiLineString
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
    else:
        parts = [geom]

    for part in parts:
        coords = list(part.coords)
        if len(coords) < 2:
            skipped_segments += 1
            continue

        for piece_coords in split_segment_at_nodes(coords):
            if len(piece_coords) < 2:
                continue
            start_rc = snap_coord(piece_coords[0][0],  piece_coords[0][1])
            end_rc   = snap_coord(piece_coords[-1][0], piece_coords[-1][1])
            line     = LineString(piece_coords)
            split_sub_segments.append((start_rc, end_rc, line, sentidod_val))

if skipped_segments:
    print(f"  ⚠  Skipped {skipped_segments} null/empty/degenerate road geometries.")

original_count = len(roads) - skipped_segments
print(f"  {original_count:,} original segments → {len(split_sub_segments):,} sub-segments after splitting.")

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD DIRECTED GRAPH WITH NETWORKX
# ══════════════════════════════════════════════════════════════════════════════

print("Building directed graph …")

G = nx.DiGraph()

for start_rc, end_rc, line, sentidod_val in split_sub_segments:
    G.add_edge(
        start_rc, end_rc,
        length=line.length,
        geom=line,
        sentidod=sentidod_val,
    )

print(f"  Graph: {G.number_of_nodes():,} graph-nodes, {G.number_of_edges():,} edges.")

# ══════════════════════════════════════════════════════════════════════════════
#  TRACE LINKS  (from each network node, follow until the next network node)
# ══════════════════════════════════════════════════════════════════════════════

print("Tracing links between consecutive nodes …")

node_coords = set(nodes["_coord"].tolist())

def is_network_node(graph_node):
    return graph_node in node_coords

links = []
link_id = 1
visited_start_edges = set()

for _, node_row in nodes.iterrows():
    from_nid  = node_row[NODE_ID_COL]
    start_pt  = node_row["_coord"]

    if start_pt not in G:
        print(f"  ⚠  Node {from_nid} at {start_pt} has no outgoing road segments.")
        continue

    for nbr in G.successors(start_pt):
        edge_key = (start_pt, nbr)
        if edge_key in visited_start_edges:
            continue

        current        = nbr
        prev           = start_pt
        total_length   = G[start_pt][nbr]["length"]
        geom_parts     = [G[start_pt][nbr]["geom"]]
        sentidod_parts = [G[start_pt][nbr].get("sentidod", None)]
        path_edges     = [edge_key]
        found_end_node = None
        max_steps      = 100_000

        steps = 0
        while steps < max_steps:
            steps += 1
            if is_network_node(current):
                found_end_node = coord_to_node[current]
                break
            successors = list(G.successors(current))
            successors = [s for s in successors if s != prev]
            if len(successors) == 0:
                print(f"  ⚠  Dead end reached before finding next node "
                      f"(from node {from_nid}, step {steps}). Skipping link.")
                break
            if len(successors) > 1:
                print(f"  ⚠  Unexpected branch at non-node coord {current} "
                      f"(from node {from_nid}). Check that all branch points have nodes.")
                break
            nxt = successors[0]
            edge_data = G[current][nxt]
            total_length += edge_data["length"]
            geom_parts.append(edge_data["geom"])
            sentidod_parts.append(edge_data.get("sentidod", None))
            path_edges.append((current, nxt))
            prev    = current
            current = nxt
        else:
            print(f"  ⚠  Max steps reached tracing from node {from_nid}. Skipping link.")
            found_end_node = None

        if found_end_node is None:
            continue

        for e in path_edges:
            visited_start_edges.add(e)

        # Merge segment geometries into a single LineString
        if len(geom_parts) == 1:
            merged_geom = geom_parts[0]
        else:
            all_coords = []
            for g in geom_parts:
                c = list(g.coords)
                if all_coords and all_coords[-1] == c[0]:
                    all_coords.extend(c[1:])
                else:
                    all_coords.extend(c)
            merged_geom = LineString(all_coords)

        # Resolve count station IDs
        from_attrs = node_to_attrs.get(from_nid, {})
        to_attrs   = node_to_attrs.get(found_end_node, {})

        from_station = (
            from_attrs.get(STATION_ID_COL)
            if from_attrs.get(IS_COUNT_COL) == 1 else None
        )
        to_station = (
            to_attrs.get(STATION_ID_COL)
            if to_attrs.get(IS_COUNT_COL) == 1 else None
        )

        # two_way = 1 if any underlying segment has sentidod = "Doble"
        two_way = 1 if any(s == SENTIDO_TWOWAY for s in sentidod_parts) else 0

        links.append({
            "link_id":         link_id,
            "from_node_id":    from_nid,
            "to_node_id":      found_end_node,
            "length_km":       round(total_length / 1000, 6),
            "from_station_id": from_station,
            "to_station_id":   to_station,
            "count":           None,
            "two_way":         two_way,
            "geometry":        merged_geom,
        })
        link_id += 1

print(f"  → {len(links):,} links traced.")

# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

gdf_links = gpd.GeoDataFrame(links, crs=TARGET_CRS)

# ── CSV (no geometry) ──────────────────────────────────────────────────────
csv_cols = ["link_id", "from_node_id", "to_node_id",
            "length_km", "from_station_id", "to_station_id", "count", "two_way"]
gdf_links[csv_cols].to_csv(OUT_CSV, index=False)
print(f"\n  CSV  saved → {OUT_CSV}")

# ── GeoPackage (with geometry) ─────────────────────────────────────────────
gpkg_cols = csv_cols + ["geometry"]
gdf_links[gpkg_cols].to_file(OUT_GPKG, layer=GPKG_LAYER, driver="GPKG")
print(f"  GPKG saved → {OUT_GPKG}  (layer: '{GPKG_LAYER}')")

# ══════════════════════════════════════════════════════════════════════════════
#  SANITY CHECK SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Summary ─────────────────────────────────────────────────────────────")
print(f"  Total links           : {len(gdf_links):,}")
print(f"  Links with from-count : {gdf_links['from_station_id'].notna().sum():,}")
print(f"  Links with to-count   : {gdf_links['to_station_id'].notna().sum():,}")
print(f"  Two-way links         : {gdf_links['two_way'].sum():,}")
print(f"  One-way links         : {(gdf_links['two_way'] == 0).sum():,}")
print(f"  Min length (km)       : {gdf_links['length_km'].min():.4f}")
print(f"  Max length (km)       : {gdf_links['length_km'].max():.4f}")
print(f"  Total network (km)    : {gdf_links['length_km'].sum():.2f}")
print("────────────────────────────────────────────────────────────────────────")
print("Done.")
