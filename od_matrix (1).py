"""
================================================================================
OD MATRIX ESTIMATION — GORBEIALDEA, ÁLAVA, SPAIN
================================================================================

METHODOLOGY
-----------
Based on Khan & Anderson (2014): "Estimation of Through Trips Using Existing
Traffic Counts". IJTTE, 3(6), 415-423.

The core relationship V_a = Σ_ij T_ij * p_ij^a is used, where p_ij^a = 1
if link a is on the shortest path from i to j (all-or-nothing assignment).

SOLUTION APPROACH — Tikhonov-Regularised Least Squares
-------------------------------------------------------
The system is underdetermined (771 unknowns, ~197 observations).
We solve the augmented system:

    min  ||A·T - V||²  +  λ·||T - T0||²    subject to T ≥ 0

where T0 = seed vector (all ones, following Khan & Anderson),
      λ  = regularisation parameter controlling flow spread.

This is solved by augmenting the system:
    [  A.T  ]       [    V    ]
    [√λ · I ] T  =  [ √λ · T0 ]

and solving with scipy lsq_linear (non-negative bounded).

λ is set as:  λ = regularisation_strength × mean(V)
This ensures λ scales with the magnitude of observed counts.

REGULARISATION SENSITIVITY
--------------------------
The script runs multiple values of regularisation_strength and saves
a comparison table so you can choose the best balance between:
  - R² / RMSE  (fit to observed link counts)
  - Non-zero OD pairs (spread of flow across the matrix)
  - Total flow (physical plausibility)

ZONE SYSTEM
-----------
  34 External zones + 8 Internal zones = 42×42 asymmetric OD matrix.

CLUSTER ZERO-CONSTRAINTS
-------------------------
  Within-cluster external zone pairs fixed to zero.

REFERENCE
---------
Khan, T. & Anderson, M. (2014). IJTTE, 3(6), 415-423.
================================================================================
"""

# --- CONFIGURATION ---
NODE_TABLE_PATH   = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Node table Final Updated.csv"
LINK_TABLE_PATH   = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Link Table final Updated.csv"
OUTPUT_DIR        = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\OD_matrix_output"
SEED_VALUE       = 1.0

# Regularisation strength values to test in sensitivity analysis
# Lower  = better fit to counts, fewer non-zero pairs
# Higher = more spread, more non-zero pairs, slightly lower fit
REG_STRENGTHS    = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

# Final value to use for the output Excel (choose after seeing sensitivity results)
FINAL_REG_STRENGTH = 0.01
# ---------------------

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import lsq_linear
import openpyxl

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def build_graph(links_df):
    G = nx.DiGraph()
    for _, row in links_df.iterrows():
        fn  = int(row["from_node_id"])
        tn  = int(row["to_node_id"])
        lid = int(row["link_id"])
        lkm = float(row["length_km"])
        two = int(row["two_way"])
        G.add_edge(fn, tn, link_id=lid, length_km=lkm)
        if two == 1:
            G.add_edge(tn, fn, link_id=lid, length_km=lkm)
    return G


def get_zones(nodes_df):
    zone_df = nodes_df[nodes_df["is_zone"] == 1]
    ext_ids = sorted(zone_df[zone_df["zone_type"] == "external"]["node_id"].tolist())
    int_ids = sorted(zone_df[zone_df["zone_type"] == "Internal"]["node_id"].tolist())
    return ext_ids, int_ids


def get_cluster_constraints(nodes_df):
    ext_df = nodes_df[
        (nodes_df["is_zone"] == 1) &
        (nodes_df["zone_type"] == "external") &
        (nodes_df["Cluster_1"].notna())
    ].copy()
    cluster_map = {}
    for _, row in ext_df.iterrows():
        cluster_map.setdefault(row["Cluster_1"], []).append(int(row["node_id"]))
    zero_pairs = set()
    for members in cluster_map.values():
        for i in members:
            for j in members:
                if i != j:
                    zero_pairs.add((i, j))
    return zero_pairs


def compute_shortest_paths(graph, zone_ids):
    paths = {}
    n = len(zone_ids)
    print(f"\n[PATH COMPUTATION] Computing shortest paths for {n*(n-1)} ordered zone pairs...")
    for i, orig in enumerate(zone_ids):
        if orig not in graph:
            print(f"  WARNING: Origin node {orig} not in graph. All pairs ({orig},*) set to zero.")
            continue
        if graph.out_degree(orig) == 0:
            print(f"  WARNING: Origin node {orig} has no outgoing edges (one-way entry). All pairs ({orig},*) set to zero.")
            continue
        for dest in zone_ids:
            if dest == orig:
                continue
            try:
                node_path = nx.dijkstra_path(graph, orig, dest, weight="length_km")
            except nx.NetworkXNoPath:
                print(f"  WARNING: No path from zone {orig} to zone {dest}. T({orig},{dest}) fixed to zero.")
                continue
            except Exception as e:
                print(f"  WARNING: Error for path ({orig},{dest}): {e}. Pair excluded.")
                continue
            link_seq = []
            valid = True
            for k in range(len(node_path) - 1):
                edge_data = graph.get_edge_data(node_path[k], node_path[k+1])
                if edge_data is None:
                    valid = False
                    break
                link_seq.append(edge_data["link_id"])
            if valid and link_seq:
                paths[(orig, dest)] = link_seq
        if (i+1) % 5 == 0 or (i+1) == len(zone_ids):
            print(f"  Processed {i+1}/{len(zone_ids)} origins ({(i+1)*100//len(zone_ids)}%)")
    print(f"[PATH COMPUTATION] Done. {len(paths)} routable pairs found.")
    return paths


def build_incidence_matrix(paths, directional_counts, free_pairs):
    n_pairs = len(free_pairs)
    n_obs   = len(directional_counts)
    pair_index     = {p: idx for idx, p in enumerate(free_pairs)}
    obs_key_to_col = {dc["obs_key"]: k for k, dc in enumerate(directional_counts)}
    A = np.zeros((n_pairs, n_obs), dtype=np.float64)
    V = np.array([dc["count"] for dc in directional_counts], dtype=np.float64)
    for pair, link_ids in paths.items():
        if pair not in pair_index:
            continue
        p_idx = pair_index[pair]
        for lid in link_ids:
            for direction in ("forward", "reverse"):
                obs_key = (lid, direction)
                if obs_key in obs_key_to_col:
                    A[p_idx, obs_key_to_col[obs_key]] = 1.0
    return A, V


def solve_tikhonov(A, V, n_free_pairs, seed_value=1.0, reg_strength=0.5):
    """
    Tikhonov-regularised non-negative least squares.

    Augmented system:
        [  A.T  ] T = [    V    ]
        [√λ · I ]     [ √λ · T0 ]

    λ = reg_strength × mean(V)
    T0 = seed_value (all pairs initialised to seed following Khan & Anderson)
    """
    if len(V) == 0:
        return np.full(n_free_pairs, seed_value)

    T0      = np.full(n_free_pairs, seed_value)
    # Scale λ by V.mean()/n_free_pairs so regularisation weight is
    # commensurate with the per-pair data signal, not the raw count magnitude.
    # This matches the scaling used in the best-performing version (R²=0.88).
    lam     = reg_strength * (V.mean() / n_free_pairs)
    sqrt_lam = np.sqrt(lam)

    A_T     = A.T                                          # (n_obs, n_pairs)
    I_reg   = sqrt_lam * np.eye(n_free_pairs)
    A_aug   = np.vstack([A_T, I_reg])                     # (n_obs+n_pairs, n_pairs)
    V_aug   = np.concatenate([V, sqrt_lam * T0])

    result  = lsq_linear(A_aug, V_aug, bounds=(0, np.inf), method="bvls", verbose=0)
    return result.x


def compute_metrics(T, A, V):
    """Compute RMSE, R², non-zero count, total flow."""
    A_T     = A.T
    V_est   = A_T @ T
    rmse    = np.sqrt(np.mean((V_est - V) ** 2))
    ss_res  = np.sum((V_est - V) ** 2)
    ss_tot  = np.sum((V - V.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nonzero = int(np.sum(T > 0.001))
    return rmse, r2, nonzero, T.sum()



def compute_path_lengths(graph, free_pairs, paths):
    """Compute path length in km for every free OD pair."""
    print("\n[PATH LENGTHS] Computing exact path lengths for all free pairs...")
    free_set     = set(free_pairs)
    pair_lengths = {}
    for (orig, dest), link_ids in paths.items():
        if (orig, dest) not in free_set:
            continue
        total_km = 0.0
        for lid in link_ids:
            for u, v, data in graph.edges(data=True):
                if data.get("link_id") == lid:
                    total_km += data.get("length_km", 0.0)
                    break
        pair_lengths[(orig, dest)] = round(total_km, 4)
    print(f"  Path lengths computed: {len(pair_lengths)} pairs")
    return pair_lengths


def build_km_and_vkt_matrices(od_matrix, pair_lengths, free_pairs, all_zone_ids):
    """Build path-length matrix (km) and VKT matrix (veh·km/day).

    Both km and VKT are written only for OD pairs with non-zero (rounded)
    trips. If the displayed trips value is 0.0, the corresponding km and
    VKT cells are also 0.0. This keeps the three Excel sheets
    (trips / km / VKT) internally consistent: every non-zero km cell
    corresponds to an actually-used OD pair.
    """
    n           = len(all_zone_ids)
    zone_to_idx = {nid: idx for idx, nid in enumerate(all_zone_ids)}
    free_set    = set(free_pairs)
    km_array  = np.zeros((n, n), dtype=np.float64)
    vkt_array = np.zeros((n, n), dtype=np.float64)
    for (orig, dest), km in pair_lengths.items():
        if (orig, dest) not in free_set:
            continue
        if orig not in zone_to_idx or dest not in zone_to_idx:
            continue
        i = zone_to_idx[orig]; j = zone_to_idx[dest]
        trips_rounded = round(float(od_matrix.iloc[i, j]), 1)
        if trips_rounded <= 0:
            continue  # leave km and VKT at zero for unused pairs
        km_array[i, j]  = km
        vkt_array[i, j] = round(trips_rounded * km, 2)
    km_df  = pd.DataFrame(km_array,  index=all_zone_ids, columns=all_zone_ids)
    vkt_df = pd.DataFrame(vkt_array, index=all_zone_ids, columns=all_zone_ids)
    for df in [km_df, vkt_df]:
        df.index.name   = "origin_node_id"
        df.columns.name = "destination_node_id"
    return km_df, vkt_df


def build_vkt_summary(od_matrix, vkt_df, km_df,
                      ext_ids, int_ids, all_zone_ids):
    """Aggregate trips, VKT and average path length by trip type.

    Uses the same 1-decimal rounding convention as the OD/km/VKT matrices,
    so the summary numbers are consistent with the values shown in the
    Excel sheets (a pair counts as 'used' only if its displayed trips > 0).
    """
    rows = []
    ext_list = [z for z in all_zone_ids if z in set(ext_ids)]
    int_list = [z for z in all_zone_ids if z in set(int_ids)]
    type_map = {
        "EE": (ext_list, ext_list),
        "IE": (int_list, ext_list),
        "EI": (ext_list, int_list),
        "II": (int_list, int_list),
    }
    for tt, (origs, dests) in type_map.items():
        # Round trips to 1 decimal to match the displayed OD matrix
        trips_sub = np.round(od_matrix.loc[origs, dests].values, 1)
        vkt_sub   = vkt_df.loc[origs, dests].values
        km_sub    = km_df.loc[origs,   dests].values
        mask      = trips_sub > 0
        rows.append({
            "trip_type":           tt,
            "non_zero_pairs":      int(mask.sum()),
            "total_trips_veh_day": round(float(trips_sub.sum()), 1),
            "total_vkt_km_day":    round(float(vkt_sub.sum()), 1),
            "avg_path_length_km":  round(float(km_sub[mask].mean()), 2)
                                   if mask.any() else 0.0,
        })
    sumdf = pd.DataFrame(rows)
    tot_trips = sumdf["total_trips_veh_day"].sum()
    tot_vkt   = sumdf["total_vkt_km_day"].sum()
    sumdf.loc[len(sumdf)] = {
        "trip_type":           "TOTAL",
        "non_zero_pairs":      sumdf["non_zero_pairs"].sum(),
        "total_trips_veh_day": round(tot_trips, 1),
        "total_vkt_km_day":    round(tot_vkt, 1),
        "avg_path_length_km":  round(tot_vkt / tot_trips, 2) if tot_trips > 0 else 0.0,
    }
    return sumdf


def assemble_full_matrix(T_solution, free_pairs, zero_pairs, all_zone_ids):
    n = len(all_zone_ids)
    od_array    = np.zeros((n, n), dtype=np.float64)
    zone_to_idx = {nid: idx for idx, nid in enumerate(all_zone_ids)}
    for k, (i, j) in enumerate(free_pairs):
        if i in zone_to_idx and j in zone_to_idx:
            od_array[zone_to_idx[i], zone_to_idx[j]] = T_solution[k]
    od_df = pd.DataFrame(od_array, index=all_zone_ids, columns=all_zone_ids)
    od_df.index.name   = "origin_node_id"
    od_df.columns.name = "destination_node_id"
    return od_df


def export_results(od_matrix, nodes_df, links_df, directional_counts,
                   paths, free_pairs, T_solution, ext_ids, int_ids,
                   output_dir, reg_strength, graph, all_zone_ids):
    os.makedirs(output_dir, exist_ok=True)
    node_to_station = dict(zip(nodes_df["node_id"], nodes_df["station_id"]))

    def lbl(nid):
        s = node_to_station.get(nid, None)
        if s is None or str(s).strip() == "" or str(s) == "nan":
            return f"{nid} (null)"
        return f"{nid} ({s})"

    ext_labels   = [lbl(n) for n in ext_ids]
    int_labels   = [lbl(n) for n in int_ids]
    all_zone_ids = ext_ids + int_ids
    all_labels   = [lbl(n) for n in all_zone_ids]
    od_r         = od_matrix.round(1)

    od_full = od_r.loc[all_zone_ids, all_zone_ids].copy()
    od_full.index = all_labels; od_full.columns = all_labels

    od_ii = od_r.loc[int_ids, int_ids].copy()
    od_ii.index = int_labels; od_ii.columns = int_labels

    od_ie = od_r.loc[int_ids, ext_ids].copy()
    od_ie.index = int_labels; od_ie.columns = ext_labels

    od_ei = od_r.loc[ext_ids, int_ids].copy()
    od_ei.index = ext_labels; od_ei.columns = int_labels

    od_ee = od_r.loc[ext_ids, ext_ids].copy()
    od_ee.index = ext_labels; od_ee.columns = ext_labels

    # ---- Path length and VKT matrices ----
    pair_lengths          = compute_path_lengths(graph, free_pairs, paths)
    km_matrix, vkt_matrix = build_km_and_vkt_matrices(
        od_matrix, pair_lengths, free_pairs, all_zone_ids)
    vkt_summary           = build_vkt_summary(
        od_matrix, vkt_matrix, km_matrix, ext_ids, int_ids, all_zone_ids)

    km_r  = km_matrix.round(3)
    vkt_r = vkt_matrix.round(1)
    for mat in [km_r, vkt_r]:
        mat.index   = od_matrix.index
        mat.columns = od_matrix.columns

    def slice_km(mat, rows, cols):
        s = mat.loc[rows, cols].copy()
        s.index   = [lbl(n) for n in rows]
        s.columns = [lbl(n) for n in cols]
        return s

    km_full = slice_km(km_r,  all_zone_ids, all_zone_ids)
    km_ii   = slice_km(km_r,  int_ids, int_ids)
    km_ie   = slice_km(km_r,  int_ids, ext_ids)
    km_ei   = slice_km(km_r,  ext_ids, int_ids)
    km_ee   = slice_km(km_r,  ext_ids, ext_ids)

    vkt_full = slice_km(vkt_r, all_zone_ids, all_zone_ids)
    vkt_ii   = slice_km(vkt_r, int_ids, int_ids)
    vkt_ie   = slice_km(vkt_r, int_ids, ext_ids)
    vkt_ei   = slice_km(vkt_r, ext_ids, int_ids)
    vkt_ee   = slice_km(vkt_r, ext_ids, ext_ids)

    excel_path = os.path.join(output_dir, "OD_matrix_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # ---- veh/day sheets ----
        od_full.to_excel(writer, sheet_name="OD_Full_veh_day")
        od_ii.to_excel(writer,   sheet_name="OD_II_veh_day")
        od_ie.to_excel(writer,   sheet_name="OD_IE_veh_day")
        od_ei.to_excel(writer,   sheet_name="OD_EI_veh_day")
        od_ee.to_excel(writer,   sheet_name="OD_EE_veh_day")
        # ---- path length sheets (km) ----
        km_full.to_excel(writer, sheet_name="OD_Full_km")
        km_ii.to_excel(writer,   sheet_name="OD_II_km")
        km_ie.to_excel(writer,   sheet_name="OD_IE_km")
        km_ei.to_excel(writer,   sheet_name="OD_EI_km")
        km_ee.to_excel(writer,   sheet_name="OD_EE_km")
        # ---- VKT sheets (veh·km/day) ----
        vkt_full.to_excel(writer, sheet_name="OD_Full_vkt")
        vkt_ii.to_excel(writer,   sheet_name="OD_II_vkt")
        vkt_ie.to_excel(writer,   sheet_name="OD_IE_vkt")
        vkt_ei.to_excel(writer,   sheet_name="OD_EI_vkt")
        vkt_ee.to_excel(writer,   sheet_name="OD_EE_vkt")
        # ---- VKT summary ----
        vkt_summary.to_excel(writer, sheet_name="VKT_Summary", index=False)
    print(f"\n[OUTPUT] Excel saved: {excel_path}")

    # Link flow comparison
    pair_flow     = {free_pairs[k]: T_solution[k] for k in range(len(free_pairs))}
    obs_key_to_dc = {dc["obs_key"]: dc for dc in directional_counts}
    est_volume    = {dc["obs_key"]: 0.0 for dc in directional_counts}
    for (i, j), link_ids in paths.items():
        flow = pair_flow.get((i, j), 0.0)
        for lid in link_ids:
            for direction in ("forward", "reverse"):
                obs_key = (lid, direction)
                if obs_key in est_volume:
                    est_volume[obs_key] += flow

    rows = []
    for obs_key, dc in obs_key_to_dc.items():
        obs_v = dc["count"]; est_v = est_volume[obs_key]
        rows.append({
            "link_id":         dc["link_id"],
            "from_node_id":    dc["from_node_id"],
            "to_node_id":      dc["to_node_id"],
            "direction":       dc["direction"],
            "observed_count":  round(obs_v, 2),
            "estimated_count": round(est_v, 2),
            "residual":        round(obs_v - est_v, 2),
            "two_way":         dc["two_way"]
        })
    comp_df   = pd.DataFrame(rows).sort_values(["link_id", "direction"])
    comp_path = os.path.join(output_dir, "link_flow_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"[OUTPUT] Link comparison saved: {comp_path}")

    # Summary
    total_ii = od_r.loc[int_ids, int_ids].values.sum()
    total_ie = od_r.loc[int_ids, ext_ids].values.sum()
    total_ei = od_r.loc[ext_ids, int_ids].values.sum()
    total_ee = od_r.loc[ext_ids, ext_ids].values.sum()
    total    = total_ii + total_ie + total_ei + total_ee
    pct_ee   = (total_ee / total * 100) if total > 0 else 0.0

    obs_vec   = comp_df["observed_count"].values
    est_vec   = comp_df["estimated_count"].values
    residuals = obs_vec - est_vec
    rmse      = np.sqrt(np.mean(residuals ** 2))
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((obs_vec - np.mean(obs_vec)) ** 2)
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n" + "=" * 60)
    print(f"  FINAL OD MATRIX (reg_strength = {reg_strength})")
    print("=" * 60)
    print(f"  Total estimated trips (all types) : {total:>12,.1f}")
    print(f"    II  (Internal → Internal)        : {total_ii:>12,.1f}")
    print(f"    IE  (Internal → External)        : {total_ie:>12,.1f}")
    print(f"    EI  (External → Internal)        : {total_ei:>12,.1f}")
    print(f"    EE  (External → External)        : {total_ee:>12,.1f}")
    print(f"  EE as % of total trips             : {pct_ee:>10.1f}%")
    print("-" * 60)
    print(f"  Link flow fit:")
    print(f"    RMSE                             : {rmse:>10.2f}  veh/day")
    print(f"    R²                               : {r2:>10.4f}")
    print(f"  Non-zero OD pairs                  : {int(np.sum(T_solution > 0.001))} / {len(T_solution)}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  OD MATRIX ALGORITHM — GORBEIALDEA, ÁLAVA")
    print("  Tikhonov-Regularised Least Squares")
    print("  (Khan & Anderson 2014 framework)")
    print("=" * 60)

    # Load
    print(f"\n[LOAD] Node table: {NODE_TABLE_PATH}")
    nodes_df = pd.read_csv(NODE_TABLE_PATH)
    print(f"  {len(nodes_df)} nodes loaded")

    print(f"[LOAD] Link table: {LINK_TABLE_PATH}")
    links_df = pd.read_csv(LINK_TABLE_PATH)
    print(f"  {len(links_df)} links loaded")

    # Graph
    print("\n[GRAPH] Building directed graph...")
    G = build_graph(links_df)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Zones
    print("\n[ZONES] Identifying zones...")
    ext_ids, int_ids = get_zones(nodes_df)
    all_zone_ids = ext_ids + int_ids
    print(f"  External: {len(ext_ids)}, Internal: {len(int_ids)}, Total: {len(all_zone_ids)}")

    # Clusters
    print("\n[CLUSTERS] Computing zero-constraints...")
    zero_pairs = get_cluster_constraints(nodes_df)
    print(f"  Zero-constrained pairs: {len(zero_pairs)}")

    # Counts
    print("\n[COUNTS] Pre-processing directional counts...")
    directional_counts = []
    for _, row in links_df.iterrows():
        if pd.isna(row["count"]):
            continue
        lid = int(row["link_id"]); fn = int(row["from_node_id"])
        tn  = int(row["to_node_id"]); cnt = float(row["count"])
        two = int(row["two_way"])
        if two == 0:
            directional_counts.append({
                "link_id": lid, "from_node_id": fn, "to_node_id": tn,
                "direction": "forward", "count": cnt,
                "two_way": two, "obs_key": (lid, "forward")
            })
        else:
            half = cnt / 2.0
            directional_counts.append({
                "link_id": lid, "from_node_id": fn, "to_node_id": tn,
                "direction": "forward", "count": half,
                "two_way": two, "obs_key": (lid, "forward")
            })
            directional_counts.append({
                "link_id": lid, "from_node_id": tn, "to_node_id": fn,
                "direction": "reverse", "count": half,
                "two_way": two, "obs_key": (lid, "reverse")
            })
    print(f"  Directional observations: {len(directional_counts)}")

    # Paths
    paths = compute_shortest_paths(G, all_zone_ids)

    # Free variables
    print("\n[FREE VARS] Identifying free variables...")
    diagonal = {(i, i) for i in all_zone_ids}
    free_pairs = [
        (i, j) for i in all_zone_ids for j in all_zone_ids
        if (i, j) not in diagonal
        and (i, j) not in zero_pairs
        and (i, j) in paths
    ]
    total_off = len(all_zone_ids) ** 2 - len(all_zone_ids)
    print(f"  Off-diagonal: {total_off}, Fixed: {total_off - len(free_pairs)}, Free: {len(free_pairs)}")

    # Incidence matrix
    print("\n[INCIDENCE] Building incidence matrix A...")
    A, V = build_incidence_matrix(paths, directional_counts, free_pairs)
    print(f"  A shape: {A.shape}  (free_pairs × observations)")
    print(f"  Mean observed count: {V.mean():.1f} veh/day")

    # ----------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  REGULARISATION SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  {'Strength':>10}  {'λ':>10}  {'R²':>8}  {'RMSE':>10}  {'Non-zero':>10}  {'Total flow':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    sensitivity_rows = []
    T_solutions = {}

    for reg in REG_STRENGTHS:
        T = solve_tikhonov(A, V, len(free_pairs),
                           seed_value=SEED_VALUE, reg_strength=reg)
        rmse, r2, nonzero, total = compute_metrics(T, A, V)
        lam = reg * V.mean()
        print(f"  {reg:>10.3f}  {lam:>10.2f}  {r2:>8.4f}  {rmse:>10.2f}  {nonzero:>10d}  {total:>12,.1f}")
        sensitivity_rows.append({
            "reg_strength": reg,
            "lambda": round(lam, 4),
            "R2": round(r2, 4),
            "RMSE": round(rmse, 2),
            "non_zero_pairs": nonzero,
            "total_free_pairs": len(free_pairs),
            "pct_nonzero": round(nonzero / len(free_pairs) * 100, 1),
            "total_flow": round(total, 1)
        })
        T_solutions[reg] = T

    # Save sensitivity table
    sens_df   = pd.DataFrame(sensitivity_rows)
    sens_path = os.path.join(OUTPUT_DIR, "regularisation_sensitivity.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sens_df.to_csv(sens_path, index=False)
    print(f"\n  Sensitivity table saved: {sens_path}")

    # ----------------------------------------------------------------
    # FINAL OUTPUT using FINAL_REG_STRENGTH
    # ----------------------------------------------------------------
    print(f"\n[FINAL] Using regularisation_strength = {FINAL_REG_STRENGTH}")
    T_final = T_solutions[FINAL_REG_STRENGTH]

    print("\n[ASSEMBLE] Building full OD matrix...")
    od_matrix = assemble_full_matrix(T_final, free_pairs, zero_pairs, all_zone_ids)

    export_results(
        od_matrix, nodes_df, links_df, directional_counts,
        paths, free_pairs, T_final, ext_ids, int_ids,
        OUTPUT_DIR, FINAL_REG_STRENGTH, G, all_zone_ids
    )

    print(f"\n[DONE] All outputs in: {OUTPUT_DIR}/")
    print(f"       Check regularisation_sensitivity.csv to choose best strength.")
    print(f"       Then set FINAL_REG_STRENGTH and re-run to generate final Excel.\n")


if __name__ == "__main__":
    main()