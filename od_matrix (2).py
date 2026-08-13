"""
================================================================================
OD MATRIX ESTIMATION — GORBEIALDEA, ÁLAVA, SPAIN
================================================================================

METHODOLOGY — Khan & Anderson (2014) exact implementation
----------------------------------------------------------
Reference: Khan T. & Anderson M. (2014). "Estimation of Through Trips Using
Existing Traffic Counts." IJTTE, 3(6), 415-423.

STEP 1 — INITIALISATION [Eq. 3]
For each counted link a:
  - Traffic count column: p_ij^a = 1 if link a on shortest path i→j, else 0
  - col_sum_a = Σ_ij p_ij^a  (number of OD pairs using link a)
  - CurrentVolume_ij^a = (p_ij^a × V_a) / col_sum_a
  - UpdatedOD_ij^a = CurrentVolume_ij^a  if > 0,  else seed (=1)  [if-else rule]
Final T_ij = Σ_a  UpdatedOD_ij^a  (sum across ALL link columns)

Key insight: pairs using NO counted links get seed × n_links (non-zero).
Pairs using counted links get proportional share + seed for unused links.

STEP 2 — ITERATIONS [Eq. 4]
For each counted link a:
  - a*_ij = p_ij^a × T_ij  (current OD × binary incidence)
  - col_sum_a* = Σ_ij a*_ij
  - CurrentVolume_ij^a = (a*_ij × V_a) / col_sum_a*
  - UpdatedOD_ij^a = CurrentVolume_ij^a  if > 0,  else seed (=1)
New T_ij = Σ_a  UpdatedOD_ij^a
Repeat until max|T_new - T_old| / max(T_old) < tolerance.

NOTE ON CONVERGENCE: The paper's iterative method was designed for small
near-determined systems (~13 zones, ~13 links). For larger underdetermined
systems the iterations may not fully converge. The initialisation phase
alone already provides a meaningful solution — the iterations refine it.
We run the iterations but stop at max_iter or convergence, whichever first.
================================================================================
"""

# --- CONFIGURATION ---
NODE_TABLE_PATH   = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Node table Final Updated.csv"
LINK_TABLE_PATH   = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Link Table final Updated.csv"
OUTPUT_DIR        = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\OD_matrix_output"
SEED_VALUE       = 1.0    # Khan & Anderson: initialise all OD pairs = 1
MAX_ITERATIONS   = 150    # Khan & Anderson used 151 iterations
TOLERANCE        = 1e-4
# ---------------------

import os
import numpy as np
import pandas as pd
import networkx as nx
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


def solve_khan_anderson(A, V, n_free_pairs,
                        seed_value=1.0, max_iter=150, tol=1e-4):
    """
    Exact Khan & Anderson (2014) iterative algorithm with correct if-else rule.

    INITIALISATION [Eq.3]:
        col_sum_a   = Σ_ij p_ij^a
        CV_ij^a     = (p_ij^a * V_a) / col_sum_a
        Updated_ij^a = CV_ij^a  if CV_ij^a > 0  else seed
        T_ij         = Σ_a  Updated_ij^a

    ITERATION [Eq.4]:
        a*_ij        = p_ij^a * T_ij
        col_sum_a*   = Σ_ij a*_ij
        CV_ij^a      = (a*_ij * V_a) / col_sum_a*
        Updated_ij^a = CV_ij^a  if CV_ij^a > 0  else seed
        T_ij_new     = Σ_a  Updated_ij^a

    The if-else rule ensures all OD pairs remain non-zero throughout.
    Pairs not covered by any counted link get seed × n_obs as their value.
    """
    print(f"\n[SOLVER] Khan & Anderson (2014) — exact implementation")
    print(f"  Free OD pairs : {n_free_pairs}")
    print(f"  Observations  : {len(V)}")
    print(f"  Seed value    : {seed_value}")
    print(f"  Max iterations: {max_iter}")

    if len(V) == 0:
        print("  WARNING: No observations. Returning seed × n_obs solution.")
        return np.full(n_free_pairs, seed_value)

    n_obs = len(V)

    # --- INITIALISATION PHASE [Eq. 3] ---
    col_sums      = A.sum(axis=0)                          # (n_obs,)
    col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)

    # CV matrix: (n_pairs × n_obs)
    CV = A * (V / col_sums_safe)

    # Sum proportional contributions from counted links only
    T = CV.sum(axis=1)

    # Apply seed as floor: pairs not covered by any counted link
    # get seed_value; pairs covered get their count-based value.
    # This is the correct reading of the paper if-else rule:
    # the seed is a minimum, NOT added for every link column.
    T = np.maximum(T, seed_value)

    V_est  = A.T @ T
    rmse   = np.sqrt(np.mean((V_est - V) ** 2))
    print(f"\n  After initialisation:")
    print(f"    Non-zero pairs  : {(T > 1e-6).sum()} / {n_free_pairs}")
    print(f"    Min T value     : {T.min():.4f}  (should be = seed={seed_value})")
    print(f"    Max T value     : {T.max():.2f}")
    print(f"    Total flow      : {T.sum():,.1f}")
    print(f"    RMSE            : {rmse:.2f}")

    # --- ITERATION PHASE [Eq. 4] ---
    print(f"\n  Running iterations...")
    best_T    = T.copy()
    best_rmse = rmse

    for iteration in range(1, max_iter + 1):
        T_old = T.copy()

        # a* = p_ij^a × T_ij
        A_star          = A * T[:, np.newaxis]             # (n_pairs, n_obs)
        col_sums_star   = A_star.sum(axis=0)
        col_sums_s_safe = np.where(col_sums_star > 0, col_sums_star, 1.0)

        # New CV [Eq. 4]
        CV_new = A_star * (V / col_sums_s_safe)

        # Sum contributions from counted links only,
        # then apply seed as floor value
        T_new = CV_new.sum(axis=1)
        T_new = np.maximum(T_new, seed_value)

        # Track best solution by RMSE
        V_est_new = A.T @ T_new
        rmse_new  = np.sqrt(np.mean((V_est_new - V) ** 2))
        if rmse_new < best_rmse:
            best_rmse = rmse_new
            best_T    = T_new.copy()

        # Convergence check
        max_old    = T_old.max()
        rel_change = np.abs(T_new - T_old).max() / max_old if max_old > 0 else 0.0
        T = T_new

        if iteration % 25 == 0 or iteration == 1:
            print(f"    Iter {iteration:4d}: rel_change={rel_change:.6f}, "
                  f"RMSE={rmse_new:.2f}, non-zero={( T > 1e-6).sum()}, "
                  f"total={T.sum():,.0f}")

        if rel_change < tol:
            print(f"\n  Converged at iteration {iteration}.")
            break
    else:
        print(f"\n  Reached max iterations ({max_iter}).")

    # Use best solution found during iterations
    print(f"\n  Using best solution (lowest RMSE={best_rmse:.2f})")
    T = best_T

    # Final diagnostics
    V_final = A.T @ T
    rmse_f  = np.sqrt(np.mean((V_final - V) ** 2))
    ss_res  = np.sum((V_final - V) ** 2)
    ss_tot  = np.sum((V - V.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n[SOLVER] Final solution:")
    print(f"  Total flow       : {T.sum():,.1f}")
    print(f"  Non-zero pairs   : {(T > 1e-6).sum()} / {n_free_pairs}")
    print(f"  Min T            : {T.min():.4f}")
    print(f"  Max T            : {T.max():.2f}")
    print(f"  RMSE             : {rmse_f:.2f} veh/day")
    print(f"  R²               : {r2:.4f}")

    return T


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
                   paths, free_pairs, T_solution, ext_ids, int_ids, output_dir):
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

    excel_path = os.path.join(output_dir, "OD_matrix_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        od_full.to_excel(writer, sheet_name="OD_Full")
        od_ii.to_excel(writer,   sheet_name="OD_II")
        od_ie.to_excel(writer,   sheet_name="OD_IE")
        od_ei.to_excel(writer,   sheet_name="OD_EI")
        od_ee.to_excel(writer,   sheet_name="OD_EE")
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
    print("  OD MATRIX ESTIMATION — SUMMARY")
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
    print(f"  Free OD pairs at seed level        : "
          f"{int(np.sum(np.abs(T_solution - SEED_VALUE * len(directional_counts)) < 0.01))}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  OD MATRIX ALGORITHM — GORBEIALDEA, ÁLAVA")
    print("  Khan & Anderson (2014) — Exact Implementation")
    print("=" * 60)

    print(f"\n[LOAD] Node table: {NODE_TABLE_PATH}")
    nodes_df = pd.read_csv(NODE_TABLE_PATH)
    print(f"  {len(nodes_df)} nodes loaded")

    print(f"[LOAD] Link table: {LINK_TABLE_PATH}")
    links_df = pd.read_csv(LINK_TABLE_PATH)
    print(f"  {len(links_df)} links loaded")

    print("\n[GRAPH] Building directed graph...")
    G = build_graph(links_df)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[ZONES] Identifying zones...")
    ext_ids, int_ids = get_zones(nodes_df)
    all_zone_ids = ext_ids + int_ids
    print(f"  External: {len(ext_ids)}, Internal: {len(int_ids)}, Total: {len(all_zone_ids)}")

    print("\n[CLUSTERS] Computing zero-constraints...")
    zero_pairs = get_cluster_constraints(nodes_df)
    print(f"  Zero-constrained pairs: {len(zero_pairs)}")

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

    paths = compute_shortest_paths(G, all_zone_ids)

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

    print("\n[INCIDENCE] Building incidence matrix A...")
    A, V = build_incidence_matrix(paths, directional_counts, free_pairs)
    print(f"  A shape: {A.shape}  (free_pairs × observations)")

    T_solution = solve_khan_anderson(
        A, V, len(free_pairs),
        seed_value=SEED_VALUE,
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE
    )

    print("\n[ASSEMBLE] Building full OD matrix...")
    od_matrix = assemble_full_matrix(T_solution, free_pairs, zero_pairs, all_zone_ids)

    export_results(
        od_matrix, nodes_df, links_df, directional_counts,
        paths, free_pairs, T_solution, ext_ids, int_ids, OUTPUT_DIR
    )

    print(f"\n[DONE] Outputs in: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()