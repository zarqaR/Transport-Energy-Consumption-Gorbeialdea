"""
================================================================================
OD MATRIX ESTIMATION — GORBEIALDEA, ÁLAVA, SPAIN
Maximum Entropy (ME2) Method — Van Zuylen & Willumsen (1980)
================================================================================

METHODOLOGY
-----------
Reference: Van Zuylen, H.J. & Willumsen, L.G. (1980). "The most likely trip
matrix estimated from traffic counts." Transportation Research B, 14(3), 281-293.

THEORETICAL FORMULATION (Section 6 of paper)
---------------------------------------------
We seek the OD matrix {T_ij} that maximises entropy:

    max S' = -Σ_ij (T_ij * ln(T_ij) - T_ij)

subject to:
    V_a = Σ_ij T_ij * p_ij^a    for all counted links a    [Eq. 1]
    T_ij ≥ 0

The formal solution via Lagrangean differentiation [Eq. 28-30] gives:

    T_ij = t_ij * ∏_a X_a^(p_ij^a)                        [Eq. 30 / 33]

where t_ij is the prior (seed) matrix (t_ij = 1 for all pairs when no prior
information is available), and X_a are scaling factors (one per counted link)
to be determined so that the count constraints are satisfied.

ALGORITHM (Section 8 of paper — Multi-Proportional Scaling)
------------------------------------------------------------
Murchland (1977) showed this is a multi-proportional problem. The algorithm
processes each counted link in turn:

Step 1: Initialise X_a = 1 for all links a.

Step 2: For each link a (loop through all counted links):
   (a) Compute current T_ij = t_ij * ∏_b X_b^(p_ij^b) for all pairs
   (b) Compute modelled volume: V_a_est = Σ_ij p_ij^a * T_ij
   (c) Find scaling factor Y_a such that:
           V_a_observed = Σ_ij p_ij^a * T_ij * Y_a^(p_ij^a / g_ij)
       where g_ij = Σ_a p_ij^a (number of counted links on path i→j)
       
       For all-or-nothing assignment with p_ij^a ∈ {0,1}:
           p_ij^a / g_ij = 1/g_ij for pairs using link a
       
       The update simplifies to:
           Y_a = (V_a_observed / V_a_est)^(1 / average_weight)
       
       In the simplest case (all pairs weighted equally):
           X_a ← X_a * (V_a_observed / V_a_est)

Step 3: Repeat Step 2 until all modelled volumes are within tolerance of
        observed volumes (paper uses ±2% or ±5%).

IMPLEMENTATION NOTE
-------------------
For all-or-nothing assignment (p_ij^a ∈ {0,1}) with uniform seed t_ij=1,
the update rule per link simplifies to:

    X_a ← X_a * (V_a_obs / V_a_est)^(1 / g_ij_avg)

where g_ij_avg is the average number of counted links per path. We use a
dampened version for numerical stability:

    X_a ← X_a * (V_a_obs / V_a_est)^damping

with damping = 1.0 / (1 + mean path length in counted links).

ZONE SYSTEM
-----------
  34 External zones + 8 Internal zones = 42×42 asymmetric OD matrix.

CLUSTER ZERO-CONSTRAINTS
-------------------------
  Within-cluster external zone pairs fixed to zero throughout.

REFERENCES
----------
Van Zuylen, H.J. & Willumsen, L.G. (1980). Transportation Research B, 14(3).
Murchland, J. (1977). The multi-proportional problem. UCL Research Note.
Wilson, A.G. (1970). Entropy in Urban and Regional Modelling. Pion.
Khan, T. & Anderson, M. (2014). IJTTE, 3(6), 415-423.
================================================================================
"""

# --- CONFIGURATION ---
NODE_TABLE_PATH = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Node table Final Updated.csv"
LINK_TABLE_PATH = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\Link Table final Updated.csv"
OUTPUT_DIR      = r"B:\Master Thesis Zarqa Claude Project\MASTER THESIS ZARQA\OD_matrix_output_ME2"

SEED_VALUE      = 1.0     # Uniform prior t_ij = 1 (no prior information)
MAX_ITERATIONS  = 500     # Maximum full passes through all counted links
CONVERGENCE_PCT = 2.0     # Stop when all links within this % of observed
DAMPING         = 0.5     # Step damping for numerical stability (0 < d ≤ 1)
#                         # Lower = more stable but slower convergence
# ---------------------

import os
import numpy as np
import pandas as pd
import networkx as nx
import openpyxl

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  SHARED NETWORK FUNCTIONS (identical to Tikhonov script)
# ============================================================

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


# ============================================================
#  ME2 SOLVER — Van Zuylen & Willumsen (1980) exact algorithm
# ============================================================

def solve_me2(A, V, n_free_pairs,
              seed_value=1.0, max_iter=500,
              convergence_pct=2.0, damping=0.5):
    """
    Maximum Entropy OD Estimation — exact algorithm from Van Zuylen &
    Willumsen (1980), Section 8: Multi-Proportional Scaling.

    Solution form [Eq. 30/33]:
        T_ij = t_ij * prod_a( X_a ^ p_ij^a )

    With all-or-nothing assignment (p_ij^a in {0,1}) and log-domain:
        ln T_ij = ln t_ij + sum_a( p_ij^a * ln X_a )
                = ln t_ij + A_ij @ ln_X

    Algorithm (Section 8):
        Initialise: X_a = 1 for all a  →  ln_X = 0
        Repeat until convergence:
            For each link a:
                V_est_a = sum_ij( p_ij^a * T_ij )  [current modelled volume]
                ratio   = V_obs_a / V_est_a
                X_a    ← X_a * ratio^damping        [multiplicative update]
                Recompute T_ij using updated X_a

    Parameters
    ----------
    A            : (n_free_pairs, n_obs) binary incidence matrix
    V            : (n_obs,) observed directional counts
    n_free_pairs : number of free OD variables
    seed_value   : uniform prior t_ij (default 1.0)
    max_iter     : maximum full passes through all counted links
    convergence_pct : stop when all links within this % of observed
    damping      : step damping 0 < d ≤ 1 (lower = more stable, slower)

    Returns
    -------
    T : (n_free_pairs,) estimated OD flows
    """
    print(f"\n[SOLVER] Maximum Entropy (ME2) — Van Zuylen & Willumsen (1980)")
    print(f"  Algorithm    : Multi-proportional scaling [Section 8, Eq.30/33]")
    print(f"  Free OD pairs: {n_free_pairs}")
    print(f"  Observations : {len(V)}")
    print(f"  Seed value   : {seed_value}")
    print(f"  Convergence  : ±{convergence_pct}% on all link volumes")
    print(f"  Damping      : {damping}")

    if len(V) == 0:
        print("  WARNING: No observations. Returning uniform seed.")
        return np.full(n_free_pairs, seed_value)

    n_obs = len(V)

    # Work in log-domain for numerical stability
    # ln T_ij = ln t_ij + A @ ln_X
    # With uniform seed: ln t_ij = ln(seed_value) = 0 when seed=1
    ln_t   = np.full(n_free_pairs, np.log(seed_value))  # prior log-values
    ln_X   = np.zeros(n_obs)                             # log scaling factors

    # g_ij = number of counted links on path i→j = row sums of A
    g = A.sum(axis=1)  # (n_free_pairs,)
    g = np.where(g > 0, g, 1.0)  # avoid division by zero

    print(f"\n  Path statistics:")
    print(f"    Pairs using ≥1 counted link : {int((A.sum(axis=1) > 0).sum())} / {n_free_pairs}")
    print(f"    Pairs using 0 counted links : {int((A.sum(axis=1) == 0).sum())} / {n_free_pairs}")
    print(f"    Mean counted links per path : {g.mean():.2f}")
    print(f"    Max counted links per path  : {g.max():.0f}")

    best_T    = np.exp(ln_t + A @ ln_X)
    V_est_init = A.T @ best_T
    best_rmse = np.sqrt(np.mean((V_est_init - V) ** 2))

    print(f"\n  Initial state (X=1, T=seed):")
    print(f"    Total flow  : {best_T.sum():,.1f}")
    print(f"    RMSE        : {best_rmse:.2f}")
    print(f"\n  Running iterations...")

    for iteration in range(1, max_iter + 1):

        # Process each counted link in turn
        for a in range(n_obs):

            # Current T from log-domain
            ln_T = ln_t + A @ ln_X          # (n_free_pairs,)
            T    = np.exp(ln_T)

            # Modelled volume on link a
            V_est_a = float(A[:, a] @ T)

            if V_est_a < 1e-10:
                # No flow on this link — cannot update
                continue

            # Ratio of observed to modelled
            ratio = V[a] / V_est_a

            # Multiplicative update in log domain
            # X_a <- X_a * ratio^damping
            # ln_X_a <- ln_X_a + damping * ln(ratio)
            ln_X[a] += damping * np.log(ratio)

        # End of full pass through all links
        # Compute diagnostics
        ln_T  = ln_t + A @ ln_X
        T     = np.exp(ln_T)
        V_est = A.T @ T

        rmse = np.sqrt(np.mean((V_est - V) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_T    = T.copy()

        # Convergence check: all links within convergence_pct %
        pct_errors = np.abs(V_est - V) / np.where(V > 0, V, 1.0) * 100
        max_pct_err = pct_errors.max()
        mean_pct_err = pct_errors.mean()
        nonzero = int(np.sum(T > 0.001))

        if iteration % 50 == 0 or iteration == 1:
            print(f"    Iter {iteration:4d}: max_link_err={max_pct_err:.2f}%, "
                  f"mean_err={mean_pct_err:.2f}%, "
                  f"RMSE={rmse:.2f}, non-zero={nonzero}, "
                  f"total={T.sum():,.1f}")

        if max_pct_err <= convergence_pct:
            print(f"\n  Converged at iteration {iteration} "
                  f"(all links within {convergence_pct}% of observed)")
            break
    else:
        print(f"\n  Reached max iterations ({max_iter}).")
        print(f"  Best RMSE achieved: {best_rmse:.2f}")

    T = best_T

    # Final diagnostics
    V_final = A.T @ T
    rmse_f  = np.sqrt(np.mean((V_final - V) ** 2))
    ss_res  = np.sum((V_final - V) ** 2)
    ss_tot  = np.sum((V - V.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n[SOLVER] Final ME2 solution:")
    print(f"  Total flow       : {T.sum():,.1f}")
    print(f"  Non-zero pairs   : {int(np.sum(T > 0.001))} / {n_free_pairs}")
    print(f"  Min T            : {T.min():.6f}")
    print(f"  Max T            : {T.max():.2f}")
    print(f"  RMSE             : {rmse_f:.2f} veh/day")
    print(f"  R²               : {r2:.4f}")

    return T



def compute_path_lengths(graph, free_pairs, paths):
    """
    Compute path length in km for every free OD pair using
    the link sequences already stored in paths dict.
    Returns dict: (orig, dest) -> path_length_km
    """
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
    """
    Build path-length matrix (km) and VKT matrix (veh·km/day).
    VKT cell = (rounded trips/day) × path_length_km

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
        km_array[i, j]   = km
        vkt_array[i, j]  = round(trips_rounded * km, 2)

    km_df  = pd.DataFrame(km_array,  index=all_zone_ids, columns=all_zone_ids)
    vkt_df = pd.DataFrame(vkt_array, index=all_zone_ids, columns=all_zone_ids)
    for df in [km_df, vkt_df]:
        df.index.name  = "origin_node_id"
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


# ============================================================
#  MATRIX ASSEMBLY AND EXPORT
# ============================================================

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
                   paths, free_pairs, T_solution, ext_ids, int_ids, output_dir,
                   graph, all_zone_ids):
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
    pair_lengths         = compute_path_lengths(graph, free_pairs, paths)
    km_matrix, vkt_matrix = build_km_and_vkt_matrices(
        od_matrix, pair_lengths, free_pairs, all_zone_ids)
    vkt_summary          = build_vkt_summary(
        od_matrix, vkt_matrix, km_matrix, ext_ids, int_ids, all_zone_ids)

    # Apply same labels to km and vkt matrices
    km_r  = km_matrix.round(3)
    vkt_r = vkt_matrix.round(1)
    for mat in [km_r, vkt_r]:
        mat.index    = od_matrix.index
        mat.columns  = od_matrix.columns

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

    excel_path = os.path.join(output_dir, "OD_matrix_results_ME2.xlsx")
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
    comp_path = os.path.join(output_dir, "link_flow_comparison_ME2.csv")
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
    print("  OD MATRIX ESTIMATION — ME2 SUMMARY")
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
    print(f"  Non-zero OD pairs                  : "
          f"{int(np.sum(T_solution > 0.001))} / {len(T_solution)}")
    print("-" * 60)
    print(f"  BOUNDARY COUNT VALIDATION:")
    print(f"    Sum boundary AADTs / 2 (expected): ~98,142 veh/day")
    print(f"    Algorithm total                  : {total:>10,.1f} veh/day")
    diff = abs(total - 98142) / 98142 * 100
    print(f"    Difference                       : {diff:>10.1f}%")
    print("=" * 60)


# ============================================================
#  MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("  OD MATRIX ALGORITHM — GORBEIALDEA, ÁLAVA")
    print("  Maximum Entropy (ME2) Method")
    print("  Van Zuylen & Willumsen (1980)")
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
    print(f"  External: {len(ext_ids)}, Internal: {len(int_ids)}, "
          f"Total: {len(all_zone_ids)}")

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
    print(f"  Off-diagonal: {total_off}, "
          f"Fixed: {total_off - len(free_pairs)}, "
          f"Free: {len(free_pairs)}")

    print("\n[INCIDENCE] Building incidence matrix A...")
    A, V = build_incidence_matrix(paths, directional_counts, free_pairs)
    print(f"  A shape: {A.shape}  (free_pairs × observations)")
    print(f"  Mean observed count: {V.mean():.1f} veh/day")

    T_solution = solve_me2(
        A, V, len(free_pairs),
        seed_value=SEED_VALUE,
        max_iter=MAX_ITERATIONS,
        convergence_pct=CONVERGENCE_PCT,
        damping=DAMPING
    )

    print("\n[ASSEMBLE] Building full OD matrix...")
    od_matrix = assemble_full_matrix(
        T_solution, free_pairs, zero_pairs, all_zone_ids
    )

    export_results(
        od_matrix, nodes_df, links_df, directional_counts,
        paths, free_pairs, T_solution, ext_ids, int_ids, OUTPUT_DIR,
        G, all_zone_ids
    )

    print(f"\n[DONE] All outputs in: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()