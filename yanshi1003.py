# In[]
# -*- coding: utf-8 -*-
# =============================================================================
# Grid Voltage (Plotly + Dash)
# 拓扑动态 + 自适应像素 + 颜色增益 + 全屏铺满 + 更明显的运行工况变化
# 依赖: plotly, pandapower, numpy, pandas, networkx, dash
# =============================================================================

import json
from typing import Optional, Tuple, Any, Dict, Set, List

import numpy as np
import pandas as pd
import pandapower as pp
import plotly.graph_objects as go
import networkx as nx

# ========= 你的工程对象（如有） =========
# 若找不到自定义模块，会自动回退到演示网络
try:
    from new746_system_v0713 import (
        CimEParser, PandaPowerFlowCalculator, PfDataPath,
        init_feeder_net, built_ppnet_for_pfcal
    )
    HAS_CUSTOM = True
except Exception:
    HAS_CUSTOM = False

# ------------------------------
# 通用工具
# ------------------------------
def copy_bus_geodata_from_main_to_sub(main_net: pp.pandapowerNet,
                                      sub_net: pp.pandapowerNet,
                                      overwrite: bool = False) -> None:
    if "geo" not in getattr(main_net, "bus", pd.DataFrame()).columns:
        return
    if "geo" not in sub_net.bus.columns:
        sub_net.bus["geo"] = None
    name_to_geo = dict(zip(main_net.bus["name"].astype(str), main_net.bus["geo"]))
    for b_idx in sub_net.bus.index:
        bname = str(sub_net.bus.at[b_idx, "name"])
        if (not overwrite) and pd.notna(sub_net.bus.at[b_idx, "geo"]):
            continue
        if bname in name_to_geo and pd.notna(name_to_geo[bname]):
            sub_net.bus.at[b_idx, "geo"] = name_to_geo[bname]

def _parse_geo_any(val: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        if isinstance(val, dict):
            if val.get("type", "").lower() == "point" and isinstance(val.get("coordinates"), (list, tuple)) and len(val["coordinates"]) >= 2:
                return float(val["coordinates"][0]), float(val["coordinates"][1])
            x = val.get("x", val.get("lon", val.get("lng")))
            y = val.get("y", val.get("lat"))
            if x is not None and y is not None:
                return float(x), float(y)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None, None
            if s.startswith("{"):
                d = json.loads(s)
                if d.get("type", "").lower() == "point" and isinstance(d.get("coordinates"), (list, tuple)) and len(d["coordinates"]) >= 2:
                    return float(d["coordinates"][0]), float(d["coordinates"][1])
                x = d.get("x", d.get("lon", d.get("lng")))
                y = d.get("y", d.get("lat"))
                if x is not None and y is not None:
                    return float(x), float(y)
            if s.upper().startswith("POINT"):
                inside = s[s.find("(") + 1:s.find(")")]
                a, b = inside.split()
                return float(a), float(b)
            s2 = s.strip("()")
            if "," in s2:
                a, b = s2.split(",", 1)
                return float(a), float(b)
    except Exception:
        pass
    return None, None

# ------------------------------
# 归一化（更小留白 + 可选等比）
# ------------------------------
def normalize_geodf(geodf: pd.DataFrame,
                    real_mask: Optional[pd.Series] = None,
                    pad: float = 0.02,          # 更小的留白，画面更满
                    keep_aspect: bool = False,
                    clip_outliers: bool = True) -> pd.DataFrame:
    g = geodf.copy()
    if real_mask is not None and real_mask.any():
        xr = g.loc[real_mask, "x"].to_numpy()
        yr = g.loc[real_mask, "y"].to_numpy()
    else:
        xr = g["x"].to_numpy(); yr = g["y"].to_numpy()
    if clip_outliers and len(xr) >= 5:
        x0, x1 = np.quantile(xr, [0.01, 0.99]); y0, y1 = np.quantile(yr, [0.01, 0.99])
    else:
        x0, x1 = float(np.min(xr)), float(np.max(xr))
        y0, y1 = float(np.min(yr)), float(np.max(yr))
    if x1 - x0 < 1e-9: x1 = x0 + 1.0
    if y1 - y0 < 1e-9: y1 = y0 + 1.0

    gx = (g["x"] - x0) / (x1 - x0); gy = (g["y"] - y0) / (y1 - y0)
    if keep_aspect:
        xmin, xmax = gx.min(), gx.max(); ymin, ymax = gy.min(), gy.max()
        w, h = (xmax - xmin), (ymax - ymin); s = max(w, h) or 1.0
        gx = (gx - xmin) / s; gy = (gy - ymin) / s
        if w < h: gx += (1.0 - w / s) * 0.5
        elif h < w: gy += (1.0 - h / s) * 0.5
    gx = gx * (1 - 2 * pad) + pad; gy = gy * (1 - 2 * pad) + pad
    g["x"], g["y"] = gx, gy
    return g

# ------------------------------
# 拓扑布局（spring + geo锚点）
# ------------------------------
def _build_bus_graph(net: pp.pandapowerNet) -> nx.Graph:
    G = nx.Graph()
    for b in net.bus.index: G.add_node(int(b))
    if hasattr(net, "line") and len(net.line):
        for _, r in net.line.iterrows():
            G.add_edge(int(r["from_bus"]), int(r["to_bus"]), kind="line")
    if hasattr(net, "trafo") and len(net.trafo):
        for _, r in net.trafo.iterrows():
            G.add_edge(int(r["hv_bus"]), int(r["lv_bus"]), kind="trafo", weight=0.7)
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for _, r in net.trafo3w.iterrows():
            hv, mv, lv = int(r["hv_bus"]), int(r["mv_bus"]), int(r["lv_bus"])
            G.add_edge(hv, mv, kind="trafo3w", weight=0.7)
            G.add_edge(mv, lv, kind="trafo3w", weight=0.7)
    if hasattr(net, "switch") and len(net.switch):
        bb = net.switch[net.switch["et"] == "b"]
        for _, r in bb.iterrows():
            G.add_edge(int(r["bus"]), int(r["element"]), kind="switch", weight=0.6)
    return G

def _anchors_from_geo(net: pp.pandapowerNet) -> Tuple[Dict[int, Tuple[float, float]], pd.Series]:
    if "geo" not in net.bus.columns:
        net.bus["geo"] = None
    pos, has_geo = {}, pd.Series(False, index=net.bus.index)
    for i, row in net.bus.iterrows():
        x, y = _parse_geo_any(row.get("geo"))
        if x is not None and y is not None:
            pos[int(i)] = (float(x), float(y))
            has_geo.at[i] = True
    return pos, has_geo

def ensure_geodf_topo(display_net: pp.pandapowerNet,
                      calculation_net: Optional[pp.pandapowerNet] = None,
                      keep_aspect: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    if calculation_net is not None and "geo" in getattr(calculation_net, "bus", pd.DataFrame()).columns:
        copy_bus_geodata_from_main_to_sub(calculation_net, display_net, overwrite=False)

    anchor_pos, has_geo = _anchors_from_geo(display_net)
    G = _build_bus_graph(display_net)
    nodes_all: Set[int] = set(int(b) for b in display_net.bus.index)
    pos_all: Dict[int, Tuple[float, float]] = {}
    comps = list(nx.connected_components(G)) if len(G) else [nodes_all]
    seed = 42
    for comp in comps:
        comp = set(int(x) for x in comp)
        Gc = G.subgraph(comp).copy()
        pos0 = {n: anchor_pos[n] for n in comp if n in anchor_pos}
        fixed0 = set(pos0.keys())
        k = 1.2 / max(np.sqrt(len(comp)), 1.0)
        pos = nx.spring_layout(Gc, pos=pos0 if pos0 else None, fixed=fixed0 if fixed0 else None,
                               k=k, iterations=250, seed=seed)
        pos_all.update(pos)
    isolated = list(nodes_all - set(pos_all.keys()))
    for k, i in enumerate(isolated):
        pos_all[i] = (float((k % 25) * 0.5), float((k // 25) * 0.5))

    geodf = pd.DataFrame([(n, p[0], p[1]) for n, p in pos_all.items()],
                         columns=["bus", "x", "y"]).set_index("bus").reindex(display_net.bus.index)
    geodf = normalize_geodf(geodf, real_mask=has_geo, pad=0.02, keep_aspect=keep_aspect, clip_outliers=True)
    return geodf, has_geo

# ------------------------------
# 线路几何（按 in_service 构建）
# ------------------------------
def assemble_line_coords(net: pp.pandapowerNet, geodf: pd.DataFrame,
                         only_in_service: bool = True) -> Tuple[List[float], List[float]]:
    lx, ly = [], []
    # lines
    if hasattr(net, "line") and len(net.line):
        tab = net.line
        if only_in_service and "in_service" in tab.columns:
            tab = tab[tab["in_service"]]
        for lid in list(tab.index):
            try:
                f = int(net.line.at[lid, "from_bus"]) ; t = int(net.line.at[lid, "to_bus"]) 
                if f in geodf.index and t in geodf.index:
                    fx, fy = geodf.at[f, "x"], geodf.at[f, "y"]
                    tx, ty = geodf.at[t, "x"], geodf.at[t, "y"]
                    if pd.notna(fx) and pd.notna(fy) and pd.notna(tx) and pd.notna(ty):
                        lx += [fx, tx, None]; ly += [fy, ty, None]
            except Exception:
                pass
    # trafos
    if hasattr(net, "trafo") and len(net.trafo):
        ttab = net.trafo
        if only_in_service and "in_service" in ttab.columns:
            ttab = ttab[ttab["in_service"]]
        for _, r in ttab.iterrows():
            hv, lv = int(r["hv_bus"]), int(r["lv_bus"])
            if hv in geodf.index and lv in geodf.index:
                lx += [geodf.at[hv, "x"], geodf.at[lv, "x"], None]
                ly += [geodf.at[hv, "y"], geodf.at[lv, "y"], None]
    return lx, ly

# ------------------------------
# 电压序列 & 自适应像素
# ------------------------------

def _autosize_markers(n_bus: int) -> Tuple[int, int, int]:
    if n_bus <= 0: return 5, 7, 2
    dyn = int(np.clip(560 / max(np.sqrt(n_bus) * 18.0, 1.0), 3, 12))  # 3~12 px
    stat = max(2, int(dyn * 0.7))
    lw = max(2, int(dyn / 2.5))
    return stat, dyn, lw

# 颜色对比度增益（仅用于着色）

def _amplify_for_color(v: np.ndarray, center: float = 1.0, gain: float = 1.9,
                       cmin: float = 0.94, cmax: float = 1.06) -> np.ndarray:
    vv = center + gain * (v - center)
    return np.clip(vv, cmin, cmax)

# ------------------------------
# 增强的动态演化系统（更“真实”的缓变 + 事件）
# ------------------------------
class EnhancedDynamicSystem:
    def __init__(self, n_buses, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_buses = n_buses
        self.V = np.ones(n_buses)
        self.mu = np.ones(n_buses)
        self.g = 0.0
        # 增强的动态参数
        self.local_drifts = self.rng.normal(0, 0.005, n_buses)
        self.oscillation_freqs = self.rng.uniform(0.05, 0.2, n_buses)
        self.oscillation_phases = self.rng.uniform(0, 2*np.pi, n_buses)
        self.oscillation_amps = self.rng.uniform(0.002, 0.01, n_buses)
        # 事件系统
        self.active_events = []
        self.event_counter = 0

    def add_event(self, event_type, bus_indices=None, duration=50, magnitude=0.03):
        if bus_indices is None:
            bus_indices = list(range(self.n_buses))
        event = {
            'type': event_type,
            'bus_indices': list(map(int, bus_indices)),
            'duration': int(duration),
            'magnitude': float(magnitude),
            'start_time': int(self.event_counter),
            'end_time': int(self.event_counter + duration)
        }
        self.active_events.append(event)

    def update(self, alpha=0.90, noise_sigma=0.012, global_drift_sigma=0.0015):
        self.event_counter += 1
        # 基础动态
        self.g += self.rng.normal(0.0, global_drift_sigma)
        self.V = alpha * self.V + (1 - alpha) * self.mu + self.rng.normal(0.0, noise_sigma, self.n_buses)
        # 局部漂移
        self.V += self.local_drifts * 0.1
        # 振荡效应
        t = self.event_counter * 0.12
        oscillations = self.oscillation_amps * np.sin(self.oscillation_freqs * t + self.oscillation_phases)
        self.V += oscillations
        # 事件影响
        for event in self.active_events[:]:
            progress = (self.event_counter - event['start_time']) / max(event['duration'], 1)
            if progress > 1.0:
                self.active_events.remove(event)
                continue
            if event['type'] == 'voltage_sag':
                impact = event['magnitude'] * (1 - abs(progress - 0.5) * 2)  # 钟形
                self.V[event['bus_indices']] -= impact
            elif event['type'] == 'voltage_swell':
                impact = event['magnitude'] * (1 - abs(progress - 0.5) * 2)
                self.V[event['bus_indices']] += impact
            elif event['type'] == 'oscillation':
                impact = event['magnitude'] * np.sin(progress * 4 * np.pi) * (1 - progress)
                self.V[event['bus_indices']] += impact
        self.V += self.g
        return self.V

# ------------------------------
# Demo 网络（当找不到自定义工程时）
# ------------------------------

def build_demo_net(seed: int = 7) -> pp.pandapowerNet:
    rng = np.random.RandomState(seed)
    net = pp.create_empty_network(sn_mva=100.)
    # 生成 1 个 110kV + 40 个 10kV 母线
    b_hv = pp.create_bus(net, vn_kv=110., name="HV")
    buses = [pp.create_bus(net, vn_kv=10., name=f"B{i:02d}") for i in range(40)]
    # 变压器 HV->10kV
    for i in range(2):
        lv = buses[i]
        pp.create_transformer_from_parameters(
            net, hv_bus=b_hv, lv_bus=lv, sn_mva=25., vn_hv_kv=110., vn_lv_kv=10.,
            vkr_percent=0.5, vk_percent=12., pfe_kw=30., i0_percent=0.05, shift_degree=0.)
    # 构建一个带分支的配电网
    for i in range(39):
        r_ohm_per_km = 0.32 + 0.05*rng.rand()
        x_ohm_per_km = 0.08 + 0.02*rng.rand()
        c_nf_per_km = 210.
        max_i_ka = 0.4
        length_km = 0.5 + 1.2*rng.rand()
        pp.create_line_from_parameters(net, from_bus=buses[i], to_bus=buses[i+1], length_km=length_km,
                                       r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km,
                                       c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka, name=f"L{i}-{i+1}")
    # 支路
    for i in range(5, 35, 6):
        j = min(i+3, 39)
        pp.create_line_from_parameters(net, from_bus=buses[i], to_bus=buses[j], length_km=0.6+0.7*rng.rand(),
                                       r_ohm_per_km=0.36, x_ohm_per_km=0.09, c_nf_per_km=200., max_i_ka=0.4,
                                       name=f"L{i}-{j}")
    # 负荷 & DG
    for i, b in enumerate(buses):
        pp.create_load(net, b, p_mw=0.2+0.6*rng.rand(), q_mvar=0.05+0.2*rng.rand(), name=f"LD{i:02d}")
        if rng.rand() < 0.25:
            pp.create_sgen(net, b, p_mw=0.05+0.25*rng.rand(), q_mvar=-0.02+0.04*rng.rand(), name=f"PV{i:02d}")
    # 源点
    pp.create_ext_grid(net, bus=b_hv, vm_pu=1.02, name="Grid")
    # 随机 geo
    G = nx.random_geometric_graph(len(net.bus), radius=0.35, seed=seed)
    pos = {n: (float(x), float(y)) for n, (x, y) in G.positions.items()} if hasattr(G, 'positions') else nx.spring_layout(G, seed=seed)
    net.bus["geo"] = None
    for idx, b in enumerate(net.bus.index):
        x, y = pos.get(idx, (np.cos(idx), np.sin(idx)))
        net.bus.at[b, "geo"] = {"type": "Point", "coordinates": [float(x), float(y)]}
    return net

# ------------------------------
# 随机化负荷/分布式电源 + 随机拓扑扰动
# ------------------------------

def randomize_loads_and_gens(net: pp.pandapowerNet, rng: np.random.RandomState,
                             base_scale: Tuple[float, float] = (0.85, 1.15),
                             step_sigma: float = 0.06) -> None:
    if hasattr(net, 'load') and len(net.load):
        s = rng.uniform(*base_scale)
        net.load['p_mw'] *= s * (1 + rng.normal(0, step_sigma, len(net.load)))
        net.load['q_mvar'] *= s * (1 + rng.normal(0, step_sigma, len(net.load)))
        net.load['p_mw'] = net.load['p_mw'].clip(lower=0.02)
    if hasattr(net, 'sgen') and len(net.sgen):
        s2 = rng.uniform(0.7, 1.25)
        net.sgen['p_mw'] *= s2 * (1 + rng.normal(0, step_sigma, len(net.sgen)))
        net.sgen['q_mvar'] *= (1 + rng.normal(0, step_sigma*0.7, len(net.sgen)))
        net.sgen['p_mw'] = net.sgen['p_mw'].clip(lower=0.)


def random_topology_perturbation(net: pp.pandapowerNet, rng: np.random.RandomState,
                                 frac_lines: float = 0.04, max_toggle: int = 6) -> List[int]:
    """随机将部分线路置为停运/恢复，返回被切换的线路 index 列表"""
    if not hasattr(net, 'line') or not len(net.line):
        return []
    cand = list(net.line.index)
    if not cand:
        return []
    k = min(max_toggle, max(1, int(len(cand) * frac_lines)))
    toggled = rng.choice(cand, size=k, replace=False)
    for lid in toggled:
        net.line.at[lid, 'in_service'] = not bool(net.line.at[lid, 'in_service']) if 'in_service' in net.line.columns else False
    return list(map(int, toggled))

# ------------------------------
# 静态层（节点底色用于电压等级/类别，可按需修改）
# ------------------------------

def build_static_layers(net: pp.pandapowerNet, geodf: pd.DataFrame,
                        bus_size_static=7, line_width=2):
    # 线路
    line_x, line_y = assemble_line_coords(net, geodf, only_in_service=True)
    line_trace = go.Scattergl(
        x=line_x, y=line_y,
        mode="lines",
        line=dict(width=line_width, color="rgba(100,100,120,0.85)"),
        hoverinfo="skip",
        name="Edges"
    )
    # 母线（底层颜色按电压等级/类型）
    bus_order, bx, by, bus_names, bus_kv = [], [], [], [], []
    for bus_idx in net.bus.index:
        if bus_idx in geodf.index:
            x, y = geodf.at[bus_idx, "x"], geodf.at[bus_idx, "y"]
            if pd.notna(x) and pd.notna(y):
                bus_order.append(bus_idx); bx.append(float(x)); by.append(float(y))
                bus_names.append(str(net.bus.at[bus_idx, "name"]))
                bus_kv.append(float(net.bus.at[bus_idx, "vn_kv"]))
    def base_color(bus_idx):
        try:
            v_level = net.bus.at[bus_idx, "vn_kv"]
        except Exception:
            return "gray"
        if v_level >= 110: return "red"
        elif v_level >= 35: return "orange"
        elif v_level >= 10: return "blue"
        else: return "gray"
    static_colors = [base_color(b) for b in bus_order]
    static_bus = go.Scattergl(
        x=bx, y=by,
        mode="markers",
        marker=dict(size=bus_size_static, color=static_colors, opacity=0.95, line=dict(width=0)),
        hovertext=[f"{bus_names[k]}<br>Vn: {bus_kv[k]:.0f} kV" for k in range(len(bus_order))],
        hoverinfo="text",
        showlegend=False,
        name="Buses"
    )
    return line_trace, static_bus, bus_order, bx, by, bus_names, bus_kv

# ------------------------------
# 实时视图（Dash）- 加强版
# ------------------------------

def run_live_grid_view(update_interval_ms=280,  # 更快更新
                       pf_refresh_every=8,       # 每 N 次 tick 做一次潮流 + 随机采样
                       cmin=0.94, cmax=1.06,
                       noise_sigma=0.017,       # 增大噪声
                       global_drift_sigma=0.0022,# 增大漂移
                       alpha=0.84,              # 更快响应
                       keep_aspect: bool = True,
                       node_scale: float = 1.4,
                       line_scale: float = 1.25,
                       color_gain: float = 2.0,  # 增强颜色对比
                       host="127.0.0.1", port=8050):
    from dash import Dash, html, dcc, Output, Input, State, callback_context

    # ====== 构建网络 ======
    if HAS_CUSTOM:
        parsed_cim = CimEParser(PfDataPath)
        pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim, slack_nd='703002137')
        feeder_cluster, base_net = init_feeder_net(pp_pf_calculator)
        calc_net = built_ppnet_for_pfcal(base_net)
    else:
        pp_pf_calculator = None
        base_net = build_demo_net()
        calc_net = base_net

    geodf, has_geo = ensure_geodf_topo(base_net, calc_net, keep_aspect=keep_aspect)
    (line_trace, static_bus, bus_order, bx, by, bus_names, bus_kv) = \
        build_static_layers(base_net, geodf, bus_size_static=7, line_width=2)

    # 自适应像素
    _auto_stat, _auto_dyn, _auto_lw = _autosize_markers(len(bus_order))
    static_bus.marker.size = int(_auto_stat * node_scale)
    line_trace.line.width = int(max(1, _auto_lw * line_scale))

    N = len(bus_order)
    rng = np.random.RandomState(42)

    # ====== 动态系统 ======
    dynamic_system = EnhancedDynamicSystem(N, seed=42)
    # 初始随机事件，拉大变化幅度
    for _ in range(3):
        bus_idx = rng.randint(0, N, size=rng.randint(4, 10))
        event_type = rng.choice(['voltage_sag', 'voltage_swell', 'oscillation'])
        dynamic_system.add_event(event_type, bus_idx,
                                 duration=rng.randint(40, 120),
                                 magnitude=rng.uniform(0.03, 0.06))

    def make_hover(v):
        return [f"{bus_names[k]}<br>Vn: {bus_kv[k]:.0f} kV<br>V: {v[k]:.3f} p.u." for k in range(N)]

    # 初始电压
    V0 = dynamic_system.update(alpha=alpha, noise_sigma=noise_sigma, global_drift_sigma=global_drift_sigma)
    V0_color = _amplify_for_color(V0, gain=color_gain, cmin=cmin, cmax=cmax)
    size0 = (6 + 140*np.abs(V0 - 1.0)).clip(6, 26)

    dynamic_bus = go.Scattergl(
        x=bx, y=by,
        mode="markers",
        marker=dict(
            size=size0,
            color=V0_color,
            cmin=cmin, cmax=cmax,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Voltage (p.u.)", thickness=16, x=1.02, xpad=10)
        ),
        hovertext=make_hover(V0),
        hoverinfo="text",
        name="电压（动态）"
    )

    base_fig = go.Figure(
        data=[line_trace, static_bus, dynamic_bus],
        layout=go.Layout(
            title=dict(text="实时配电网电压视图（拓扑+工况动态）", y=0.98, x=0.01),
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1], scaleanchor="x", scaleratio=1),
            showlegend=False,
            autosize=True,
            margin=dict(l=8, r=8, t=56, b=8),
            paper_bgcolor="#e9eef5",
            plot_bgcolor="#e9eef5",
            uirevision=True
        )
    )

    LINE_I, STATIC_I, DYN_I = 0, 1, 2

    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Button("轻载 (+0.02 pu)", id="btn_light", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("重载 (-0.03 pu)", id="btn_heavy", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("电压暂降事件", id="btn_sag", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("电压暂升事件", id="btn_swell", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("振荡事件", id="btn_osc", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("拓扑拨动 + 潮流", id="btn_switch_pf", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("重置", id="btn_reset", n_clicks=0),
            dcc.RadioItems(
                id="mode",
                options=[{"label": "随机采样", "value": "random"}, {"label": "潮流驱动", "value": "pf"}],
                value="pf", inline=True,
                style={"marginLeft": "16px", "display": "inline-block"}
            )
        ], style={"padding": "6px 0"}),
        html.Div(id="summary", style={"fontSize": "14px", "padding": "4px 2px", "color": "#333"}),
        dcc.Graph(
            id="grid_fig",
            figure=base_fig,
            animate=False,
            style={"width": "100%", "height": "90vh"},
            config={"responsive": True, "displayModeBar": True}
        ),
        dcc.Interval(id="tick", interval=update_interval_ms, n_intervals=0),
        dcc.Store(id="store_state",
                  data={"dynamic_system": {
                      "V": dynamic_system.V.tolist(),
                      "mu": dynamic_system.mu.tolist(),
                      "g": dynamic_system.g,
                      "local_drifts": dynamic_system.local_drifts.tolist(),
                      "event_counter": dynamic_system.event_counter,
                      "active_events": dynamic_system.active_events
                  }, "tick_count": 0})
    ])

    # ========== Tick 更新 ==========
    @app.callback(
        Output("grid_fig", "figure"),
        Output("store_state", "data"),
        Output("summary", "children"),
        Input("tick", "n_intervals"),
        State("grid_fig", "figure"),
        State("store_state", "data"),
        State("mode", "value"),
        prevent_initial_call=False
    )
    def on_tick(n, fig_json, st, mode):
        # 恢复动态系统状态
        ds_data = st["dynamic_system"]
        dynamic_system.V = np.array(ds_data["V"], dtype=float)
        dynamic_system.mu = np.array(ds_data["mu"], dtype=float)
        dynamic_system.g = ds_data["g"]
        dynamic_system.local_drifts = np.array(ds_data["local_drifts"], dtype=float)
        dynamic_system.event_counter = ds_data["event_counter"]
        dynamic_system.active_events = ds_data["active_events"]

        # 每 pf_refresh_every 次进行一次：随机负荷/发电 -> 潮流 -> 以结果更新 mu（更真实）
        st["tick_count"] = int(st.get("tick_count", 0)) + 1
        do_pf = (st["tick_count"] % pf_refresh_every == 0)
        if mode == "pf" and do_pf:
            # 适度拓扑扰动（小概率）
            if rng.rand() < 0.35:
                toggled = random_topology_perturbation(base_net, rng, frac_lines=0.05, max_toggle=8)
                # 若导致潮流失败将回滚
            # 负荷/分布式电源随机采样
            randomize_loads_and_gens(base_net, rng, base_scale=(0.85, 1.20), step_sigma=0.05)
            try:
                pp.runpp(base_net, max_iteration=60, tolerance_mva=1e-5, enforce_q_lims=True, numba=False)
                name2vm = dict(zip(
                    base_net.bus["name"].astype(str).tolist(),
                    base_net.res_bus["vm_pu"].astype(float).tolist()
                ))
                new_mu = np.array([name2vm.get(str(nm), np.nan) for nm in base_net.bus["name"].astype(str)], dtype=float)
                bad = ~np.isfinite(new_mu); new_mu[bad] = dynamic_system.mu[bad]
                dynamic_system.mu = np.clip(new_mu, cmin, cmax)
            except Exception:
                # 回滚拓扑
                if hasattr(base_net, 'line') and len(base_net.line) and 'in_service' in base_net.line.columns:
                    base_net.line['in_service'] = True
        elif mode == "random" and do_pf:
            # 无潮流版本：直接给 mu 一个新的随机目标（更明显）
            dynamic_system.mu = np.clip(1.0 + rng.normal(0, 0.018, N), cmin, cmax)

        # 随机添加新事件（更明显）
        if rng.random() < 0.08:
            bus_idx = rng.randint(0, N, size=rng.randint(4, 12))
            event_type = rng.choice(['voltage_sag', 'voltage_swell', 'oscillation'])
            dynamic_system.add_event(event_type, bus_idx,
                                     duration=rng.randint(50, 140),
                                     magnitude=rng.uniform(0.035, 0.07))

        # 更新系统
        V = dynamic_system.update(alpha=alpha, noise_sigma=noise_sigma, global_drift_sigma=global_drift_sigma)
        V_color = _amplify_for_color(V, gain=color_gain, cmin=cmin, cmax=cmax)
        sizes = (6 + 155*np.abs(V - 1.0)).clip(6, 28)

        # 更新图
        fig = go.Figure(fig_json)
        # 线路根据 in_service 重新拼装（模拟开关动作）
        lx, ly = assemble_line_coords(base_net, geodf, only_in_service=True)
        fig.data[LINE_I].x = lx
        fig.data[LINE_I].y = ly
        # 动态节点
        fig.data[DYN_I].marker.color = V_color
        fig.data[DYN_I].marker.size = sizes
        fig.data[DYN_I].hovertext = [f"{bus_names[k]}<br>Vn: {bus_kv[k]:.0f} kV<br>V: {V[k]:.3f} p.u." for k in range(len(V))]

        # 摘要信息（更直观）
        vmin, vmax, vavg = float(np.min(V)), float(np.max(V)), float(np.mean(V))
        cnt_low = int(np.sum(V < 0.97))
        cnt_high = int(np.sum(V > 1.03))
        summary = f"Vmin={vmin:.3f}  Vavg={vavg:.3f}  Vmax={vmax:.3f}  |  <0.97: {cnt_low}  >1.03: {cnt_high}  | 模式: {'潮流驱动' if mode=='pf' else '随机采样'}"

        # 更新存储状态
        st["dynamic_system"] = {
            "V": dynamic_system.V.tolist(),
            "mu": dynamic_system.mu.tolist(),
            "g": dynamic_system.g,
            "local_drifts": dynamic_system.local_drifts.tolist(),
            "event_counter": dynamic_system.event_counter,
            "active_events": dynamic_system.active_events
        }
        return fig, st, summary

    # ========== 按钮事件 ==========
    @app.callback(
        Output("store_state", "data"),
        Input("btn_light", "n_clicks"),
        Input("btn_heavy", "n_clicks"),
        Input("btn_sag", "n_clicks"),
        Input("btn_swell", "n_clicks"),
        Input("btn_osc", "n_clicks"),
        Input("btn_switch_pf", "n_clicks"),
        Input("btn_reset", "n_clicks"),
        State("store_state", "data"),
        prevent_initial_call=True
    )
    def on_buttons(n_light, n_heavy, n_sag, n_swell, n_osc, n_switch_pf, n_reset, st):
        if not callback_context.triggered:
            return st
        which = callback_context.triggered[0]["prop_id"].split(".")[0]
        ds_data = st["dynamic_system"]
        dynamic_system.V = np.array(ds_data["V"], dtype=float)
        dynamic_system.mu = np.array(ds_data["mu"], dtype=float)
        dynamic_system.g = ds_data["g"]
        dynamic_system.local_drifts = np.array(ds_data["local_drifts"], dtype=float)
        dynamic_system.event_counter = ds_data["event_counter"]
        dynamic_system.active_events = ds_data["active_events"]

        if which == "btn_light":
            dynamic_system.mu = np.clip(dynamic_system.mu + 0.02, cmin, cmax)
            affected_buses = np.arange(len(dynamic_system.mu))
            dynamic_system.add_event('voltage_swell', affected_buses, duration=90, magnitude=0.045)
        elif which == "btn_heavy":
            dynamic_system.mu = np.clip(dynamic_system.mu - 0.03, cmin, cmax)
            affected_buses = np.arange(len(dynamic_system.mu))
            dynamic_system.add_event('voltage_sag', affected_buses, duration=120, magnitude=0.055)
        elif which == "btn_sag":
            affected_buses = np.random.RandomState(0).choice(len(dynamic_system.mu), size=min(12, len(dynamic_system.mu)), replace=False)
            dynamic_system.add_event('voltage_sag', affected_buses, duration=80, magnitude=0.065)
        elif which == "btn_swell":
            affected_buses = np.random.RandomState(1).choice(len(dynamic_system.mu), size=min(10, len(dynamic_system.mu)), replace=False)
            dynamic_system.add_event('voltage_swell', affected_buses, duration=90, magnitude=0.055)
        elif which == "btn_osc":
            affected_buses = np.random.RandomState(2).choice(len(dynamic_system.mu), size=min(8, len(dynamic_system.mu)), replace=False)
            dynamic_system.add_event('oscillation', affected_buses, duration=110, magnitude=0.05)
        elif which == "btn_switch_pf":
            # 触发一次明显的拓扑拨动 + 潮流
            random_topology_perturbation(base_net, np.random.RandomState(), frac_lines=0.08, max_toggle=12)
            try:
                pp.runpp(base_net, max_iteration=60, tolerance_mva=1e-5, enforce_q_lims=True, numba=False)
                name2vm = dict(zip(
                    base_net.bus["name"].astype(str).tolist(),
                    base_net.res_bus["vm_pu"].astype(float).tolist()
                ))
                new_mu = np.array([name2vm.get(str(nm), np.nan) for nm in base_net.bus["name"].astype(str)], dtype=float)
                bad = ~np.isfinite(new_mu); new_mu[bad] = dynamic_system.mu[bad]
                dynamic_system.mu = np.clip(new_mu, cmin, cmax)
                dynamic_system.add_event('oscillation', list(range(len(dynamic_system.mu))), duration=70, magnitude=0.04)
            except Exception:
                if hasattr(base_net, 'line') and len(base_net.line) and 'in_service' in base_net.line.columns:
                    base_net.line['in_service'] = True
        elif which == "btn_reset":
            dynamic_system.mu = np.ones_like(dynamic_system.mu)
            dynamic_system.active_events = []
            dynamic_system.g = 0.0
            if hasattr(base_net, 'line') and len(base_net.line) and 'in_service' in base_net.line.columns:
                base_net.line['in_service'] = True

        st["dynamic_system"] = {
            "V": dynamic_system.V.tolist(),
            "mu": dynamic_system.mu.tolist(),
            "g": dynamic_system.g,
            "local_drifts": dynamic_system.local_drifts.tolist(),
            "event_counter": dynamic_system.event_counter,
            "active_events": dynamic_system.active_events
        }
        return st

    print(f"Dash 实时视图启动： http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


# ------------------------------
# 示例入口
# ------------------------------
if __name__ == "__main__":
    run_live_grid_view(
        update_interval_ms=280,     # 更快更新
        pf_refresh_every=8,         # 每 8 个 tick 触发一次“更真实”的工况采样 + 潮流
        cmin=0.94, cmax=1.06,
        noise_sigma=0.017,          # 增加噪声
        global_drift_sigma=0.0022,  # 增加漂移
        alpha=0.84,                 # 更快响应
        keep_aspect=True,
        node_scale=1.4,
        line_scale=1.25,
        color_gain=2.0,             # 增强颜色对比
        host="0.0.0.0", port=8050
    )
# %%
