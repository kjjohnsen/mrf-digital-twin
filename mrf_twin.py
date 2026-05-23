#!/usr/bin/env python3
"""
MRF Digital Twin
================
A discrete-event simulation of a Materials Recovery Facility (MRF) processing
single-stream curbside recyclables, written in pure open-source Python as an
illustration that material flow analysis / facility digital twins can be built
with SimPy + Plotly rather than commercial platforms like AnyLogic.

Models one operating day:
  - Stochastic truck arrivals (Poisson) dumping mixed recyclables on a tipping floor
  - A sequential sorting line: pre-sort, OCC screen, paper screen, glass breaker,
    magnet, eddy-current separator, optical sorters (PET, HDPE), manual QC
  - Per-station recovery rates and cross-contamination
  - Stochastic equipment breakdowns and repairs
  - Mass balance tracking (every gram of inbound material is accounted for)

Outputs a self-contained HTML dashboard with:
  - Animated cumulative-output bars (scrub through the day)
  - Sankey diagram of full-day mass flow
  - Tipping-floor inventory and cumulative throughput time series
  - Gantt-style equipment downtime timeline
  - Recovery and utilization summary tables

Run:
    python mrf_twin.py
"""

from __future__ import annotations

import random
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import plotly.graph_objects as go
import simpy
from plotly.io import to_html
from plotly.subplots import make_subplots


# ============================================================
# Configuration
# ============================================================

# Mass fractions of incoming single-stream recyclables (representative US curbside)
INBOUND_COMPOSITION: dict[str, float] = {
    "OCC":         0.20,   # corrugated cardboard
    "mixed_paper": 0.25,
    "PET":         0.06,
    "HDPE":        0.04,
    "aluminum":    0.02,
    "steel":       0.03,
    "glass":       0.18,
    "residue":     0.22,   # film, food waste, non-recyclables
}

# Sequential sorting line. Each station: (name, throughput_tph, {target: recovery_rate})
PROCESS_LINE: list[tuple[str, float, dict[str, float]]] = [
    ("pre_sort",      40, {"residue": 0.30}),
    ("OCC_screen",    35, {"OCC": 0.92}),
    ("paper_screen",  35, {"mixed_paper": 0.85}),
    ("glass_breaker", 35, {"glass": 0.75}),
    ("magnet",        35, {"steel": 0.95}),
    ("eddy_current",  35, {"aluminum": 0.88}),
    ("optical_PET",   30, {"PET": 0.90}),
    ("optical_HDPE",  30, {"HDPE": 0.85}),
    ("manual_QC",     22, {"residue": 0.55}),
]
STATIONS: list[str] = [s[0] for s in PROCESS_LINE]

# Fraction of non-target material that gets pulled into a target bale (impurity)
CONTAMINATION_RATE = 0.03

# Daily schedule (minutes from midnight)
DAY_START = 7 * 60   #  7:00 AM
SHIFT_END = 15 * 60  #  3:00 PM  (last truck arrives)
SIM_END   = 19 * 60  #  7:00 PM  (line wraps up residual material)

# Truck arrivals
TRUCK_INTERARRIVAL_MIN = 16.0
TRUCK_MEAN_TONS = 8.0
TRUCK_SD_TONS = 1.5

# Equipment reliability
ENABLE_BREAKDOWNS = True
MTBF_MIN = 220.0   # mean time between failures (minutes)
MTTR_MIN = 12.0    # mean time to repair

# Snapshot cadence for the dashboard animation
SAMPLE_INTERVAL_MIN = 5.0

RANDOM_SEED = 42

# Colors for each material
MATERIAL_COLORS = {
    "OCC":         "#8B4513",
    "mixed_paper": "#DEB887",
    "PET":         "#4682B4",
    "HDPE":        "#FF8C00",
    "aluminum":    "#C0C0C0",
    "steel":       "#708090",
    "glass":       "#2E8B57",
    "residue":     "#696969",
}

OUTPUT_HTML = "mrf_dashboard.html"


# ============================================================
# Domain objects
# ============================================================

@dataclass
class TruckLoad:
    """A single truck's payload of mixed recyclables flowing through the MRF."""
    id: int
    arrival_time: float
    initial_mass: float
    composition: dict[str, float] = field(default_factory=dict)
    # Sequence of (time, location) — locations are "tipping_floor", station names, or "done".
    # Used by the factory-layout animation to position the load over time.
    history: list[tuple[float, str]] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return sum(max(0.0, v) for v in self.composition.values())


# ============================================================
# Simulation
# ============================================================

class MRF:
    """Discrete-event model of the facility."""

    def __init__(self, env: simpy.Environment, seed: int = RANDOM_SEED):
        self.env = env
        self.rng = random.Random(seed)

        # Each sorting station processes one load at a time. PriorityResource
        # lets breakdown requests jump ahead of waiting loads.
        self.resources = {n: simpy.PriorityResource(env, capacity=1) for n in STATIONS}
        self.station_running = {n: True for n in STATIONS}

        # Counters
        self.tipping_floor_tons = 0.0
        self.total_inbound = 0.0
        self.recovered: dict[str, float] = defaultdict(float)
        self.residue_to_landfill = 0.0
        self.loads_arrived = 0
        self.loads_completed = 0

        # Utilization tracking (minutes)
        self.station_busy = {n: 0.0 for n in STATIONS}
        self.station_downtime = {n: 0.0 for n in STATIONS}

        # Event logs
        self.status_log: list[tuple[float, str, str]] = []
        self.snapshots: list[dict] = []
        self.flow_edges: dict[tuple[str, str], float] = defaultdict(float)
        self.loads: list[TruckLoad] = []  # every truckload, for factory-layout animation

        # Background processes
        env.process(self._truck_arrivals())
        env.process(self._telemetry())
        if ENABLE_BREAKDOWNS:
            for name in STATIONS:
                env.process(self._breakdown_cycle(name))

    # ----- generators -----

    def _truck_arrivals(self):
        """Trucks arrive as a Poisson process until end-of-shift."""
        load_id = 0
        while True:
            yield self.env.timeout(self.rng.expovariate(1.0 / TRUCK_INTERARRIVAL_MIN))
            if self.env.now >= SHIFT_END:
                return
            load_id += 1
            tons = max(2.0, self.rng.gauss(TRUCK_MEAN_TONS, TRUCK_SD_TONS))
            # Per-truck variation in composition
            raw = {m: frac * self.rng.uniform(0.85, 1.15)
                   for m, frac in INBOUND_COMPOSITION.items()}
            norm = sum(raw.values())
            composition = {m: tons * v / norm for m, v in raw.items()}
            load = TruckLoad(load_id, self.env.now, tons, composition)
            load.history.append((self.env.now, "tipping_floor"))
            self.loads.append(load)
            self.tipping_floor_tons += tons
            self.total_inbound += tons
            self.loads_arrived += 1
            self.env.process(self._process_load(load))

    def _process_load(self, load: TruckLoad):
        """Push a load through the sorting line in sequence."""
        prev_node = "Inbound"
        is_first = True
        for station_name, tph, recoveries in PROCESS_LINE:
            with self.resources[station_name].request(priority=10) as req:
                yield req
                load.history.append((self.env.now, station_name))

                # When the first station starts, take the load off the tipping floor
                if is_first:
                    self.tipping_floor_tons -= load.initial_mass
                    is_first = False

                mass_at_entry = load.mass
                self.flow_edges[(prev_node, station_name)] += mass_at_entry

                if mass_at_entry > 0:
                    duration = (mass_at_entry / tph) * 60.0
                    self.station_busy[station_name] += duration
                    yield self.env.timeout(duration)

                    for material, recovery in recoveries.items():
                        available = max(0.0, load.composition.get(material, 0.0))
                        extracted = available * recovery
                        load.composition[material] = available - extracted

                        # Cross-contamination from other materials
                        contam = 0.0
                        for other, amt in list(load.composition.items()):
                            if other == material or amt <= 0:
                                continue
                            pull = amt * CONTAMINATION_RATE * recovery
                            contam += pull
                            load.composition[other] = amt - pull

                        bale_mass = extracted + contam
                        if bale_mass <= 0:
                            continue
                        if material == "residue":
                            self.residue_to_landfill += bale_mass
                            self.flow_edges[(station_name, "Landfill")] += bale_mass
                        else:
                            self.recovered[material] += bale_mass
                            self.flow_edges[(station_name, f"Bale: {material}")] += bale_mass

                prev_node = station_name

        # Anything left in the load after the last station is residue
        leftover = load.mass
        if leftover > 0:
            self.residue_to_landfill += leftover
            self.flow_edges[(prev_node, "Landfill")] += leftover

        load.history.append((self.env.now, "done"))
        self.loads_completed += 1

    def _breakdown_cycle(self, station_name: str):
        """Random failures modeled as the station resource being held by a 'repair' task."""
        while True:
            yield self.env.timeout(self.rng.expovariate(1.0 / MTBF_MIN))
            with self.resources[station_name].request(priority=1) as req:
                yield req
                t0 = self.env.now
                self.station_running[station_name] = False
                self.status_log.append((t0, station_name, "broken"))
                ttr = self.rng.expovariate(1.0 / MTTR_MIN)
                yield self.env.timeout(ttr)
                self.station_downtime[station_name] += self.env.now - t0
                self.station_running[station_name] = True
                self.status_log.append((self.env.now, station_name, "running"))

    def _telemetry(self):
        """Snapshot facility state at fixed intervals to power the dashboard animation."""
        while True:
            self.snapshots.append({
                "t": self.env.now,
                "tipping_floor": self.tipping_floor_tons,
                "recovered": dict(self.recovered),
                "residue": self.residue_to_landfill,
                "stations": dict(self.station_running),
                "loads_arrived": self.loads_arrived,
                "loads_completed": self.loads_completed,
            })
            yield self.env.timeout(SAMPLE_INTERVAL_MIN)


def run_simulation(seed: int = RANDOM_SEED) -> MRF:
    env = simpy.Environment(initial_time=DAY_START)
    mrf = MRF(env, seed=seed)
    env.run(until=SIM_END)
    return mrf


# ============================================================
# Visualization
# ============================================================

def hhmm(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_sankey(mrf: MRF) -> go.Figure:
    ordered = (
        ["Inbound"]
        + STATIONS
        + [f"Bale: {m}" for m in INBOUND_COMPOSITION if m != "residue"]
        + ["Landfill"]
    )
    appearing = {n for edge in mrf.flow_edges for n in edge}
    nodes = [n for n in ordered if n in appearing]
    idx = {n: i for i, n in enumerate(nodes)}

    sources, targets, values, link_colors = [], [], [], []
    for (a, b), v in mrf.flow_edges.items():
        if v <= 0:
            continue
        sources.append(idx[a]); targets.append(idx[b]); values.append(v)
        if b.startswith("Bale: "):
            link_colors.append(_hex_rgba(MATERIAL_COLORS.get(b.split(": ", 1)[1], "#888888"), 0.55))
        elif b == "Landfill":
            link_colors.append("rgba(80,80,80,0.55)")
        else:
            link_colors.append("rgba(120,140,170,0.30)")

    node_colors = []
    for n in nodes:
        if n == "Inbound":
            node_colors.append("#333333")
        elif n.startswith("Bale: "):
            node_colors.append(MATERIAL_COLORS.get(n.split(": ", 1)[1], "#888"))
        elif n == "Landfill":
            node_colors.append("#1f1f1f")
        else:
            node_colors.append("#5b8fb9")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=18, color=node_colors,
                  line=dict(color="#222", width=0.5)),
        link=dict(source=sources, target=targets, value=values, color=link_colors,
                  hovertemplate="%{source.label} -> %{target.label}<br>%{value:.2f} tons<extra></extra>"),
    ))
    fig.update_layout(title="Material flow — full-day mass balance",
                      height=560, margin=dict(t=60, l=20, r=20, b=20))
    return fig


def build_inventory_chart(mrf: MRF) -> go.Figure:
    times = [s["t"] / 60.0 for s in mrf.snapshots]
    tipping = [s["tipping_floor"] for s in mrf.snapshots]
    total_baled = [sum(s["recovered"].values()) + s["residue"] for s in mrf.snapshots]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=times, y=tipping, name="Tipping floor (WIP)",
        line=dict(color="#d62728", width=2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=times, y=total_baled, name="Cumulative output",
        line=dict(color="#2ca02c", width=2.5),
    ), secondary_y=True)
    fig.update_xaxes(title_text="Hour of day")
    fig.update_yaxes(title_text="Tons on tipping floor", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Cumulative tons processed", secondary_y=True)
    fig.update_layout(
        title="Tipping floor inventory vs. cumulative throughput",
        height=400, margin=dict(t=70, l=60, r=60, b=60),
        legend=dict(orientation="h", x=0.0, y=1.15),
    )
    return fig


def build_gantt(mrf: MRF) -> go.Figure:
    fig = go.Figure()
    any_down = False
    for st in STATIONS:
        events = sorted([(t, s) for (t, n, s) in mrf.status_log if n == st])
        in_repair = None
        for t, status in events:
            if status == "broken":
                in_repair = t
            elif status == "running" and in_repair is not None:
                any_down = True
                fig.add_trace(go.Bar(
                    base=in_repair / 60.0, x=[(t - in_repair) / 60.0], y=[st],
                    orientation="h", marker_color="#d62728",
                    hovertemplate=f"{st}<br>{hhmm(in_repair)} – {hhmm(t)}"
                                  f" ({(t - in_repair):.1f} min)<extra></extra>",
                    showlegend=False,
                ))
                in_repair = None
    if not any_down:
        fig.add_annotation(text="No breakdowns this run",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(
        title="Equipment downtime timeline (each bar = a breakdown)",
        xaxis=dict(title="Hour of day", range=[DAY_START / 60.0, SIM_END / 60.0]),
        yaxis=dict(title="Station", categoryorder="array", categoryarray=STATIONS[::-1]),
        height=400, margin=dict(t=70, l=130, r=20, b=60), barmode="overlay",
    )
    return fig


def build_factory_layout(mrf: MRF) -> go.Figure:
    """Animated top-down 2D layout of the facility showing material flow over the day.

    Stations sit on a horizontal conveyor. Truck loads appear as dots that move from
    station to station as they're processed. Bale piles above each station grow as
    material accumulates. The tipping-floor pile (left) and landfill pile (below)
    expand and contract in real time.
    """
    # ---- positions ----
    TIPPING_POS = (0.4, 3.0)
    OUTPUT_POS = (10.6, 3.0)
    POS = {
        "pre_sort":      (1.6, 3.0),
        "OCC_screen":    (2.6, 3.0),
        "paper_screen":  (3.6, 3.0),
        "glass_breaker": (4.6, 3.0),
        "magnet":        (5.6, 3.0),
        "eddy_current":  (6.6, 3.0),
        "optical_PET":   (7.6, 3.0),
        "optical_HDPE":  (8.6, 3.0),
        "manual_QC":     (9.6, 3.0),
    }
    BALE_POS = {
        "OCC":         (2.6, 5.0),
        "mixed_paper": (3.6, 5.0),
        "glass":       (4.6, 5.0),
        "steel":       (5.6, 5.0),
        "aluminum":    (6.6, 5.0),
        "PET":         (7.6, 5.0),
        "HDPE":        (8.6, 5.0),
    }
    LANDFILL_POS = (5.6, 1.0)

    # Map station -> non-residue material it extracts (for chute drawing)
    extracts: dict[str, str] = {}
    for st_name, _, recoveries in PROCESS_LINE:
        for material in recoveries:
            if material != "residue":
                extracts[st_name] = material
                break

    # ---- status timeline lookup ----
    status_timeline: dict[str, list[tuple[float, str]]] = {
        st: [(DAY_START, "running")] for st in STATIONS
    }
    for t, n, s in mrf.status_log:
        status_timeline[n].append((t, s))

    def station_status_at(st: str, t: float) -> str:
        last = "running"
        for et, s in status_timeline[st]:
            if et <= t:
                last = s
            else:
                break
        return last

    def load_location_at(load: TruckLoad, t: float) -> str | None:
        if t < load.arrival_time:
            return None
        loc = "tipping_floor"
        for entry_t, entry_loc in load.history:
            if entry_t <= t:
                loc = entry_loc
            else:
                break
        return None if loc == "done" else loc

    # pre-index snapshots by ascending time for fast lookup
    snapshot_times = [s["t"] for s in mrf.snapshots]

    def snapshot_at(t: float) -> dict:
        idx = max((i for i, st in enumerate(snapshot_times) if st <= t), default=0)
        return mrf.snapshots[idx]

    # ---- static layout: conveyor belt, chutes, labels ----
    shapes: list[dict] = []
    annotations: list[dict] = []

    # Conveyor belt (thick rectangle)
    shapes.append(dict(
        type="rect",
        x0=TIPPING_POS[0] - 0.2, y0=2.82, x1=OUTPUT_POS[0] + 0.2, y1=3.18,
        fillcolor="#b0b0b0", line=dict(color="#555", width=1), layer="below",
    ))
    # Direction arrows along the belt
    for x in (1.0, 4.0, 7.0, 10.0):
        annotations.append(dict(
            x=x + 0.35, y=3.0, ax=x - 0.05, ay=3.0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2,
            arrowcolor="#444", standoff=0, startstandoff=0,
        ))

    # Output chutes: station -> bale (dashed line going up)
    for st_name, material in extracts.items():
        sx, sy = POS[st_name]
        bx, by = BALE_POS[material]
        shapes.append(dict(
            type="line", x0=sx, y0=sy + 0.30, x1=bx, y1=by - 0.30,
            line=dict(color="#aaa", width=2, dash="dot"), layer="below",
        ))

    # Residue chutes: pre_sort and manual_QC -> landfill
    for st_name in ("pre_sort", "manual_QC"):
        sx, sy = POS[st_name]
        lx, ly = LANDFILL_POS
        shapes.append(dict(
            type="line", x0=sx, y0=sy - 0.30, x1=sx, y1=ly + 0.25,
            line=dict(color="#888", width=2, dash="dot"), layer="below",
        ))
        if sx != lx:
            shapes.append(dict(
                type="line", x0=sx, y0=ly + 0.05, x1=lx, y1=ly + 0.05,
                line=dict(color="#888", width=2, dash="dot"), layer="below",
            ))

    # Tipping floor feed chute (pile -> conveyor entry)
    shapes.append(dict(
        type="line",
        x0=TIPPING_POS[0] + 0.05, y0=TIPPING_POS[1] - 0.05,
        x1=POS["pre_sort"][0] - 0.30, y1=POS["pre_sort"][1],
        line=dict(color="#888", width=2, dash="dot"), layer="below",
    ))

    # Station labels (white text over the squares)
    for st in STATIONS:
        x, y = POS[st]
        annotations.append(dict(
            x=x, y=y, xref="x", yref="y",
            text=st.replace("_", "<br>"), showarrow=False,
            font=dict(size=8, color="white"),
            xanchor="center", yanchor="middle",
        ))

    def pretty_material(m: str) -> str:
        # Keep uppercase abbreviations (OCC, PET, HDPE) as-is; capitalize the rest
        return m if m.isupper() else m.replace("_", " ").capitalize()

    # Bale name labels (above each pile, with extra clearance for the tonnage line below)
    for material, (bx, by) in BALE_POS.items():
        annotations.append(dict(
            x=bx, y=by + 1.0, xref="x", yref="y",
            text=f"<b>{pretty_material(material)}</b>", showarrow=False,
            font=dict(size=10, color="#333"),
        ))

    # Tipping floor label (positioned well above the pile so it doesn't clash with
    # the pile's tonnage readout when the pile gets large)
    annotations.append(dict(
        x=TIPPING_POS[0], y=TIPPING_POS[1] + 1.35, xref="x", yref="y",
        text="<b>Tipping floor</b>", showarrow=False,
        font=dict(size=10, color="#333"), align="center",
    ))
    # Landfill label
    annotations.append(dict(
        x=LANDFILL_POS[0], y=LANDFILL_POS[1] - 0.55, xref="x", yref="y",
        text="<b>Landfill (residue)</b>", showarrow=False,
        font=dict(size=10, color="#333"),
    ))

    # ---- dynamic data builders ----
    station_x_arr = [POS[s][0] for s in STATIONS]
    station_y_arr = [POS[s][1] for s in STATIONS]
    bale_materials = list(BALE_POS.keys())
    bale_x_arr = [BALE_POS[m][0] for m in bale_materials]
    bale_y_arr = [BALE_POS[m][1] for m in bale_materials]
    bale_color_arr = [MATERIAL_COLORS[m] for m in bale_materials]

    def stations_colors(t: float) -> list[str]:
        return ["#2ecc71" if station_status_at(st, t) == "running" else "#e74c3c"
                for st in STATIONS]

    # Pre-allocate one marker slot per load so each load has a stable index across
    # frames. Plotly's frame transition interpolates marker positions by index — if
    # the array shrank when a load finished, every remaining load would shift one
    # slot and appear to slide backward.
    total_loads = len(mrf.loads)

    def loads_xy(t: float) -> tuple[list, list, list[str]]:
        xs: list = [None] * total_loads
        ys: list = [None] * total_loads
        txts: list[str] = [""] * total_loads
        for i, load in enumerate(mrf.loads):
            loc = load_location_at(load, t)
            if loc is None:
                continue
            # Each load gets a permanent offset within its current station's box,
            # keyed by load.id, so it doesn't visually jump when neighbors leave.
            if loc == "tipping_floor":
                base = TIPPING_POS
                col = load.id % 4
                row = (load.id // 4) % 3
                xs[i] = base[0] - 0.18 + col * 0.12
                ys[i] = base[1] + 0.35 - row * 0.13
            elif loc in POS:
                base = POS[loc]
                col = load.id % 2
                row = (load.id // 2) % 3
                xs[i] = base[0] - 0.12 + col * 0.24
                ys[i] = base[1] - 0.50 - row * 0.18
            else:
                continue
            txts[i] = f"Load #{load.id} &middot; {loc}"
        return xs, ys, txts

    def bales_sizes_texts(t: float) -> tuple[list[float], list[str]]:
        snap = snapshot_at(t)
        sizes, texts = [], []
        for material in bale_materials:
            amount = snap["recovered"].get(material, 0.0)
            sizes.append(20 + amount * 1.1)
            texts.append(f"{amount:.1f}t")
        return sizes, texts

    def tipping_size_text(t: float) -> tuple[float, str]:
        snap = snapshot_at(t)
        tip = max(0.0, snap["tipping_floor"])
        return 24 + tip * 2.4, f"{tip:.1f}t"

    def landfill_size_text(t: float) -> tuple[float, str]:
        snap = snapshot_at(t)
        res = snap["residue"]
        return 22 + res * 0.7, f"{res:.1f}t"

    # ---- initial trace data (at simulation start) ----
    t0 = mrf.snapshots[0]["t"]
    st_colors0 = stations_colors(t0)
    lx0, ly0, lt0 = loads_xy(t0)
    bs0, bt0 = bales_sizes_texts(t0)
    tsize0, ttext0 = tipping_size_text(t0)
    fsize0, ftext0 = landfill_size_text(t0)

    stations_trace = go.Scatter(
        x=station_x_arr, y=station_y_arr, mode="markers",
        marker=dict(symbol="square", size=46, color=st_colors0,
                    line=dict(color="black", width=1.2)),
        hovertext=STATIONS, hoverinfo="text",
        showlegend=False, name="stations",
    )
    loads_trace = go.Scatter(
        x=lx0, y=ly0, mode="markers",
        marker=dict(symbol="circle", size=11, color="#f39c12",
                    line=dict(color="black", width=0.6)),
        hovertext=lt0, hoverinfo="text",
        showlegend=False, name="loads",
    )
    # Bale squares: markers only — tonnage text on small piles overflowed.
    bales_trace = go.Scatter(
        x=bale_x_arr, y=bale_y_arr, mode="markers",
        marker=dict(symbol="square", size=bs0, color=bale_color_arr,
                    line=dict(color="black", width=1)),
        hoverinfo="skip", showlegend=False, name="bales",
    )
    # Bale tonnage labels at a fixed y just above the squares, independent of size.
    bale_tonnage_y_arr = [by + 0.55 for (_, by) in BALE_POS.values()]
    bale_tonnages_trace = go.Scatter(
        x=bale_x_arr, y=bale_tonnage_y_arr, mode="text",
        text=bt0, textfont=dict(size=10, color="#333"),
        hoverinfo="skip", showlegend=False, name="bale_tonnages",
    )
    tipping_trace = go.Scatter(
        x=[TIPPING_POS[0]], y=[TIPPING_POS[1] + 0.3], mode="markers+text",
        marker=dict(symbol="triangle-up", size=tsize0, color="#7f8c8d",
                    line=dict(color="black", width=1)),
        text=[ttext0], textposition="top center",
        textfont=dict(size=10, color="#333"),
        hoverinfo="skip", showlegend=False, name="tipping",
    )
    landfill_trace = go.Scatter(
        x=[LANDFILL_POS[0]], y=[LANDFILL_POS[1]], mode="markers+text",
        marker=dict(symbol="square", size=fsize0, color=MATERIAL_COLORS["residue"],
                    line=dict(color="black", width=1)),
        text=[ftext0], textposition="middle center",
        textfont=dict(size=9, color="white"),
        hoverinfo="skip", showlegend=False, name="landfill",
    )

    def title_for(t: float) -> str:
        snap = snapshot_at(t)
        baled = sum(snap["recovered"].values())
        tip = max(0.0, snap["tipping_floor"])
        return (f"<b>{hhmm(t)}</b>   "
                f"trucks arrived: {snap['loads_arrived']}   "
                f"loads completed: {snap['loads_completed']}   "
                f"tipping floor: {tip:.1f}t   "
                f"baled: {baled:.1f}t   residue: {snap['residue']:.1f}t")

    # ---- frames ----
    frames: list[go.Frame] = []
    for snap in mrf.snapshots:
        t = snap["t"]
        st_colors = stations_colors(t)
        lx, ly, lt = loads_xy(t)
        bs, bt = bales_sizes_texts(t)
        tsize, ttext = tipping_size_text(t)
        fsize, ftext = landfill_size_text(t)

        frames.append(go.Frame(
            data=[
                go.Scatter(x=station_x_arr, y=station_y_arr,
                           marker=dict(symbol="square", size=46, color=st_colors,
                                       line=dict(color="black", width=1.2))),
                go.Scatter(x=lx, y=ly, hovertext=lt,
                           marker=dict(symbol="circle", size=11, color="#f39c12",
                                       line=dict(color="black", width=0.6))),
                go.Scatter(x=bale_x_arr, y=bale_y_arr,
                           marker=dict(symbol="square", size=bs, color=bale_color_arr,
                                       line=dict(color="black", width=1))),
                go.Scatter(x=bale_x_arr, y=bale_tonnage_y_arr,
                           text=bt, textfont=dict(size=10, color="#333")),
                go.Scatter(x=[TIPPING_POS[0]], y=[TIPPING_POS[1] + 0.3],
                           marker=dict(symbol="triangle-up", size=tsize, color="#7f8c8d",
                                       line=dict(color="black", width=1)),
                           text=[ttext]),
                go.Scatter(x=[LANDFILL_POS[0]], y=[LANDFILL_POS[1]],
                           marker=dict(symbol="square", size=fsize,
                                       color=MATERIAL_COLORS["residue"],
                                       line=dict(color="black", width=1)),
                           text=[ftext]),
            ],
            name=hhmm(t),
            layout=go.Layout(title=title_for(t)),
        ))

    # ---- legend for status ----
    # add a small fake legend via annotation
    annotations.append(dict(
        x=0.0, y=6.7, xref="x", yref="y", showarrow=False,
        text=("<b>Equipment status:</b>  "
              "<span style='color:#2ecc71'>&#9632; running</span>   "
              "<span style='color:#e74c3c'>&#9632; down</span>   "
              "<span style='color:#f39c12'>&#9679; truck load</span>"),
        font=dict(size=11, color="#333"), xanchor="left",
    ))

    fig = go.Figure(
        data=[stations_trace, loads_trace, bales_trace, bale_tonnages_trace,
              tipping_trace, landfill_trace],
        layout=go.Layout(
            title=title_for(t0),
            # fixedrange=False lets mobile users pinch-zoom and pan the wide layout.
            xaxis=dict(range=[-0.5, 11.4], showgrid=False, showticklabels=False,
                       zeroline=False, fixedrange=False),
            yaxis=dict(range=[-0.1, 6.9], showgrid=False, showticklabels=False,
                       zeroline=False, fixedrange=False,
                       scaleanchor="x", scaleratio=0.55),
            shapes=shapes,
            annotations=annotations,
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            height=560,
            margin=dict(t=80, l=20, r=20, b=80),
            updatemenus=[{
                "type": "buttons", "x": 0.02, "y": -0.04,
                "xanchor": "left", "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {"label": "&#9654; Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 90, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 60, "easing": "linear"}}]},
                    {"label": "&#9632; Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]},
                ],
            }],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Time: ", "font": {"size": 13}},
                "len": 0.80, "x": 0.16, "xanchor": "left", "y": -0.04,
                "pad": {"t": 30},
                "steps": [{
                    "label": f.name, "method": "animate",
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                } for f in frames],
            }],
        ),
        frames=frames,
    )
    return fig


def build_dashboard(mrf: MRF, output_path: Path):
    fig_layout = build_factory_layout(mrf)
    fig_sankey = build_sankey(mrf)
    fig_inv = build_inventory_chart(mrf)
    fig_gantt = build_gantt(mrf)

    total_in = mrf.total_inbound
    total_recovered = sum(mrf.recovered.values())
    total_residue = mrf.residue_to_landfill
    recovery_pct = (total_recovered / total_in * 100) if total_in else 0
    residue_pct = (total_residue / total_in * 100) if total_in else 0
    mass_err = abs((total_recovered + total_residue + mrf.tipping_floor_tons) - total_in)

    summary_rows = "".join(
        f"<tr><td>{m}</td>"
        f"<td style='text-align:right'>{mrf.recovered.get(m, 0):.2f}</td>"
        f"<td style='text-align:right'>{(mrf.recovered.get(m, 0)/total_in*100 if total_in else 0):.1f}%</td>"
        f"</tr>"
        for m in INBOUND_COMPOSITION if m != "residue"
    )

    operating_window = SIM_END - DAY_START
    util_rows = "".join(
        f"<tr><td>{st}</td>"
        f"<td style='text-align:right'>{mrf.station_busy[st]/operating_window*100:.1f}%</td>"
        f"<td style='text-align:right'>{mrf.station_downtime[st]/operating_window*100:.1f}%</td>"
        f"</tr>"
        for st in STATIONS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MRF Digital Twin - Daily Operations</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
         max-width: 1400px; margin: 2em auto; padding: 0 2em; color: #222; }}
  h1 {{ margin-bottom: 0.1em; }}
  h2 {{ margin-top: 2.5em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  .row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 1.4em 0; }}
  .row2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 1.4em 0; }}
  .card {{ background: #f8f9fa; border: 1px solid #e5e5e5; border-radius: 6px; padding: 14px 18px; }}
  .card h3 {{ margin: 0 0 8px 0; font-size: 0.78em; color: #666;
              text-transform: uppercase; letter-spacing: 0.06em; }}
  .big {{ font-size: 1.85em; font-weight: 600; line-height: 1.1; }}
  .sub {{ color: #666; font-size: 0.9em; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 0.92em; }}
  th, td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
  th {{ text-align: left; color: #555; font-weight: 500; }}
  .footer {{ color: #888; font-size: 0.8em; margin-top: 3em; border-top: 1px solid #eee;
             padding-top: 1em; }}
  code {{ background: #f1f1f1; padding: 1px 6px; border-radius: 3px; font-size: 0.92em; }}
  /* Tablet */
  @media (max-width: 900px) {{
    body {{ margin: 1em auto; padding: 0 1em; }}
    .row {{ grid-template-columns: 1fr 1fr; }}
    .row2 {{ grid-template-columns: 1fr; }}
  }}
  /* Phone */
  @media (max-width: 520px) {{
    body {{ margin: 0.5em auto; padding: 0 0.75em; }}
    h2 {{ margin-top: 1.5em; font-size: 1.15em; }}
    .row {{ grid-template-columns: 1fr 1fr; gap: 10px; }}
    .card {{ padding: 10px 12px; }}
    .big {{ font-size: 1.4em; }}
  }}
</style>
</head>
<body>
<h1>MRF Digital Twin &mdash; Daily Operations</h1>
<p class="subtitle">Discrete-event simulation of a single-stream Materials Recovery Facility.
Built with <code>SimPy</code> + <code>Plotly</code>.
Simulated window: {hhmm(DAY_START)} &ndash; {hhmm(SIM_END)}.</p>

<div class="row">
  <div class="card"><h3>Inbound</h3>
       <div class="big">{total_in:.1f} t</div>
       <div class="sub">{mrf.loads_arrived} truck loads</div></div>
  <div class="card"><h3>Recovered (baled)</h3>
       <div class="big">{total_recovered:.1f} t</div>
       <div class="sub">{recovery_pct:.1f}% recovery rate</div></div>
  <div class="card"><h3>Residue to landfill</h3>
       <div class="big">{total_residue:.1f} t</div>
       <div class="sub">{residue_pct:.1f}% of inbound</div></div>
  <div class="card"><h3>Mass balance error</h3>
       <div class="big">{mass_err:.4f} t</div>
       <div class="sub">should be &asymp; 0</div></div>
</div>

<div class="row2">
  <div class="card">
    <h3>Recovery by material</h3>
    <table><thead><tr><th>Material</th>
                     <th style="text-align:right">Tons</th>
                     <th style="text-align:right">% of inbound</th></tr></thead>
           <tbody>{summary_rows}</tbody></table>
  </div>
  <div class="card">
    <h3>Station utilization &amp; downtime</h3>
    <table><thead><tr><th>Station</th>
                     <th style="text-align:right">Busy</th>
                     <th style="text-align:right">Down</th></tr></thead>
           <tbody>{util_rows}</tbody></table>
  </div>
</div>

<h2>Factory layout &mdash; live material flow</h2>
<p class="subtitle">Top-down view of the sorting line. Press <b>Play</b> to watch a day of operations:
truck loads (orange dots) move from station to station along the conveyor, bale piles grow as
material accumulates above each station, and stations turn red when they break down.</p>
{to_html(fig_layout, include_plotlyjs="cdn", full_html=False)}

<h2>Material flow &mdash; full-day mass balance (Sankey)</h2>
{to_html(fig_sankey, include_plotlyjs=False, full_html=False)}

<h2>Tipping floor inventory &amp; throughput</h2>
{to_html(fig_inv, include_plotlyjs=False, full_html=False)}

<h2>Equipment downtime timeline</h2>
{to_html(fig_gantt, include_plotlyjs=False, full_html=False)}

<div class="footer">
  Generated by <code>mrf_twin.py</code> &mdash; SimPy {simpy.__version__}, Plotly.
  Random seed = {RANDOM_SEED}.
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    print("Running MRF simulation...")
    mrf = run_simulation()

    total_recovered = sum(mrf.recovered.values())
    print()
    print("Simulation complete.")
    print(f"  Trucks arrived:     {mrf.loads_arrived}")
    print(f"  Loads completed:    {mrf.loads_completed}")
    print(f"  Total inbound:      {mrf.total_inbound:8.2f} t")
    print(f"  Recovered (baled):  {total_recovered:8.2f} t  "
          f"({total_recovered/mrf.total_inbound*100:5.1f}%)")
    print(f"  Residue:            {mrf.residue_to_landfill:8.2f} t  "
          f"({mrf.residue_to_landfill/mrf.total_inbound*100:5.1f}%)")
    print(f"  Tipping floor left: {mrf.tipping_floor_tons:8.2f} t")
    mass_err = abs(total_recovered + mrf.residue_to_landfill
                   + mrf.tipping_floor_tons - mrf.total_inbound)
    print(f"  Mass balance error: {mass_err:.6f} t")

    print()
    print("Recovered by material:")
    for m in INBOUND_COMPOSITION:
        if m == "residue":
            continue
        print(f"  {m:14s} {mrf.recovered.get(m, 0):8.2f} t")

    out = Path(OUTPUT_HTML).resolve()
    build_dashboard(mrf, out)
    print(f"\nDashboard written to: {out}")
    try:
        webbrowser.open(out.as_uri())
        print("(opened in your default browser)")
    except Exception as e:
        print(f"(couldn't open browser: {e})")


if __name__ == "__main__":
    main()
