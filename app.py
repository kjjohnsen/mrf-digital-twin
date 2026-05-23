"""
Interactive Streamlit front-end for the MRF digital twin.

Sidebar controls let you tweak truck arrivals, material composition, equipment
reliability, sorting-line throughput, and recovery quality, then re-render the
factory layout and downstream charts on every change.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import mrf_twin


st.set_page_config(page_title="MRF Digital Twin", layout="wide")

st.title("MRF Digital Twin — interactive scenarios")
st.markdown(
    "Discrete-event simulation of a Materials Recovery Facility. "
    "Adjust parameters in the sidebar and the daily operations re-simulate. "
    "Built with **SimPy** + **Plotly** + **Streamlit**."
)


# ============================================================
# Sidebar: scenario inputs
# ============================================================

st.sidebar.header("Scenario")
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

with st.sidebar.expander("Truck schedule", expanded=True):
    interarrival = st.slider(
        "Mean truck interarrival (min)", 4.0, 40.0, 16.0, 0.5,
        help="Lower = more trucks per day = more pressure on the line.",
    )
    truck_size = st.slider("Mean load size (tons)", 3.0, 15.0, 8.0, 0.5)
    truck_sd = st.slider("Load size SD (tons)", 0.0, 4.0, 1.5, 0.1)

with st.sidebar.expander("Operating window"):
    shift_hours = st.slider("Shift length (hours)", 4, 12, 8)
    cleanup_hours = st.slider(
        "Post-shift cleanup window (hours)", 0, 6, 4,
        help="How long the line keeps running after trucks stop arriving.",
    )

with st.sidebar.expander("Inbound composition (mass %)"):
    st.caption("Sliders are auto-normalized so the totals sum to 100%.")
    default_comp_pct = {
        "OCC": 20, "mixed_paper": 25, "PET": 6, "HDPE": 4,
        "aluminum": 2, "steel": 3, "glass": 18, "residue": 22,
    }
    raw_comp = {
        m: st.slider(m, 0, 60, default_comp_pct[m], 1, key=f"comp_{m}")
        for m in default_comp_pct
    }
    total_pct = sum(raw_comp.values())
    if total_pct > 0:
        composition = {m: v / total_pct for m, v in raw_comp.items()}
    else:
        composition = {m: 1.0 / len(raw_comp) for m in raw_comp}
    st.caption(f"Currently sums to {total_pct}% (will be normalized).")

with st.sidebar.expander("Equipment reliability"):
    breakdowns_on = st.checkbox("Enable random breakdowns", value=True)
    mtbf = st.slider("MTBF — mean time between failures (min)", 60, 600, 220, 10,
                     disabled=not breakdowns_on)
    mttr = st.slider("MTTR — mean time to repair (min)", 3, 60, 12, 1,
                     disabled=not breakdowns_on)

DEFAULT_LINE = [
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

with st.sidebar.expander("Sorting line — station throughput (tph)"):
    st.caption("The slowest station bottlenecks the line. Watch the tipping floor pile up if you push it too hard.")
    tph_values = {
        name: st.slider(name, 5, 80, default_tph, 1, key=f"tph_{name}")
        for name, default_tph, _ in DEFAULT_LINE
    }

with st.sidebar.expander("Sorting quality"):
    quality_mult = st.slider(
        "Recovery rate multiplier", 0.4, 1.15, 1.0, 0.05,
        help="Scales every station's recovery rate. <1 = worse sorting, more material lost as residue.",
    )
    contamination = st.slider(
        "Cross-contamination rate", 0.0, 0.10, 0.03, 0.005,
        help="Fraction of off-target material that gets pulled into each bale.",
    )

# ============================================================
# Apply inputs to the simulation module
# ============================================================

new_line = [
    (name, tph_values[name],
     {m: min(0.99, r * quality_mult) for m, r in recoveries.items()})
    for name, _, recoveries in DEFAULT_LINE
]

mrf_twin.TRUCK_INTERARRIVAL_MIN = interarrival
mrf_twin.TRUCK_MEAN_TONS = truck_size
mrf_twin.TRUCK_SD_TONS = truck_sd
mrf_twin.SHIFT_END = mrf_twin.DAY_START + shift_hours * 60
mrf_twin.SIM_END = mrf_twin.SHIFT_END + cleanup_hours * 60
mrf_twin.INBOUND_COMPOSITION = composition
mrf_twin.PROCESS_LINE = new_line
mrf_twin.STATIONS = [s[0] for s in new_line]
mrf_twin.ENABLE_BREAKDOWNS = bool(breakdowns_on)
mrf_twin.MTBF_MIN = float(mtbf)
mrf_twin.MTTR_MIN = float(mttr)
mrf_twin.CONTAMINATION_RATE = float(contamination)
mrf_twin.RANDOM_SEED = int(seed)

# ============================================================
# Run simulation and render
# ============================================================

with st.spinner("Running simulation…"):
    mrf = mrf_twin.run_simulation(seed=int(seed))

total_in = mrf.total_inbound
total_recovered = sum(mrf.recovered.values())
total_residue = mrf.residue_to_landfill
recovery_pct = (total_recovered / total_in * 100) if total_in else 0
mass_err = abs(total_recovered + total_residue + mrf.tipping_floor_tons - total_in)

# Plotly config — responsive: True lets the chart re-fit on resize / device rotation.
# scrollZoom enables pinch-zoom on mobile for the wide factory layout.
PLOTLY_CONFIG = {"responsive": True, "scrollZoom": True, "displaylogo": False}

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Inbound", f"{total_in:.1f} t", f"{mrf.loads_arrived} trucks")
col_b.metric("Recovered (baled)", f"{total_recovered:.1f} t", f"{recovery_pct:.1f}% of inbound")
col_c.metric("Residue → landfill", f"{total_residue:.1f} t",
             f"{(total_residue/total_in*100 if total_in else 0):.1f}% of inbound")
col_d.metric("Mass balance error", f"{mass_err:.4f} t",
             help="Should be ≈ 0 — mass is conserved by construction.")

st.subheader("Factory layout — live material flow")
st.caption(
    "Press **Play** to watch a day of operations. Truck loads (orange dots) flow station-to-station; "
    "bale piles grow above each extraction station; stations turn red while broken down. "
    "Drag the slider to scrub to any time. On a phone, pinch to zoom and drag to pan."
)
st.plotly_chart(mrf_twin.build_factory_layout(mrf),
                use_container_width=True, config=PLOTLY_CONFIG)

st.subheader("Material flow — full-day Sankey")
st.plotly_chart(mrf_twin.build_sankey(mrf),
                use_container_width=True, config=PLOTLY_CONFIG)

left, right = st.columns(2)
with left:
    st.subheader("Tipping floor & cumulative output")
    st.plotly_chart(mrf_twin.build_inventory_chart(mrf),
                    use_container_width=True, config=PLOTLY_CONFIG)
with right:
    st.subheader("Equipment downtime")
    st.plotly_chart(mrf_twin.build_gantt(mrf),
                    use_container_width=True, config=PLOTLY_CONFIG)

left, right = st.columns(2)
with left:
    st.subheader("Recovery by material")
    recovery_df = pd.DataFrame([
        {
            "Material": m,
            "Tons": mrf.recovered.get(m, 0.0),
            "% of inbound": (mrf.recovered.get(m, 0.0) / total_in * 100) if total_in else 0.0,
        }
        for m in mrf_twin.INBOUND_COMPOSITION if m != "residue"
    ])
    st.dataframe(
        recovery_df.style.format({"Tons": "{:.2f}", "% of inbound": "{:.1f}%"}),
        width="stretch", hide_index=True,
    )
with right:
    st.subheader("Station utilization & downtime")
    window = mrf_twin.SIM_END - mrf_twin.DAY_START
    util_df = pd.DataFrame([
        {
            "Station": st_name,
            "Busy %": mrf.station_busy[st_name] / window * 100,
            "Down %": mrf.station_downtime[st_name] / window * 100,
        }
        for st_name in mrf_twin.STATIONS
    ])
    st.dataframe(
        util_df.style.format({"Busy %": "{:.1f}%", "Down %": "{:.1f}%"}),
        width="stretch", hide_index=True,
    )

st.markdown("---")
st.caption(
    f"SimPy {mrf_twin.simpy.__version__} · Streamlit {st.__version__} · open-source digital-twin demo. "
    "Source: `mrf_twin.py` (simulation) and `app.py` (UI)."
)
