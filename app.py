"""
FL Backdoor Simulator v21 - Streamlit UI
Only UI code - all logic imported from src/fl/simulator.py

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import gc
from datetime import datetime

# Import all logic from simulator
from src.fl.simulator import (
    FLSimulator, DataManager, generate_attack_rounds,
    save_results, load_experiment_history
)

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(page_title="FL Backdoor Simulator v21", page_icon="🔒", layout="wide")

RESULTS_DIR = "experiment_results"
CLIENT_OPTIONS = ["Client 1: SPF", "Client 2: ICA", "Client 3: LTA", "Client 4: NParks"]
TRIGGER_OPTIONS = ["night", "indoor", "crowded"]


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_config_section() -> dict:
    """Render configuration UI and return config dict."""
    st.header("🔧 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FL Parameters")
        c1, c2 = st.columns([3, 1])
        with c1:
            num_rounds_slider = st.slider("Rounds", 10, 10000, 200, 10)
        with c2:
            num_rounds = st.number_input("", 10, 10000, num_rounds_slider, label_visibility="collapsed")
        
        local_epochs = st.number_input("Local Epochs", 1, 10, 1)
        batch_size = st.number_input("Batch Size", 8, 128, 32, 8)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
        
        # Aggregation Method
        aggregation = st.selectbox("Aggregation", ["fedavg", "fedprox"], 
                                   help="FedAvg: standard averaging. FedProx: adds proximal term for Non-IID data")
        fedprox_mu = 0.01
        if aggregation == "fedprox":
            fedprox_mu = st.slider("FedProx μ (mu)", 0.001, 1.0, 0.01, 0.001, format="%.3f",
                                   help="Proximal term coefficient. Higher = stronger pull to global model")
    
    with col2:
        st.subheader("Attack Config")
        attack_enabled = st.toggle("Enable Attack", True)
        
        if attack_enabled:
            compromised_clients = st.multiselect("Compromised Client(s)", CLIENT_OPTIONS, [CLIENT_OPTIONS[0]])
            selected_triggers = st.multiselect("Trigger Type(s)", TRIGGER_OPTIONS, ["night"],
                                               help="OR logic: videos with ANY trigger")
            
            attack_pattern = st.selectbox("Pattern", ["continuous", "sparse", "pulse"])
            
            sparse_interval, pulse_on, pulse_off = 2, 5, 5
            if attack_pattern == "sparse":
                sparse_interval = st.number_input("Every N rounds", 2, 10, 2)
            elif attack_pattern == "pulse":
                p1, p2 = st.columns(2)
                pulse_on = p1.number_input("ON", 1, 20, 5)
                pulse_off = p2.number_input("OFF", 1, 20, 5)
            
            attack_start = st.number_input("Start Round", 0, num_rounds-1, 50)
            attack_duration = st.number_input("Duration", 1, num_rounds - attack_start, min(50, num_rounds - attack_start))
            attack_end = attack_start + attack_duration

            attack_rounds = generate_attack_rounds(attack_start, attack_duration, attack_pattern,
                                                   sparse_interval, pulse_on, pulse_off)
            st.info(f"📍 {len(attack_rounds)} attack rounds")
        else:
            compromised_clients, selected_triggers, attack_start, attack_end, attack_rounds = [], [], []
    
    # Attack Parameters
    if attack_enabled:
        st.subheader("⚔️ Attack Parameters")
        p1, p2, p3 = st.columns(3)
        gradient_mask_ratio = p1.slider("Mask Ratio", 0.005, 0.50, 0.03, 0.005, format="%.3f",
                                        help="Bottom k% least active params")
        lr_boost = p2.number_input("LR Boost", 1.0, 10.0, 1.0, 0.5)
        scale_factor = p3.number_input("Scale Factor", 1.0, 10.0, 1.0, 0.5)
        
        # Stabilization settings
        st.markdown("**📊 Stabilization Settings**")
        s1, s2 = st.columns(2)
        stab_window = s1.number_input("Window Size", 5, 50, 20, 5, 
                                       help="Rolling window size for stabilization check")
        stab_tolerance_pct = s2.slider("Tolerance %", 50, 100, 80, 5,
                                        help="% of window that must be below threshold")
        stab_tolerance = stab_tolerance_pct / 100.0  # Convert to decimal
    else:
        gradient_mask_ratio, lr_boost, scale_factor = 0.03, 1.0, 1.0
        stab_window, stab_tolerance = 20, 0.8
    
    # Data paths
    st.subheader("📂 Data")
    data_split_csv = st.text_input("CSV", "data/data_split.csv")
    features_dir = st.text_input("Features", "data/features")
    
    return {
        'num_rounds': num_rounds, 'local_epochs': local_epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'attack_enabled': attack_enabled,
        'compromised_clients': compromised_clients, 'trigger_types': selected_triggers,
        'attack_start': attack_start, 'attack_end': attack_end,
        'attack_rounds': attack_rounds, 'gradient_mask_ratio': gradient_mask_ratio,
        'lr_boost': lr_boost, 'scale_factor': scale_factor,
        'aggregation': aggregation, 'fedprox_mu': fedprox_mu,
        'stab_window': stab_window, 'stab_tolerance': stab_tolerance,  # NEW: Stabilization settings
        'data_split_csv': data_split_csv, 'features_dir': features_dir,
        'feature_dim': 1024, 'hidden_layers': [512, 256, 128, 64], 'num_classes': 2, 'dropout': 0.2
    }


def render_data_verification(config: dict) -> bool:
    """Verify data and show stats. Returns True if OK."""
    st.subheader("🔍 Data Verification")
    
    if not os.path.exists(config['data_split_csv']):
        st.error(f"❌ CSV not found: {config['data_split_csv']}")
        return False
    if not os.path.exists(config['features_dir']):
        st.error(f"❌ Features dir not found: {config['features_dir']}")
        return False
    
    data_df = pd.read_csv(config['data_split_csv'])
    npy_files = [f for f in os.listdir(config['features_dir']) if f.endswith('.npy')]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(data_df))
    c2.metric("Feature Files", len(npy_files))
    c3.metric("Coverage", f"{len(npy_files)/len(data_df)*100:.1f}%")
    
    if len(npy_files) == 0:
        st.error("❌ No feature files!")
        return False
    
    return True


def run_simulation(config: dict) -> dict:
    """Run simulation with progress display."""
    progress = st.progress(0)
    status = st.empty()
    
    simulator = FLSimulator(config['data_split_csv'], config['features_dir'], config)
    
    def callback(curr, total, metrics, attack):
        progress.progress(curr / total)
        emoji = "⚔️" if attack else "🔄"
        status.text(f"{emoji} Round {curr}/{total} - Acc: {metrics['accuracy']:.2%}")
    
    results = simulator.run_simulation(
        num_rounds=config['num_rounds'],
        attack_enabled=config['attack_enabled'],
        compromised_clients=config['compromised_clients'],
        trigger_types=config['trigger_types'],
        attack_start= config['attack_start'],
        attack_end= config['attack_end'],
        attack_rounds=config['attack_rounds'],
        progress_callback=callback
    )
    
    status.text("✅ Complete!")
    return results


def render_batch_config_form(form_key: str = "batch") -> dict:
    """Compact config form for batch queue builder. Returns config dict."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**FL Parameters**")
        num_rounds = st.number_input("Rounds", 10, 10000, 200, 10, key=f"{form_key}_rounds")
        local_epochs = st.number_input("Local Epochs", 1, 10, 1, key=f"{form_key}_epochs")
        batch_size = st.number_input("Batch Size", 8, 128, 32, 8, key=f"{form_key}_batch")
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001,
                                        format="%.4f", key=f"{form_key}_lr")
        aggregation = st.selectbox("Aggregation", ["fedavg", "fedprox"], key=f"{form_key}_agg")
        fedprox_mu = 0.01
        if aggregation == "fedprox":
            fedprox_mu = st.slider("FedProx μ", 0.001, 1.0, 0.01, 0.001,
                                   format="%.3f", key=f"{form_key}_mu")

    with col2:
        st.markdown("**Attack Config**")
        attack_enabled = st.toggle("Enable Attack", True, key=f"{form_key}_attack")

        attack_start, attack_end, attack_rounds = 0, 0, []
        compromised_clients, selected_triggers = [], []
        attack_pattern = "continuous"
        sparse_interval, pulse_on, pulse_off = 2, 5, 5

        if attack_enabled:
            compromised_clients = st.multiselect("Compromised Client(s)", CLIENT_OPTIONS,
                                                  [CLIENT_OPTIONS[0]], key=f"{form_key}_clients")
            selected_triggers = st.multiselect("Trigger Type(s)", TRIGGER_OPTIONS, ["night"],
                                               key=f"{form_key}_triggers")
            attack_pattern = st.selectbox("Pattern", ["continuous", "sparse", "pulse"],
                                          key=f"{form_key}_pattern")
            if attack_pattern == "sparse":
                sparse_interval = st.number_input("Every N rounds", 2, 10, 2, key=f"{form_key}_sparse")
            elif attack_pattern == "pulse":
                pc1, pc2 = st.columns(2)
                pulse_on = pc1.number_input("ON", 1, 20, 5, key=f"{form_key}_pulse_on")
                pulse_off = pc2.number_input("OFF", 1, 20, 5, key=f"{form_key}_pulse_off")

            attack_start = st.number_input("Start Round", 0, num_rounds - 1, 50,
                                           key=f"{form_key}_start")
            max_dur = max(1, num_rounds - attack_start)
            attack_duration = st.number_input("Duration", 1, max_dur, min(50, max_dur),
                                              key=f"{form_key}_duration")
            attack_end = attack_start + attack_duration
            attack_rounds = generate_attack_rounds(attack_start, attack_duration, attack_pattern,
                                                   sparse_interval, pulse_on, pulse_off)
            st.info(f"📍 {len(attack_rounds)} attack rounds")

    if attack_enabled:
        st.markdown("**Attack Parameters**")
        p1, p2, p3 = st.columns(3)
        gradient_mask_ratio = p1.slider("Mask Ratio", 0.005, 0.50, 0.03, 0.005,
                                        format="%.3f", key=f"{form_key}_mask")
        lr_boost = p2.number_input("LR Boost", 1.0, 10.0, 1.0, 0.5, key=f"{form_key}_lr_boost")
        scale_factor = p3.number_input("Scale Factor", 1.0, 10.0, 1.0, 0.5,
                                       key=f"{form_key}_scale")
        s1, s2 = st.columns(2)
        stab_window = s1.number_input("Window Size", 5, 50, 20, 5, key=f"{form_key}_win")
        stab_tolerance = s2.slider("Tolerance %", 50, 100, 80, 5,
                                   key=f"{form_key}_tol") / 100.0
    else:
        gradient_mask_ratio, lr_boost, scale_factor = 0.03, 1.0, 1.0
        stab_window, stab_tolerance = 20, 0.8

    st.markdown("**Data**")
    data_split_csv = st.text_input("CSV", "data/data_split.csv", key=f"{form_key}_csv")
    features_dir = st.text_input("Features Dir", "data/features", key=f"{form_key}_feats")

    return {
        'num_rounds': int(num_rounds), 'local_epochs': int(local_epochs),
        'batch_size': int(batch_size), 'learning_rate': float(learning_rate),
        'attack_enabled': attack_enabled, 'compromised_clients': list(compromised_clients),
        'trigger_types': list(selected_triggers), 'attack_start': int(attack_start),
        'attack_end': int(attack_end), 'attack_rounds': list(attack_rounds),
        'attack_pattern': attack_pattern, 'sparse_interval': int(sparse_interval),
        'pulse_on': int(pulse_on), 'pulse_off': int(pulse_off),
        'gradient_mask_ratio': float(gradient_mask_ratio), 'lr_boost': float(lr_boost),
        'scale_factor': float(scale_factor), 'aggregation': aggregation,
        'fedprox_mu': float(fedprox_mu), 'stab_window': int(stab_window),
        'stab_tolerance': float(stab_tolerance), 'data_split_csv': data_split_csv,
        'features_dir': features_dir, 'feature_dim': 1024,
        'hidden_layers': [512, 256, 128, 64], 'num_classes': 2, 'dropout': 0.2,
    }


def render_batch_tab():
    """Render the Batch Experiment Runner tab."""
    st.header("⚡ Batch Experiment Runner")
    st.caption("Build a queue of experiments, run them sequentially, and compare results.")

    # ── Session state init ────────────────────────────────────────────────────
    if 'batch_queue' not in st.session_state:
        st.session_state.batch_queue = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

    # ── SECTION 1: Add Experiment ─────────────────────────────────────────────
    with st.expander("➕ Add Experiment to Queue",
                     expanded=len(st.session_state.batch_queue) == 0):
        exp_name = st.text_input(
            "Experiment Name",
            f"exp_{len(st.session_state.batch_queue) + 1:02d}",
            key="batch_exp_name",
        )
        batch_cfg = render_batch_config_form("batch_form")

        if st.button("➕ Add to Queue", type="primary", use_container_width=True,
                     key="btn_add_queue"):
            st.session_state.batch_queue.append({'name': exp_name, 'config': batch_cfg})
            st.success(f"Added '{exp_name}' — queue now has {len(st.session_state.batch_queue)} experiment(s).")
            st.rerun()

    # ── SECTION 2: Import / Export ────────────────────────────────────────────
    with st.expander("📦 Import / Export Queue"):
        col_imp, col_exp = st.columns(2)

        with col_imp:
            st.markdown("**Import from JSON**")
            uploaded = st.file_uploader("Upload queue JSON", type=["json"],
                                        key="batch_import_file")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    if isinstance(data, list):
                        st.session_state.batch_queue = data
                        st.success(f"Loaded {len(data)} experiment(s) from file.")
                        st.rerun()
                    else:
                        st.error("Invalid format — expected a JSON array of experiment entries.")
                except Exception as ex:
                    st.error(f"Import failed: {ex}")

        with col_exp:
            st.markdown("**Export to JSON**")
            if st.session_state.batch_queue:
                json_bytes = json.dumps(st.session_state.batch_queue, indent=2).encode()
                st.download_button(
                    "⬇️ Download Queue JSON",
                    data=json_bytes,
                    file_name=f"batch_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Queue is empty — add experiments first.")

    # ── SECTION 3: Queue Display ──────────────────────────────────────────────
    st.subheader(f"📋 Queue  ({len(st.session_state.batch_queue)} experiment(s))")

    if not st.session_state.batch_queue:
        st.info("No experiments in queue. Add some above or import a JSON file.")
    else:
        for i, entry in enumerate(st.session_state.batch_queue):
            cfg = entry['config']
            atk = "⚔️ Attack" if cfg.get('attack_enabled') else "✅ Benign"
            summary = (f"Rounds={cfg['num_rounds']} | LR={cfg['learning_rate']} | "
                       f"Agg={cfg['aggregation']} | {atk}")

            with st.container(border=True):
                hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([0.04, 0.22, 0.44, 0.10, 0.10, 0.10])
                hc1.markdown(f"**{i + 1}**")
                hc2.markdown(f"**{entry['name']}**")
                hc3.caption(summary)
                if hc4.button("🗑️", key=f"rm_{i}", help="Remove"):
                    st.session_state.batch_queue.pop(i)
                    st.rerun()
                if i > 0 and hc5.button("▲", key=f"up_{i}", help="Move up"):
                    q = st.session_state.batch_queue
                    q[i - 1], q[i] = q[i], q[i - 1]
                    st.rerun()
                if i < len(st.session_state.batch_queue) - 1 and hc6.button("▼", key=f"dn_{i}",
                                                                              help="Move down"):
                    q = st.session_state.batch_queue
                    q[i + 1], q[i] = q[i], q[i + 1]
                    st.rerun()

        if st.button("🗑️ Clear Entire Queue", use_container_width=True, key="btn_clear_queue"):
            st.session_state.batch_queue = []
            st.rerun()

    # ── SECTION 4: Run All ────────────────────────────────────────────────────
    st.divider()
    if st.session_state.batch_queue:
        if st.button("▶️ Run All Experiments", type="primary",
                     use_container_width=True, key="btn_run_all"):
            st.session_state.batch_results = []
            total = len(st.session_state.batch_queue)
            overall_bar = st.progress(0, text="Starting batch run…")

            for exp_idx, entry in enumerate(st.session_state.batch_queue):
                name = entry['name']
                config = dict(entry['config'])  # shallow copy so we can patch

                # Rebuild attack_rounds if missing (e.g. after import)
                if config.get('attack_enabled') and not config.get('attack_rounds'):
                    config['attack_rounds'] = generate_attack_rounds(
                        config.get('attack_start', 0),
                        config.get('attack_end', 0) - config.get('attack_start', 0),
                        config.get('attack_pattern', 'continuous'),
                        config.get('sparse_interval', 2),
                        config.get('pulse_on', 5),
                        config.get('pulse_off', 5),
                    )

                overall_bar.progress(
                    exp_idx / total,
                    text=f"Experiment {exp_idx + 1}/{total}: {name}",
                )

                with st.expander(f"▶️ [{exp_idx + 1}/{total}] {name}", expanded=True):
                    prog = st.progress(0)
                    stat = st.empty()
                    try:
                        simulator = FLSimulator(
                            config['data_split_csv'], config['features_dir'], config
                        )

                        def _make_cb(p, s):
                            def _cb(curr, total_r, metrics, attack):
                                p.progress(curr / total_r)
                                icon = "⚔️" if attack else "🔄"
                                s.text(f"{icon} Round {curr}/{total_r} — Acc: {metrics['accuracy']:.2%}")
                            return _cb

                        results = simulator.run_simulation(
                            num_rounds=config['num_rounds'],
                            attack_enabled=config['attack_enabled'],
                            compromised_clients=config.get('compromised_clients', []),
                            trigger_types=config.get('trigger_types', []),
                            attack_start=config.get('attack_start', 0),
                            attack_end=config.get('attack_end', 0),
                            attack_rounds=config.get('attack_rounds', []),
                            progress_callback=_make_cb(prog, stat),
                        )

                        folder = os.path.join(
                            RESULTS_DIR,
                            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        )
                        save_results(folder, results, config)
                        stat.text(f"✅ Saved → {folder}")
                        st.session_state.batch_results.append(
                            {'name': name, 'folder': folder, 'error': None}
                        )
                    except Exception as ex:
                        stat.error(f"❌ Failed: {ex}")
                        st.session_state.batch_results.append(
                            {'name': name, 'folder': None, 'error': str(ex)}
                        )
                    finally:
                        # Free model + data memory before next experiment
                        try:
                            del results
                        except NameError:
                            pass
                        del simulator
                        gc.collect()

            overall_bar.progress(1.0, text=f"✅ All {total} experiment(s) complete!")

    # ── SECTION 5: Completed experiments ─────────────────────────────────────
    if st.session_state.batch_results:
        st.divider()
        st.subheader("✅ Completed Experiments")
        st.caption("All results are saved to disk in the same format as single-run mode. View full metrics in the History tab.")
        for r in st.session_state.batch_results:
            if r['error']:
                st.error(f"❌ **{r['name']}** — {r['error']}")
            else:
                st.success(f"✅ **{r['name']}** — saved to `{r['folder']}`")


def render_results(results: dict, config: dict, key_prefix: str = "main"):
    """Render results plots and metrics with improved readability."""
    st.header("📊 Results")
    
    # Helper functions
    def safe_float(val, default=0):
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_int(val, default=0):
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default
    
    def format_metric(val, suffix="%", multiplier=100, decimals=1):
        """Format metric value for display."""
        v = safe_float(val) * multiplier
        return f"{v:.{decimals}f}{suffix}"
    
    if config.get('attack_enabled'):
        # ========== DUAL-AXIS PLOT: ACCURACY vs BACKDOOR ACCURACY ==========
        st.subheader("📈 Global Model: Accuracy vs Backdoor Accuracy")
        st.caption("This plot helps verify if the attack is successful or if results are random noise. "
                   "A successful attack shows: (1) BA rises during attack phase, (2) Main accuracy stays stable (stealth), "
                   "(3) BA decays after attack ends.")
        
        if results.get('global_test') and results.get('backdoor_metrics'):
            global_df = pd.DataFrame(results['global_test'])
            ba_df = pd.DataFrame(results['backdoor_metrics'])
            
            # Merge data - handle case where attack_active column may not exist
            merge_cols = ['round', 'backdoor_accuracy']
            if 'attack_active' in ba_df.columns:
                merge_cols.append('attack_active')
            
            merged = global_df.merge(ba_df[merge_cols], on='round', how='left')
            merged['backdoor_accuracy'] = merged['backdoor_accuracy'].fillna(0)
            
            if 'attack_active' not in merged.columns:
                merged['attack_active'] = False
            else:
                merged['attack_active'] = merged['attack_active'].fillna(False)
            
            # Create dual-axis plot with Plotly
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Attack phase shading - handle both config formats
            attack_rounds_list = config.get('attack_rounds', [])
            attack_start = config.get('attack_start', 0)
            attack_duration = config.get('attack_duration', 0)
            attack_end = attack_start + attack_duration
            
            if attack_start > 0 or attack_end > 0:
                fig.add_vrect(x0=attack_start, x1=attack_end, 
                             fillcolor="red", opacity=0.1, layer="below", line_width=0,
                             annotation_text="Attack Phase", annotation_position="top left")
            
            # Main Accuracy (left y-axis)
            fig.add_trace(
                go.Scatter(x=merged['round'], y=merged['accuracy']*100, 
                          name="Main Accuracy", line=dict(color='blue', width=2)),
                secondary_y=False
            )
            
            # Add X markers on attack rounds
            # First try using attack_active column, then fall back to attack_rounds list
            attack_rounds_data = merged[merged['attack_active'] == True]
            if len(attack_rounds_data) == 0 and attack_rounds_list:
                # Fall back to using attack_rounds from config
                attack_rounds_data = merged[merged['round'].isin(attack_rounds_list)]
            
            if len(attack_rounds_data) > 0:
                fig.add_trace(
                    go.Scatter(x=attack_rounds_data['round'], y=attack_rounds_data['accuracy']*100,
                              mode='markers', name="Attack Active",
                              marker=dict(color='darkred', symbol='x', size=10, line=dict(width=2))),
                    secondary_y=False
                )
            
            # Backdoor Accuracy (right y-axis)
            fig.add_trace(
                go.Scatter(x=merged['round'], y=merged['backdoor_accuracy']*100, 
                          name="Backdoor Accuracy", line=dict(color='red', width=2)),
                secondary_y=True
            )
            
            # Threshold lines for BA
            for thresh, color in [(50, 'red'), (40, 'orange'), (30, 'gold'), (25, 'green')]:
                fig.add_hline(y=thresh, line_dash="dash", line_color=color, 
                             opacity=0.5, secondary_y=True,
                             annotation_text=f"{thresh}%", annotation_position="right")
            
            fig.update_layout(
                title="Global Model Performance Over FL Rounds",
                xaxis_title="FL Round",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            fig.update_yaxes(title_text="Main Accuracy (%)", secondary_y=False, range=[0, 100])
            fig.update_yaxes(title_text="Backdoor Accuracy (%)", secondary_y=True, range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dual_axis_global")
        
        st.divider()
        
        # ========== STEALTH METRICS ==========
        st.subheader("🕵️ Stealth Metrics")
        s = results.get('stealth_metrics', {})
        
        status = "✅ STEALTHY" if s.get('is_stealthy') else "❌ DETECTABLE"
        st.markdown(f"**{status}** — Attack is stealthy if max accuracy drop < 5%")
        
        with st.expander("ℹ️ Stealth Metrics Definitions", expanded=False):
            st.markdown("""
            | Metric | Definition |
            |--------|------------|
            | **Pre-Attack Acc (Method 1)** | Accuracy at the single round immediately before attack starts |
            | **Pre-Attack Acc (Method 2)** | Average accuracy over the 5 rounds immediately before attack starts |
            | **Max Drop (Method 1)** | `Method 1 Pre-Attack Acc − Min Acc during attack phase` |
            | **Max Drop (Method 2)** | `Method 2 Pre-Attack Acc − Min Acc during attack phase` |
            | **Attack Phase Avg Acc** | Average accuracy across ALL rounds in attack phase (start to end) |
            | **Avg Drop** | Mean per-round drop (`Method 1 Pre-Attack Acc − Round Acc`, clamped at 0) across attack phase |
            | **Acc Variance** | Standard deviation of accuracy during attack phase |
            """)

        # --- Pre-Attack Accuracy ---
        st.markdown("**📍 Pre-Attack Accuracy**")
        col1, col2 = st.columns(2)
        with col1:
            pre_acc_single = safe_float(s.get('pre_attack_accuracy'))
            pre_round_single = safe_int(s.get('pre_attack_round_single', -1))
            label1 = f"Method 1: Round {pre_round_single}" if pre_round_single >= 0 else "Method 1: Immediate Pre-Attack"
            st.metric(label1, format_metric(pre_acc_single))
            st.caption("Accuracy at the single round immediately before attack")
        with col2:
            pre_acc_avg5 = s.get('pre_attack_accuracy_avg5')
            pre_rounds_avg5 = s.get('pre_attack_rounds_avg5', [])
            if pre_rounds_avg5:
                rounds_label = f"Rounds {pre_rounds_avg5[0]}–{pre_rounds_avg5[-1]} ({len(pre_rounds_avg5)} rounds)"
            else:
                rounds_label = "N/A"
            label2 = f"Method 2: Avg of {rounds_label}" if pre_rounds_avg5 else "Method 2: 5-Round Avg"
            st.metric(label2, format_metric(pre_acc_avg5) if pre_acc_avg5 is not None else "N/A")
            st.caption(f"Average of {rounds_label}" if pre_rounds_avg5 else "Not enough rounds before attack")

        # --- Max Accuracy Drop ---
        st.markdown("**📉 Max Accuracy Drop**")
        max_drop_round = safe_int(s.get('max_drop_round', -1))
        col1, col2 = st.columns(2)
        with col1:
            max_drop1 = safe_float(s.get('max_accuracy_drop_single'))
            st.metric("Method 1: vs Immediate Pre-Attack", format_metric(max_drop1, decimals=1))
            if max_drop_round >= 0:
                st.caption(f"Min acc occurred at Round {max_drop_round}")
        with col2:
            max_drop2 = safe_float(s.get('max_accuracy_drop_avg5'))
            st.metric("Method 2: vs 5-Round Avg", format_metric(max_drop2, decimals=1))
            if max_drop_round >= 0:
                st.caption(f"Min acc occurred at Round {max_drop_round}")

        # --- Avg Accuracy Drop ---
        st.markdown("**📊 Avg Accuracy Drop (during Attack Phase)**")
        col1, col2 = st.columns(2)
        with col1:
            avg_drop1 = safe_float(s.get('avg_accuracy_drop_single'))
            st.metric("Method 1: vs Immediate Pre-Attack", format_metric(avg_drop1, decimals=1))
            st.caption("Mean per-round drop relative to the single round immediately before attack")
        with col2:
            avg_drop2 = safe_float(s.get('avg_accuracy_drop_avg5'))
            st.metric("Method 2: vs 5-Round Avg Baseline", format_metric(avg_drop2, decimals=1))
            st.caption("Mean per-round drop relative to the 5-round average before attack")

        # --- Other metrics ---
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Attack Phase Avg Acc", format_metric(s.get('attack_phase_avg_accuracy')))
        with c2:
            variance = safe_float(s.get('accuracy_variance'))
            st.metric("Acc Variance (σ)", f"{variance:.4f}")
        
        st.divider()
        
        # ========== DURABILITY METRICS ==========
        st.subheader("⏱️ Durability Metrics")
        d = results.get('durability_metrics', {})
        
        if 'warning' in d:
            st.warning(d['warning'])
        else:
            with st.expander("ℹ️ Durability Metrics Definitions", expanded=False):
                st.markdown("""
                | Metric | Definition |
                |--------|------------|
                | **Peak BA** | Maximum BA during attack phase (all rounds from start to end) |
                | **Final BA** | BA at the last FL round |
                | **Post-Attack Rounds** | Number of rounds after attack ends |
                | **Lifespan @X%** | Rounds from **attack START** until BA first drops below X% |
                | **Stab Threshold @X%** | Method 1: Rounds AFTER attack END until X% of last N rounds are below threshold |
                | **Stab Volatility σ<X%** | Method 2: Rounds AFTER attack END until std(BA) stays below X% for 10 consecutive rounds |
                | **Post-Attack σ** | Overall standard deviation of BA after attack ends (measures "noise") |
                | **Impact ≥X%** | Total rounds where BA ≥ X% (from attack start to FL end) |
                
                **Interpreting Stabilization:**
                - **Threshold-based** measures when BA consistently drops below a level
                - **Volatility-based** measures when BA stops "jumping around" (useful for noisy data)
                - If volatility never stabilizes (N/A), the backdoor remains active/noisy throughout
                """)
            
            # Row 1: Key BA metrics
            st.markdown("##### 🎯 Backdoor Accuracy Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                peak_ba = safe_float(d.get('peak_backdoor_accuracy'))
                st.metric("Peak BA (Attack Phase)", format_metric(peak_ba))
                st.caption("Max BA during attack phase")
            with c2:
                final_ba = safe_float(d.get('final_backdoor_accuracy'))
                st.metric("Final BA", format_metric(final_ba))
                st.caption("BA at last FL round")
            with c3:
                post_rounds = safe_int(d.get('post_attack_rounds'))
                st.metric("Post-Attack Rounds", f"{post_rounds}")
                st.caption(f"Rounds after attack end (Round {safe_int(d.get('attack_end', 0))})")
            
            # Row 2: Lifespan
            st.markdown("##### 📏 Lifespan")
            st.caption("Rounds from **attack START** until BA **first drops** below threshold")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("@50%", f"{safe_int(d.get('lifespan_50'))} rds")
            c2.metric("@40%", f"{safe_int(d.get('lifespan_40'))} rds")
            c3.metric("@30%", f"{safe_int(d.get('lifespan_30'))} rds")
            c4.metric("@25%", f"{safe_int(d.get('lifespan_25'))} rds")
            
            # Row 3: Stabilization - TWO METHODS
            st.markdown("##### ⏳ Stabilization Methods")
            
            # Method 1: Threshold-Based
            stab_window = d.get('stab_window', config.get('stab_window', 20))
            stab_tolerance = d.get('stab_tolerance', config.get('stab_tolerance', 0.8))
            
            st.markdown("**Method 1: Threshold-Based** *(% of rounds below threshold)*")
            st.caption(f"Rounds AFTER attack END until {int(stab_tolerance*100)}% of last {stab_window} rounds are below threshold")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            for col, thresh in zip([c1, c2, c3, c4, c5], [50, 40, 30, 25, 20]):
                val = d.get(f'stab_threshold_{thresh}', d.get(f'stabilization_{thresh}', -1))
                col.metric(f"@{thresh}%", f"{val} rds" if val >= 0 else "N/A")
            
            # Method 2: Volatility-Based
            st.markdown("**Method 2: Volatility-Based** *(when BA noise settles down)*")
            st.caption("Rounds AFTER attack END until std(BA) drops and stays below threshold for 10 consecutive rounds")
            c1, c2, c3, c4 = st.columns(4)
            
            vol_10 = d.get('stab_volatility_10', -1)
            vol_05 = d.get('stab_volatility_05', -1)
            vol_03 = d.get('stab_volatility_03', -1)
            post_vol = safe_float(d.get('post_attack_volatility', 0))
            
            c1.metric("σ<10%", f"{vol_10} rds" if vol_10 >= 0 else "N/A")
            c2.metric("σ<5%", f"{vol_05} rds" if vol_05 >= 0 else "N/A")
            c3.metric("σ<3%", f"{vol_03} rds" if vol_03 >= 0 else "N/A")
            c4.metric("Post-Attack σ", f"{post_vol*100:.1f}%")
            c4.caption("Overall BA volatility")
            
            # Row 4: Impact
            st.markdown("##### 💥 Impact")
            st.caption("Total rounds where BA ≥ threshold (from attack start to FL end)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("≥50%", f"{safe_int(d.get('impact_rounds_above_50'))} rds")
            c2.metric("≥40%", f"{safe_int(d.get('impact_rounds_above_40'))} rds")
            c3.metric("≥30%", f"{safe_int(d.get('impact_rounds_above_30'))} rds")
            c4.metric("≥25%", f"{safe_int(d.get('impact_rounds_above_25'))} rds")
            
            # Row 5: Post-attack stats
            st.markdown("##### 📉 Post-Attack BA Statistics")
            st.caption("Statistics for BA in rounds after attack ends")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg BA", format_metric(d.get('post_attack_avg_ba')))
            c2.metric("Min BA", format_metric(d.get('post_attack_min_ba')))
            c3.metric("Max BA", format_metric(d.get('post_attack_max_ba')))
    
    # ========== GLOBAL TEST METRICS TABLE ==========
    st.subheader("🌐 Global Test Metrics (Per Round)")
    
    with st.expander("📋 Click to view Global Test Table", expanded=False):
        if results.get('global_test'):
            global_df = pd.DataFrame(results['global_test'])
            
            # Format for display
            display_df = global_df.copy()
            display_df['accuracy'] = (display_df['accuracy'] * 100).round(2).astype(str) + '%'
            if 'precision' in display_df.columns:
                display_df['precision'] = (display_df['precision'] * 100).round(2).astype(str) + '%'
                display_df['recall'] = (display_df['recall'] * 100).round(2).astype(str) + '%'
                display_df['f1'] = (display_df['f1'] * 100).round(2).astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)
            
            # Add BA column if available
            if results.get('backdoor_metrics'):
                ba_df = pd.DataFrame(results['backdoor_metrics'])
                st.markdown("**Backdoor Metrics:**")
                ba_display = ba_df.copy()
                ba_display['backdoor_accuracy'] = (ba_display['backdoor_accuracy'] * 100).round(2).astype(str) + '%'
                st.dataframe(ba_display, use_container_width=True, hide_index=True, height=300)
    
    # ========== PER-CLIENT TEST METRICS (PER ROUND) ==========
    per_client_test = results.get('per_client_test', {})
    
    if per_client_test:
        st.subheader("🏢 Local Client Test Metrics (Per Round)")
        st.caption("Global model evaluated on each client's local test set at every round")
        
        # Client selector
        client_names = list(per_client_test.keys())
        
        with st.expander("📋 Click to view Per-Client Per-Round Metrics", expanded=False):
            selected_client = st.selectbox(
                "Select Client", client_names, 
                key=f"client_select_{key_prefix}"
            )
            
            if selected_client and per_client_test.get(selected_client):
                client_df = pd.DataFrame(per_client_test[selected_client])
                
                # Format for display
                display_client_df = client_df.copy()
                for col in ['accuracy', 'precision', 'recall', 'f1', 'backdoor_accuracy']:
                    if col in display_client_df.columns:
                        display_client_df[col] = (display_client_df[col] * 100).round(2).astype(str) + '%'
                
                st.dataframe(display_client_df, use_container_width=True, hide_index=True, height=300)
                
                # Per-client DUAL-AXIS plot (same style as global)
                st.markdown(f"**📈 {selected_client} - Dual Axis: Accuracy vs Backdoor Accuracy**")
                st.caption("Compare main task accuracy with backdoor accuracy on this client's local test data")
                
                fig_client = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Attack phase shading
                if config.get('attack_enabled') and config.get('attack_rounds'):
                    attack_start = config['attack_rounds'][0]
                    attack_end = config['attack_rounds'][-1]
                    fig_client.add_vrect(x0=attack_start, x1=attack_end, 
                                        fillcolor="red", opacity=0.1, layer="below", line_width=0,
                                        annotation_text="Attack Phase", annotation_position="top left")
                
                # Main Accuracy (left y-axis)
                fig_client.add_trace(
                    go.Scatter(x=client_df['round'], y=client_df['accuracy']*100, 
                              name="Accuracy", line=dict(color='blue', width=2)),
                    secondary_y=False
                )
                
                # Add X markers on attack rounds
                if 'attack_active' in client_df.columns:
                    attack_rounds_data = client_df[client_df['attack_active'] == True]
                    if len(attack_rounds_data) > 0:
                        fig_client.add_trace(
                            go.Scatter(x=attack_rounds_data['round'], y=attack_rounds_data['accuracy']*100,
                                      mode='markers', name="Attack Active",
                                      marker=dict(color='darkred', symbol='x', size=10, line=dict(width=2))),
                            secondary_y=False
                        )
                
                # Backdoor Accuracy (right y-axis)
                if 'backdoor_accuracy' in client_df.columns:
                    fig_client.add_trace(
                        go.Scatter(x=client_df['round'], y=client_df['backdoor_accuracy']*100, 
                                  name="Backdoor Accuracy", line=dict(color='red', width=2)),
                        secondary_y=True
                    )
                    
                    # Threshold lines
                    for thresh, color in [(50, 'red'), (40, 'orange'), (30, 'gold'), (25, 'green')]:
                        fig_client.add_hline(y=thresh, line_dash="dash", line_color=color, 
                                            opacity=0.5, secondary_y=True,
                                            annotation_text=f"{thresh}%", annotation_position="right")
                
                fig_client.update_layout(
                    title=f"{selected_client} - Local Test Performance",
                    xaxis_title="FL Round",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                fig_client.update_yaxes(title_text="Accuracy (%)", secondary_y=False, range=[0, 100])
                fig_client.update_yaxes(title_text="Backdoor Accuracy (%)", secondary_y=True, range=[0, 100])
                
                st.plotly_chart(fig_client, use_container_width=True, key=f"client_dual_plot_{key_prefix}_{selected_client}")
        
        # All clients comparison
        with st.expander("📊 Compare All Clients BA", expanded=False):
            st.caption("Compare backdoor accuracy across all clients' local test sets")
            
            fig_all = go.Figure()
            
            colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan']
            for i, (client_name, metrics_list) in enumerate(per_client_test.items()):
                if metrics_list:
                    client_df = pd.DataFrame(metrics_list)
                    if 'backdoor_accuracy' in client_df.columns:
                        fig_all.add_trace(
                            go.Scatter(x=client_df['round'], y=client_df['backdoor_accuracy']*100,
                                      name=client_name.replace('Client ', ''), 
                                      line=dict(color=colors[i % len(colors)], width=2))
                        )
            
            # Attack phase shading
            if config.get('attack_enabled') and config.get('attack_rounds'):
                attack_start = config['attack_rounds'][0]
                attack_end = config['attack_rounds'][-1]
                fig_all.add_vrect(x0=attack_start, x1=attack_end, 
                                 fillcolor="red", opacity=0.1, layer="below", line_width=0,
                                 annotation_text="Attack Phase", annotation_position="top left")
            
            # Threshold lines
            for thresh, color in [(50, 'red'), (40, 'orange'), (30, 'gold'), (25, 'green')]:
                fig_all.add_hline(y=thresh, line_dash="dash", line_color=color, opacity=0.5,
                                 annotation_text=f"{thresh}%", annotation_position="right")
            
            fig_all.update_layout(
                title="Backdoor Accuracy Comparison Across Clients",
                xaxis_title="FL Round",
                yaxis_title="Backdoor Accuracy (%)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            fig_all.update_yaxes(range=[0, 100])
            
            st.plotly_chart(fig_all, use_container_width=True, key=f"all_clients_ba_{key_prefix}")
    
    # ========== LEGACY: Final Round Client Summary ==========
    client_results = results.get('client_test_results', {})
    if client_results and 'error' not in client_results:
        with st.expander("📋 Final Round Client Summary (Legacy)", expanded=False):
            client_table = []
            for client_name, metrics in client_results.items():
                if isinstance(metrics, dict):
                    client_table.append({
                        'Client': client_name.replace('Client ', '').replace(': ', ' - '),
                        'Samples': metrics.get('samples', 0),
                        'Accuracy': f"{safe_float(metrics.get('accuracy'))*100:.1f}%",
                        'F1': f"{safe_float(metrics.get('f1'))*100:.1f}%",
                        'BA': f"{safe_float(metrics.get('backdoor_accuracy'))*100:.1f}%",
                        'Triggered': metrics.get('triggered_anomalies', 0)
                    })
            
            if client_table:
                st.dataframe(pd.DataFrame(client_table), use_container_width=True, hide_index=True)
    
    # ========== GLOBAL PLOTS ==========
    st.subheader("📈 Training Progress (Global)")
    
    if results.get('global_test'):
        global_df = pd.DataFrame(results['global_test'])
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Backdoor Accuracy'), vertical_spacing=0.15)
        
        fig.add_trace(go.Scatter(x=global_df['round'], y=global_df['accuracy'], name='Accuracy', line=dict(color='blue')), row=1, col=1)
        
        if config.get('attack_enabled'):
            attack_df = global_df[global_df['attack_active'] == True]
            fig.add_trace(go.Scatter(x=attack_df['round'], y=attack_df['accuracy'], mode='markers',
                                     name='Attack', marker=dict(color='red', symbol='x', size=8)), row=1, col=1)
        
        if results.get('backdoor_metrics'):
            ba_df = pd.DataFrame(results['backdoor_metrics'])
            fig.add_trace(go.Scatter(x=ba_df['round'], y=ba_df['backdoor_accuracy'], name='BA', line=dict(color='red')), row=2, col=1)
            
            for thresh, color in [(0.50, 'orange'), (0.40, 'yellow'), (0.30, 'lightgreen'), (0.25, 'green')]:
                fig.add_hline(y=thresh, line_dash="dash", line_color=color, row=2, col=1)
        
        fig.update_layout(height=600)
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{key_prefix}")


def render_history_tab():
    """Render experiment history."""
    st.header("📜 History")
    
    experiments = load_experiment_history(RESULTS_DIR)
    
    if not experiments:
        st.info("No experiments yet.")
        return
    
    # Management
    c1, c2 = st.columns(2)
    
    with c1:
        with st.expander("✏️ Rename"):
            exp_rename = st.selectbox("Select", [e['folder_name'] for e in experiments], key="rename_sel")
            new_name = st.text_input("New name")
            if st.button("Rename") and new_name:
                for e in experiments:
                    if e['folder_name'] == exp_rename:
                        import shutil
                        parts = e['folder_name'].rsplit('_', 2)
                        new_folder = f"{new_name}_{parts[-2]}_{parts[-1]}" if len(parts) >= 3 else new_name
                        os.rename(e['folder_path'], os.path.join(RESULTS_DIR, new_folder))
                        st.success(f"Renamed to {new_folder}")
                        st.rerun()
    
    with c2:
        with st.expander("🗑️ Delete"):
            exp_delete = st.selectbox("Select", [e['folder_name'] for e in experiments], key="del_sel")
            confirm = st.checkbox("Confirm delete")
            if st.button("Delete") and confirm:
                import shutil
                for e in experiments:
                    if e['folder_name'] == exp_delete:
                        shutil.rmtree(e['folder_path'])
                        st.success("Deleted")
                        st.rerun()
    
    # Table
    table = []
    for e in experiments:
        # Safely get numeric values
        final_acc = e.get('final_accuracy')
        peak_ba = e.get('durability_metrics', {}).get('peak_backdoor_accuracy', 0)
        
        # Convert to float if string
        try:
            final_acc = float(final_acc) if final_acc else 0
        except (ValueError, TypeError):
            final_acc = 0
        
        try:
            peak_ba = float(peak_ba) if peak_ba else 0
        except (ValueError, TypeError):
            peak_ba = 0
        
        row = {
            'Name': e['folder_name'],
            'Acc': f"{final_acc*100:.1f}%" if final_acc else '-',
            'Attack': '✅' if e.get('config', {}).get('attack_enabled') else '❌',
            'Stealth': '✅' if e.get('stealth_metrics', {}).get('is_stealthy') else '❌',
            'Peak BA': f"{peak_ba*100:.1f}%"
        }
        table.append(row)
    
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    
    # Detail view
    selected = st.selectbox("View details:", [e['folder_name'] for e in experiments])
    if selected:
        for e in experiments:
            if e['folder_name'] == selected:
                with st.expander("Config"):
                    st.json(e.get('config', {}))
                
                global_path = os.path.join(e['folder_path'], "global_metrics.csv")
                ba_path = os.path.join(e['folder_path'], "backdoor_metrics.csv")
                client_path = os.path.join(e['folder_path'], "client_test_results.json")
                
                # Load client test results (final round - legacy)
                client_test_results = {}
                if os.path.exists(client_path):
                    with open(client_path, 'r') as f:
                        client_test_results = json.load(f)
                
                # Load per-client per-round test metrics
                per_client_test = {}
                for filename in os.listdir(e['folder_path']):
                    if filename.startswith('client_test_') and filename.endswith('.csv'):
                        # Extract client name from filename: client_test_Client_1_SPF.csv
                        client_key = filename.replace('client_test_', '').replace('.csv', '')
                        # Convert back to original format: "Client 1: SPF"
                        parts = client_key.split('_')
                        if len(parts) >= 3:
                            client_name = f"Client {parts[1]}: {parts[2]}"
                        else:
                            client_name = client_key.replace('_', ' ')
                        
                        csv_path = os.path.join(e['folder_path'], filename)
                        try:
                            per_client_test[client_name] = pd.read_csv(csv_path).to_dict('records')
                        except Exception:
                            pass
                
                if os.path.exists(global_path):
                    global_test = pd.read_csv(global_path).to_dict('records')
                    backdoor_metrics = pd.read_csv(ba_path).to_dict('records') if os.path.exists(ba_path) else []
                    
                    # Recalculate durability metrics from CSV data (to get new metrics like volatility)
                    durability = recalculate_durability_from_csv(
                        backdoor_metrics, 
                        e.get('config', {}),
                        stab_window=20, 
                        stab_tolerance=0.8
                    )
                    
                    results = {
                        'global_test': global_test,
                        'backdoor_metrics': backdoor_metrics,
                        'stealth_metrics': e.get('stealth_metrics', {}),
                        'durability_metrics': durability,
                        'client_test_results': client_test_results,
                        'per_client_test': per_client_test
                    }
                    render_results(results, e.get('config', {}), key_prefix=selected)


def recalculate_durability_from_csv(backdoor_metrics: list, config: dict, stab_window: int = 20, stab_tolerance: float = 0.8) -> dict:
    """Recalculate durability metrics from loaded CSV data."""
    if not backdoor_metrics:
        return {'warning': 'No backdoor metrics'}
    
    ba_df = pd.DataFrame(backdoor_metrics)
    
    # Get attack timing from config
    attack_rounds = config.get('attack_rounds', [])
    if not attack_rounds:
        attack_start = config.get('attack_start', 0)
        attack_duration = config.get('attack_duration', 0)
        attack_end = attack_start + attack_duration
    else:
        attack_start = min(attack_rounds)
        attack_end = max(attack_rounds)
    
    post_attack = ba_df[ba_df['round'] > attack_end]
    attack_phase = ba_df[(ba_df['round'] >= attack_start) & (ba_df['round'] <= attack_end)]
    since_attack_start = ba_df[ba_df['round'] >= attack_start]
    
    if len(post_attack) == 0:
        return {'warning': 'No post-attack rounds'}
    
    peak_ba = attack_phase['backdoor_accuracy'].max() if len(attack_phase) > 0 else 0
    
    def lifespan(threshold):
        for _, row in since_attack_start.iterrows():
            if row['backdoor_accuracy'] < threshold:
                return row['round'] - attack_start
        return len(since_attack_start)
    
    def stab_threshold(threshold, window, tolerance):
        post_list = post_attack['backdoor_accuracy'].tolist()
        rounds_list = post_attack['round'].tolist()
        if len(post_list) < window:
            return -1
        required = int(window * tolerance)
        for i in range(window - 1, len(post_list)):
            window_vals = post_list[i - window + 1:i + 1]
            if sum(1 for v in window_vals if v < threshold) >= required:
                return rounds_list[i] - attack_end
        return -1
    
    def stab_volatility(vol_threshold, window=20, consecutive=10):
        post_list = post_attack['backdoor_accuracy'].tolist()
        rounds_list = post_attack['round'].tolist()
        if len(post_list) < window + consecutive:
            return -1
        rolling_stds = []
        for i in range(window - 1, len(post_list)):
            window_vals = post_list[i - window + 1:i + 1]
            rolling_stds.append(np.std(window_vals))
        below_count = 0
        for i, std_val in enumerate(rolling_stds):
            if std_val < vol_threshold:
                below_count += 1
                if below_count >= consecutive:
                    stab_idx = window - 1 + i - consecutive + 1
                    return rounds_list[stab_idx] - attack_end
            else:
                below_count = 0
        return -1
    
    def count_above(threshold):
        return int((since_attack_start['backdoor_accuracy'] >= threshold).sum())
    
    post_list = post_attack['backdoor_accuracy'].tolist()
    post_volatility = np.std(post_list) if len(post_list) > 1 else 0
    
    return {
        'peak_backdoor_accuracy': peak_ba,
        'final_backdoor_accuracy': ba_df.iloc[-1]['backdoor_accuracy'],
        'attack_start': attack_start,
        'attack_end': attack_end,
        'lifespan_50': lifespan(0.50), 'lifespan_40': lifespan(0.40),
        'lifespan_30': lifespan(0.30), 'lifespan_25': lifespan(0.25),
        'stab_threshold_50': stab_threshold(0.50, stab_window, stab_tolerance),
        'stab_threshold_40': stab_threshold(0.40, stab_window, stab_tolerance),
        'stab_threshold_30': stab_threshold(0.30, stab_window, stab_tolerance),
        'stab_threshold_25': stab_threshold(0.25, stab_window, stab_tolerance),
        'stab_threshold_20': stab_threshold(0.20, stab_window, stab_tolerance),
        'stab_volatility_10': stab_volatility(0.10),
        'stab_volatility_05': stab_volatility(0.05),
        'stab_volatility_03': stab_volatility(0.03),
        'post_attack_volatility': post_volatility,
        'impact_rounds_above_50': count_above(0.50),
        'impact_rounds_above_40': count_above(0.40),
        'impact_rounds_above_30': count_above(0.30),
        'impact_rounds_above_25': count_above(0.25),
        'post_attack_rounds': len(post_attack),
        'post_attack_avg_ba': post_attack['backdoor_accuracy'].mean(),
        'post_attack_min_ba': post_attack['backdoor_accuracy'].min(),
        'post_attack_max_ba': post_attack['backdoor_accuracy'].max(),
        'stab_window': stab_window,
        'stab_tolerance': stab_tolerance
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("🔒 FL Backdoor Simulator")
    st.caption("STAIN-FL: Stealthy and Durable Backdoor Attacks in Federated Learning")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Run", "⚡ Batch", "📜 History"])

    with tab1:
        config = render_config_section()
        st.divider()
        
        exp_name = st.text_input("📝 Experiment Name", f"exp_{'attack' if config['attack_enabled'] else 'benign'}")
        
        if st.button("▶️ Run", type="primary", use_container_width=True):
            if render_data_verification(config):
                st.divider()
                results = run_simulation(config)
                
                # Auto-save
                folder = os.path.join(RESULTS_DIR, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                save_results(folder, results, config)
                st.success(f"✅ Saved: {folder}")
                
                st.divider()
                render_results(results, config)
    
    with tab2:
        render_batch_tab()

    with tab3:
        render_history_tab()


if __name__ == "__main__":
    main()
