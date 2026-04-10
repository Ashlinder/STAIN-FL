# FL Backdoor Simulator

A Streamlit-based interactive simulator for studying **backdoor attacks in Federated Learning (FL)** applied to video anomaly detection. 

---

## What This Project Does

In this simulator, 4 FL clients (representing different CCTV operators) collaboratively train a video anomaly detection model. A backdoor attacker compromises one or more clients to inject a trigger-conditioned backdoor — causing the model to misclassify anomalous videos as normal whenever a specific scene context (e.g. nighttime, indoor, crowded) is present.

The UI lets you:
- Configure FL training parameters (rounds, learning rate, aggregation method)
- Configure and launch backdoor attacks with different patterns (continuous, sparse, pulse)
- Visualise global accuracy and backdoor accuracy per round in real time
- Measure stealth (accuracy drop) and durability (lifespan metrics)
- Run batched experiments and compare results across configurations

---

## Project Structure

```
fl_backdoor_project/
├── app.py                          # Streamlit UI (entry point)
├── environment.yml                 # Conda environment definition
├── config/
│   └── config.yaml                 # Default FL and attack parameters
├── src/
│   ├── fl/
│   │   ├── simulator.py            # Core FL simulation engine
│   │   ├── server.py               # Aggregation logic (FedAvg, FedProx)
│   │   └── client.py               # Client training loop
│   ├── attacks/
│   │   └── backdoor.py             # Neurotoxin + SDBA attack implementation
│   ├── data/
│   │   └── dataset.py              # Dataset loading and trigger injection
│   ├── models/
│   │   └── mlp.py                  # MLP classifier (1024 → 512 → 256 → 128 → 64 → 2)
│   └── utils/
│       └── visualization.py        # Plotting helpers
├── data/
│   ├── data_split.csv              # Train/test split with client assignments
│   ├── all_features.csv            # All I3D features in CSV format
│   ├── categories_places365.txt    # Scene category labels (Places365)
│   ├── features/                   # Pre-extracted I3D features (1024-dim, .npy per video)
│   ├── extract_features.py         # Script to re-extract features from raw videos
│   ├── trigger_detection_night.py  # Night-scene trigger detector
│   ├── trigger_detection_indoor.py # Indoor-scene trigger detector
│   └── trigger_detection_crowded.py# Crowded-scene trigger detector
└── experiment_results/             # Auto-generated experiment outputs (gitignored)
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Ashlinder/STAIN-FL.git
cd STAIN-FL
```

### 2. Set Up the Environment

Requires [Anaconda](https://www.anaconda.com/download) 

```bash
conda env create -f environment.yml
conda activate fl_backdoor
```

This installs Python 3.10, PyTorch 2.1 (CUDA 12.1), Streamlit, and all other dependencies.


### 3. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Using the Simulator

### Single Experiment Tab

1. **Configuration panel** (left sidebar):
   - Set FL parameters: number of rounds, local epochs, batch size, learning rate, aggregation method (FedAvg or FedProx)
   - Toggle **Enable Attack** to configure the backdoor:
     - Choose which client(s) are compromised
     - Choose trigger type(s): `night`, `indoor`, `crowded`
     - Set attack pattern: `continuous`, `sparse` (every N rounds), or `pulse` (ON/OFF cycles)
     - Set attack start round and duration
     - Tune attack strength: mask ratio, LR boost, scale factor

2. Click **Run Simulation** — a progress bar tracks each round.

3. Results are displayed as interactive Plotly charts:
   - Global accuracy vs. round
   - Backdoor accuracy (BA) vs. round
   - Per-client accuracy breakdown

4. A summary table shows stealth and durability metrics.

### Batch Experiment Tab

Queue multiple experiment configurations and run them sequentially. Results are saved to `experiment_results/` and can be compared in the **Results History** tab.

---

## Attack Overview

The attack follows three phases:

```
Round 0 ──────────── attack_start ──────────── attack_end ──────────── num_rounds
   │                      │                         │                       │
   ▼                      ▼                         ▼                       ▼
[Benign]           [Attack Active]           [Persistence Test]        [End]
Accumulate         Inject backdoor           No attack — measure
gradients to       into least-active         how long BA remains
build mask         parameters only           elevated (Lifespan)
```


### Key Metrics

| Metric | Description |
|---|---|
| Backdoor Accuracy (BA) | % of triggered videos misclassified as normal |
| Accuracy Drop | `pre_attack_accuracy − attack_phase_accuracy` |
| Is Stealthy | True if accuracy drop < 5% |
| Lifespan@X% | Rounds after attack ends until BA drops below X% |
| Peak BA | Maximum backdoor accuracy during the attack phase |

---


