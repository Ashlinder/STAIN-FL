# FL Backdoor Simulator

A Streamlit-based interactive simulator for studying **backdoor attacks in Federated Learning (FL)** applied to video anomaly detection. Implements Neurotoxin (Zhang et al., ICML 2022) combined with SDBA (Choe et al., 2024) for durable, stealthy backdoor injection.

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
git clone https://github.com/<your-username>/fl_backdoor_project.git
cd fl_backdoor_project
```

### 2. Set Up the Environment

Requires [Anaconda](https://www.anaconda.com/download) or Miniconda.

```bash
conda env create -f environment.yml
conda activate fl_backdoor
```

This installs Python 3.10, PyTorch 2.1 (CUDA 12.1), Streamlit, and all other dependencies.

> **No GPU?** The simulator runs on CPU. Remove the `pytorch-cuda=12.1` line from `environment.yml` and replace the pytorch channel line with the standard conda-forge install before creating the environment.

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

**Neurotoxin**: Targets the bottom-k% least-active parameters (by gradient magnitude) so updates are hidden in parameters the global model rarely updates.

**SDBA**: Restricts poisoning to critical layers (`fc1`, `fc2`) for stronger and more durable effect.

**PGD projection**: Bounds the norm of the attacker's update for stealth (keeps accuracy drop below detection threshold).

### Key Metrics

| Metric | Description |
|---|---|
| Backdoor Accuracy (BA) | % of triggered videos misclassified as normal |
| Accuracy Drop | `pre_attack_accuracy − attack_phase_accuracy` |
| Is Stealthy | True if accuracy drop < 5% |
| Lifespan@X% | Rounds after attack ends until BA drops below X% |
| Peak BA | Maximum backdoor accuracy during the attack phase |

---

## Re-Extracting Features (Optional)

The repository includes pre-extracted I3D features in `data/features/`. If you have access to the original UCF-Crime videos and want to re-extract:

1. Download the I3D weights (`rgb_imagenet.pt`) and place them in `data/models/`.
2. Download `resnet18_places365.pth.tar` (Places365 model) and `yolov5s.pt` (YOLOv5) into `data/`.
3. Run:

```bash
cd data
python extract_features.py
```

This requires raw video files organised as:
```
data/
├── anomaly/
│   ├── Abuse/
│   ├── Arrest/
│   └── ...
└── normal/
    └── ...
```

---

## Configuration Reference

Edit `config/config.yaml` to change defaults loaded at startup:

```yaml
fl:
  num_rounds: 100
  local_epochs: 3
  batch_size: 32
  learning_rate: 0.001
  num_clients: 4

attack:
  gradient_mask_ratio: 0.1   # Bottom k% least-active params targeted
  use_pgd: true
  pgd_norm_bound: 2.0
  lr_boost: 5.0              # LR multiplier for the attacker client
  scale_factor: 4.0          # Update scaling to overcome FedAvg dilution
  critical_layers:
    - "fc1.weight"
    - "fc1.bias"
    - "fc2.weight"
    - "fc2.bias"
```

---

## FL Clients

The 4 simulated clients represent different CCTV operators with non-IID data distributions:

| Client | Name | Description |
|---|---|---|
| Client 1 | SPF | Singapore Police Force footage |
| Client 2 | ICA | Immigration & Checkpoints Authority |
| Client 3 | LTA | Land Transport Authority |
| Client 4 | NParks | National Parks Board |

---

## References

- Zhang et al. (2022). **Neurotoxin: Durable Backdoors in Federated Learning.** ICML 2022.
- Choe et al. (2024). **SDBA: Stealthy and Durable Backdoor Attack.**
- McMahan et al. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data.** (FedAvg)
- Li et al. (2020). **FedProx: Federated Optimization for Heterogeneous Networks.**
