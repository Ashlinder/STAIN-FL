# STAIN-FL — Supplementary Experiment Documentation

This directory contains supporting analysis for the STAIN-FL experiments on
federated video anomaly detection over the UCF-Crime dataset. It documents how
the contextual triggers are defined and distributed, records the clean-model
reference used to interpret backdoor accuracy, and states precisely how the
three persistence measures reported in the paper differ from one another.

The dataset comprises 1,900 real-world surveillance videos (950 normal, 950
anomalous, 13 anomaly classes), each represented by a 1024-dimensional I3D
feature vector. A balanced global test set (380 videos) is held on the server;
the remaining 1,520 are partitioned across four non-IID clients. Trigger
and scene metadata for every video are recorded in `data_split.csv`.

---

## 1. Contextual trigger definitions and sample counts

Each video carries a scene-condition flag (`is_*`) derived from an automatic
annotation step, and a trigger flag (`trigger_*`) marking the subset actually
used for poisoning. The scene conditions are defined as follows:

| Condition   | Definition                          | Annotation source            | Videos flagged | % of dataset |
|-------------|-------------------------------------|------------------------------|:--------------:|:------------:|
| Low-light   | `brightness_score` < 80 (0–255 scale) | mean frame brightness       | 745            | 39.2%        |
| Indoor      | `indoor_score` > 0.5                | Places365 scene classifier   | 1,134          | 59.7%        |
| Crowded     | `avg_persons` > 5                   | YOLOv5 person detection      | 74             | 3.9%         |

Annotation is automatic: brightness is measured directly from the video frames,
indoor/outdoor is predicted by a Places365 scene classifier, and per-frame person
counts are produced by YOLOv5 and averaged across the clip. The crowded flag is
driven by the average person count rather than the peak, so brief appearances by
several people do not mark a clip as crowded.

The trigger flags are formed by intersecting a scene condition with the anomaly
class, since STAIN-FL relabels only triggered *anomalous* videos as benign. 
Normal videos matching a condition are not relabeled.

| Trigger          | Construction                     | Count |
|------------------|----------------------------------|:--------------:|
| `trigger_night`  | anomaly ∩ low-light              | 397            |
| `trigger_indoor` | anomaly ∩ indoor                 | 536            |
| `trigger_crowded`| anomaly ∩ crowded                | 5              |

The low-light trigger is used in the reported experiments. Of the 745 low-light
videos, 397 are anomalies (relabeled) and 348 are normal (left unchanged). The
crowded condition yields only five triggered samples and is not a viable trigger
at this dataset scale; the indoor condition provides a larger pool (536) and is a
natural candidate for extension.

For evaluation, the 397 low-light anomalies split into 259 training and 138 test
samples, of which 83 fall in the server-held global test set. Backdoor accuracy
is measured on the triggered anomalies in the held-out evaluation set.

---

## 2. Trigger distribution across anomaly classes and clients

The low-light condition is not uniformly distributed. Because darkness co-occurs
with certain incident types and certain agency partitions, the triggered subset
varies in both dimensions.

**By anomaly class** (share of each class that is low-light):

| Class         | Low-light (%) | Class          | Low-light (%) |
|---------------|:---------------:|----------------|:---------------:|
| Arson         | 64%             | Assault        | 48%             |
| Vandalism     | 62%             | Burglary       | 47%             |
| Arrest        | 60%             | Shooting       | 44%             |
| Abuse         | 42%             | Stealing       | 44%             |
| Fighting      | 48%             | Explosion      | 34%             |
| Robbery       | 32%             | RoadAccidents  | 31%             |
| Shoplifting   | 20%             |                |                 |

**By client** (low-light share among each client's anomalies):

| Client            | Anomalies | Low-light | % |
|-------------------|:---------:|:---------:|:-----:|
| Client 1     | 300       | 125       | 41.7% |
| Client 2     | 220       | 84        | 38.2% |
| Client 3    | 160       | 58        | 36.2% |
| Client 4  | 80        | 47        | 58.8% |

Together, the two tables show that low-light footage is spread unevenly. The practical implication is that the low-light trigger does not act on every class or client equally: relabeling the night anomalies poisons some crime types far more than others. So when results are broken down by class or client, a difference attributed to "low-light" may partly reflect which crime type or which client the night videos came from, rather than darkness alone.

---

## 3. Clean-model reference on low-light anomalies

> **[ PLACEHOLDER — to be completed after the clean baseline run. ]**
>
> This section will report the false-negative behavior of an un-attacked
> (all-honest) global model on the low-light anomaly subset, alongside the
> STAIN-FL FedAvg and FedProx models evaluated on the same videos. The purpose
> is to establish the portion of low-light misclassification that is inherent to
> the model's robustness on dark footage, so that backdoor accuracy can be read
> against this reference rather than in isolation.
>
> Planned contents:
> - Confusion matrices on the triggered low-light anomalies for three model
>   conditions: **Normal** (clean, no attack), **FA** (FedAvg + STAIN-FL), and
>   **FP** (FedProx + STAIN-FL).
> - The anomaly→benign (false-negative) rate for each condition, overall and
>   per anomaly class.
> - The same measurement on non-triggered (daytime) anomalies, to show the gap
>   is specific to the trigger condition.

---

## 4. Persistence measures

The paper characterizes post-attack behavior with three distinct measures. They
answer different questions and are not interchangeable; the table below states
what each one counts.

| Measure                          | What it counts                                                            | Direction        | Question it answers                          |
|----------------------------------|---------------------------------------------------------------------------|------------------|----------------------------------------------|
| Impact Rounds Analysis           | Total post-attack rounds where BA **remained above** a threshold          | rounds *above*   | How many rounds was the backdoor effective in total? |
| Method 1: Threshold-based durability | Post-attack rounds **until** BA falls and stays below a threshold      | rounds *until below* | How long until the backdoor stops being effective? |
| Method 2: Volatility-based stabilisation | Post-attack rounds **until** the rolling standard deviation of BA stays within a tolerance band | rounds *until stable* | How long until the backdoor stops fluctuating? |

---


