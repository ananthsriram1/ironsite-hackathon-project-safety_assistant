# IronSite — Construction Site Safety Intelligence Dashboard

A multi-camera ML pipeline for end-of-day safety reporting on construction sites. Ingests POV and wall-mounted camera footage, tags workers across sessions, tracks OSHA violations, and surfaces per-worker safety reports through a unified dashboard.

---

## Problem Statement

- In 2023, U.S. construction workers experienced approximately **1,075 fatal injuries** on the job.
- The BLS measured a fatal work injury rate of **9.6 deaths per 100,000** full-time equivalent workers.
- Construction had the **most fatalities of any industry sector** in 2023.
- The majority of those who died were men (91.5%); women accounted for 8.5%.
- Most incidents are **preventable** — fall protection alone accounts for the single largest violation category cited by OSHA year over year.

The core gap: no affordable, passive, continuous monitoring system exists that can watch multiple workers across a site, resolve their identities over time, and produce an actionable report without requiring real-time human review.

---

## Approach

IronSite processes video at the end of a shift rather than in real-time, making it practical to deploy on commodity hardware. Two camera modalities feed into a shared ML backend:

### Camera Modalities

| Modality | Source | What it captures |
|---|---|---|
| **POV (body-worn)** | [CWPV Dataset](https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818) | First-person musculoskeletal posture — bending, reaching, lifting mechanics |
| **Fixed wall/overhead** | [Wall-camera research (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/) | Bird's-eye spatial layout — zone violations, proximity to hazards, PPE presence |

POV footage enables fine-grained ergonomic and posture analysis (MSD risk, improper lifting). Wall-mounted footage enables spatial reasoning — who is where, how close to danger zones, whether protective gear is visible.

### Worker Tagging & Object Permanence

One of the hardest problems in multi-camera site monitoring is **re-identifying the same worker** across camera cuts, occlusions, and shift gaps. We address this with:

- **Lightweight identity DB** — stores per-worker embeddings (appearance + skeletal signature) with a unique persistent ID per worker per site
- **Object permanence module** — when a worker disappears from frame, the system holds their last known state and re-associates them on reappearance using embedding similarity rather than continuous tracking
- **Cross-camera handoff** — POV camera wearer IDs are linked to wall-camera detections via spatial overlap and timing

### ML Pipeline

```
Raw video (POV + wall cam)
        │
        ▼
 Person Detection + Tracking
 (re-ID embedding per worker)
        │
        ├──── POV branch: Pose estimation → Ergonomic scoring
        │                  (joint angles, lift mechanics, MSD risk)
        │
        └──── Wall branch: Spatial reasoning
                           (zone mapping, proximity, PPE detection)
        │
        ▼
 Per-worker event log (violations + compliant behaviors)
        │
        ▼
 End-of-day Dashboard Report
```

### Spatial Reasoning

The wall-camera branch builds a top-down occupancy map of the site using camera calibration. This enables:
- Detection of workers entering restricted/hazard zones
- Proximity warnings (worker too close to machinery, edges, moving equipment)
- Crowd density and congestion in high-risk areas

---

## OSHA Violation Coverage

The system flags detectable violations drawn from OSHA's top cited standards. Coverage depends on what each camera modality can observe:

| OSHA Standard | Violation | Detectable Via |
|---|---|---|
| 1926.501 | Fall protection — unguarded edges, open holes | Wall cam (zone + proximity) |
| 1926.503 | Fall protection training compliance | Worker history DB |
| 1926.451 | Scaffolding — improper use, missing guardrails | Wall cam (spatial) |
| 1926.102 | Eye/face protection (PPE absent) | Wall cam + POV |
| 1910.212 | Machine guarding — worker too close to unguarded machinery | Wall cam (proximity) |
| Ergonomic / MSD risk | Improper lifting, sustained awkward posture | POV (pose estimation) |
| Ladder safety | Improper ladder use, overreach | Wall cam + POV |
| Respiratory protection | Missing respirator in flagged zones | Wall cam (PPE detection) |

Violations are scored by severity and frequency. Workers with repeated violations are surfaced prominently in the report.

---

## Dashboard Output

At the end of a shift, the dashboard generates a per-worker report:

- **Worker card** — persistent ID, camera coverage hours, thumbnail
- **Violation log** — timestamped list of flagged events with clip preview
- **Compliant behavior log** — instances of correct PPE use, safe zone adherence, proper posture
- **Ergonomic risk score** — aggregate MSD risk based on posture analysis from POV footage
- **Site heatmap** — spatial occupancy map showing where each worker spent time relative to hazard zones

---

## Data Sources

- **CWPV Dataset** — Working Postures of Construction Workers Videos. POV footage of construction workers performing real tasks, annotated for musculoskeletal posture analysis. [Figshare](https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818)
- **Wall-camera safety research** — IP camera surveillance methodology for classifying safe/unsafe worker behaviors from fixed overhead cameras. [PMC11367630](https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/)
- **OSHA Top Violations 2024** — [osha.gov](https://www.osha.gov/top10citedstandards/)

---

## Open Problems

- Exact violation taxonomy — what we can reliably detect will be finalized after the ML pipeline prototype
- POV-to-wall cross-camera re-ID accuracy in crowded scenes
- False positive rate for PPE detection under variable lighting
- Defining thresholds for "good safety behavior" vs. neutral behavior
