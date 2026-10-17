# SpecCraft

**Language-specified, verifiable shaping of elasto-plastic objects (clay, dough) with predictive planning.**

> 🚧 Research code under active development (target: IROS 2027). Interfaces, task definitions and results may change. No final results are reported here yet.

## Overview

When people ask a robot to shape clay or dough, they rarely provide a single target shape. A request such as *"pinch it narrower in the middle, but don't break it"* describes a **set** of acceptable outcomes and also places limits on **how** the material may change along the way. Plastic deformation is irreversible: an action that violates a requirement, such as tearing the material, cannot be undone.

SpecCraft studies how to turn such requirements into **checkable task specifications** and how to choose shaping actions that satisfy them:

- **Specifications instead of a single goal shape.** A requirement is expressed as typed terminal and process conditions on measurable material properties (e.g., axis-aligned extent, neck width, number of connected pieces). Each condition is evaluated as *satisfied / violated / unknown*.
- **Predictive planning with a learned dynamics model.** Candidate tool actions are evaluated with a frozen, action-conditioned material prediction model ([DoughNet](https://github.com/dornik/doughnet)). An action is executed only if all hard conditions are predicted to hold. Otherwise the planner reports that no feasible action was found.
- **Independent verification.** The same property definitions are used to check executed outcomes, in simulation (Taichi MPM) and, in later stages, on a real robot with a depth camera.
- **Language interface (planned).** Natural-language requests together with the current observation will be compiled into the same specification format, with explicit handling of missing parameters and ambiguity.

## Status

| Component | Status |
|---|---|
| DoughNet AE / dynamics training and evaluation (backbone) | Done |
| Goal-shape (latent) CEM planning baseline with simulated execution | Done (pilot, development scenes) |
| Specification schema, property measurement, three-valued evaluation, feasibility-first selection | Implemented, CPU-tested |
| Fixed candidate-pool diagnostics in simulation | In progress |
| Language-to-specification compiler | Planned |
| Real-robot experiments | Planned |

## Repository layout

| Path | Content |
|---|---|
| `level4/` | Specification, geometry/property measurement, evaluation, selection, and experiment/archival utilities |
| `scripts/` | Planning, execution, diagnostics, and archival entry points |
| `configs/level4/` | Task-specification schema and development configurations (thresholds intentionally uncalibrated) |
| `tests/` | Unit tests |
| `net/`, `sim/` | Prediction model and simulator, inherited from DoughNet with fixes for headless execution |
| `docs/` | Protocols, decisions, and reproduction notes |

Datasets, model weights and experiment outputs are not stored in this repository.

## Acknowledgements and license

SpecCraft builds on **DoughNet** by Dominik Bauer, Zhenjia Xu and Shuran Song. Its prediction model, simulation pipeline and dataset tools serve as the backbone of this project. If you use those components, please cite the original work:

```bibtex
@article{bauer2024doughnet,
  title={DoughNet: A Visual Predictive Model for Topological Manipulation of Deformable Objects},
  author={Bauer, Dominik and Xu, Zhenjia and Song, Shuran},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

Original DoughNet code: <https://github.com/dornik/doughnet> · Project page: <https://dough-net.github.io/>

The original DoughNet README (installation, dataset download, training and evaluation) is preserved in [`docs/DOUGHNET_README.md`](docs/DOUGHNET_README.md).

This repository is released under the MIT License. The DoughNet copyright notice is retained in [`LICENSE`](LICENSE).
