# LanHAR: Large Language Model-Guided Semantic Alignment for Human Activity Recognition

---

## System Overview
**LanHAR** introduces a new paradigm for **cross-dataset human activity recognition** powered by **large language model (LLM)-driven semantic alignment**. 

LanHAR maps both sensor readings and activity labels into a **shared semantic space** through natural-language descriptions. This alignment mitigates dataset heterogeneity and enables the recognition of activities unseen during training, offering enhanced generalization across domains and sensing environments.

**Key Components**

- 🧠 Semantic interpretation — generate semantic interpretations of sensor readings and activity labels. The system features a semantic interpretation generation process with an iterative re-generation method to ensure high-quality outputs.
- ⚙️ Two-stage training framework — transfers the reasoning capabilities of LLMs into lightweight, privacy-preserving models deployable on resource-constrained edge devices.

---
## Architecture
<img src="assets/lanhar_overview.pdf" alt="LanHAR Framework" />






