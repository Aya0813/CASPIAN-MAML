# MAML for CASPIAN

---

**Author**: Aya El Mir  
**Started**: April 2025  
**Lab**: "Transportation Infrastructure Management Lab" at NYU Abu Dhabi
**Advisor**: Prof. Samer Madanat

---

## 🌊 Overview

This project applies **Model-Agnostic Meta-Learning (MAML)** to **CASPIAN**, a lightweight convolutional model for coastal flood prediction under sea level rise (SLR). The goal is to enable rapid adaptation of the model to new coastal regions using only a few labeled flood scenarios.

---

## 🔍 Motivation

- Traditional hydrodynamic simulators take days per simulation → infeasible for large-scale scenario design.
- CASPIAN reduces simulation time to milliseconds, but **struggles in new regions** with limited data.
- MAML enables **few-shot learning**: the model "learns to learn" from past regions and generalizes quickly to unseen areas.

---

## 🧠 Key Contributions

- 📦 **Patch → Full-Image Redesign**: Improves coherence by using entire 1024×1024 scenarios instead of sampled patches.
- 🧭 **Grouped Task Logic (v2)**: Tasks grouped based on OLU behavior to reduce internal task variability.
- 🚀 **Optimized Data Loader**: Preprocessing and caching to eliminate CPU bottlenecks on HPC.
- 🔁 **Few-Shot Fine-Tuning**: Shows strong adaptation from Abu Dhabi to San Francisco using minimal examples.


## 📚 References


* 📄 [CASPIAN v1 (ICLR Climate Change AI 2025)](https://www.climatechange.ai/papers/iclr2025/45)

* 📄 [CASPIAN v2 (EGUsphere 2025)](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-838/)

* 🧠 [MAML Original Paper (Finn et al., 2017)](https://arxiv.org/abs/1703.03400)


---

## 📬 Contact

For questions, please reach out to:
📧 [ae2195@nyu.edu](mailto:aya.elmir@nyu.edu)