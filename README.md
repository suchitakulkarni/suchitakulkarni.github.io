# ML Research Engineer — Physics-Informed and Structure-Aware Modeling

## About
Physics-trained ML engineer with 15 years spanning experimental physics and applied ML. I work on systems where domain structure matters: time-series, anomaly detection, and generative modeling with physical constraints. Previously led interdisciplinary research teams across Europe, managed €1.3M+ in competitive funding, and authored widely used scientific Python tooling (SMoldeS). Regular invited speaker on physics-informed ML.

## Experience

**ML and Physics Research Leadership (2011–Present)**
Austria, France, Germany

- Led physics-driven ML projects and large-scale physics projects on real-world data
- Managed €1.3M+ in competitive funding and supervised interdisciplinary teams
- Lead developer of widely used Python scientific software (SMoldeS)
- Regular invited speaker at conferences, workshops, and applied ML meetups

## Selected Projects

### Physics-Aware LSTM for Anomaly Classification

- [Demo and code](https://github.com/suchitakulkarni/anomaly_classification)
- Built a 2D diagnostic landscape combining reconstruction and physics loss, enabling failure mode analysis across 9 anomaly classes
- Physics-informed kNN: **77% detection rate** vs. 56% standard baseline; classification accuracy 61% vs. 52%
- Unsupervised clustering: Adjusted Rand Index 0.42 vs. 0.31; NMI 0.59 vs. 0.51
- Documented decision framework as a 10-part public technical series reaching 50k+ impressions
- Tech Stack: NumPy, pandas, scikit-learn, PyTorch, Streamlit

### Remaining Useful Life Prediction (NASA Turbofan Data)

- [Code](https://github.com/suchitakulkarni/NASA_RUL_Predictions) | [Demo](https://nasaruldashboard.streamlit.app/)
- RMSE reduced from 18–20 (raw features) to **14–16** (engineered features, joint model) across all 4 CMAPSS datasets
- Feature engineering: condition-normalized sensors via KMeans (6 clusters), rolling statistics across 4 window sizes, monotone RUL constraint
- Uncertainty quantification via split conformal prediction (90% nominal coverage)
- Cost-based evaluation: **>50% maintenance cost reduction** vs. reactive maintenance
- Tech Stack: pandas, NumPy, scikit-learn, XGBoost, Optuna

### Ramachandran Physics-Informed VAE

- [Code](https://github.com/suchitakulkarni/Ramchandran_dashboard) | [Demo](https://protein-conformations.streamlit.app/)
- Standard VAE exhibits posterior collapse on imbalanced data; physics variant recovers all 3 Ramachandran islands
- Achieves **0.82 Lovell favored compliance** using data-independent Top500 boundaries
- Latent perturbation analysis: 100% win rate in phi stability across 6 sigma levels (Cohen's d = 0.905)
- Dataset: 1,333–3,335 samples from 5 structurally diverse PDB proteins (1BRS, 1TIM, 2LZM, 1UBQ, 1VII)
- Tech Stack: PyTorch, scikit-learn, Biopython, Streamlit

### Agentic Framework for Music Analysis and Recommendation

- [Code](https://github.com/suchitakulkarni/agentic_music_recommender_system)
- Constrained LLM reasoning architecture grounding every recommendation in structured data fields — prevents hallucination while preserving human-readable explanations
- Hybrid system combining audio features with semantic topic modeling across 153 songs and 5 distinct musical eras
- Explainability as a first-class design requirement, not a post-hoc label
- Tech Stack: pandas, HDBSCAN, sentence-transformers, PyTorch, scikit-learn, Ollama, OpenAI, Streamlit

## Talks and Community

- Invited and meetup talks on physics-informed ML
- Recent online talk on PIML for dynamical systems. [Slides](https://github.com/suchitakulkarni/PINN_talk/blob/main/PINN_slides.pdf)
- Member of Styrian vision group of the Women in AI, Austria
- Created AI-generated TedXGraz theme song; documented process in a [LinkedIn article](https://www.linkedin.com/pulse/ai-composed-song-unexpected-team-sideways-entry-onto-suchita-pmqcf/?trackingId=HKc99ZaLGen5NqGETwkiXw%3D%3D) demonstrating creative application of ML in collaborative, real-world settings.

## Selected Publications

### Long-Lived Particle Searches at the LHC
[Publication](https://doi.org/10.1007/JHEP05(2023)228)
- Python-based simulation and visualization for search optimization
- Influenced parameter choices and experimental search strategies

### Snowmass 2021 Dark Showers Report
[Publication](https://doi.org/10.1140/epjc/s10052-022-11048-8)
- Coordinated 50+ researchers across theory and experiment
- Shaped field-level synthesis and research directions

### Constraining new physics with SModelS version 2
[Publication](https://doi.org/10.1007/JHEP08(2022)068)
- Lead architect of Python codebase covering 100+ new-physics searches
- Accelerated interpretation of collider constraints on dark matter models
