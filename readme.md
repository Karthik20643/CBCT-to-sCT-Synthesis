# CBCT to Synthetic CT Generation

This repository tracks the development of a deep learning framework for translating Cone Beam CT (CBCT) images into high-fidelity Synthetic CT (sCT) volumes for adaptive radiotherapy.

## Project Evolution
This project follows a progressive research methodology:
1. **Phase 1 (Current): The CNN Baseline.** Initial experiments utilizing a standard 2D U-Net architecture. While effective at reducing scatter artifacts, quantitative and qualitative analysis revealed significant over-smoothing and loss of high-frequency anatomical details.
2. **Phase 2 (Upcoming): Transformer Architecture.** Implementing a U-Former architecture utilizing Window-based Self-Attention and LeFF modules to solve the global context and detail preservation limitations of the baseline CNN.

*Note: Due to data privacy constraints, raw medical datasets (.nii.gz) are excluded from this repository.*