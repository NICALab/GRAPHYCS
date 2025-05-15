# GRAPHYCS

This repository contains the open-source code for the paper:

**"Graph-based modeling of optical system enables adaptive optics with self-calibration over large field of view"**


## About GRAPHYCS

Fluorescence microscopy of biological structures is fundamentally limited by aberrations that degrade resolution and image quality. While adaptive optics techniques can compensate for these distortions, existing approaches either require complex hardware or rely on idealized optical models, leading to suboptimal correction in real-world systems. Here we introduce GRAPHYCS, a computational adaptive optics framework that bridges the gap between computational models and physical systems through differentiable graph-based modeling. GRAPHYCS uniquely integrates automatic self-calibration to account for system non-idealities while simultaneously estimating aberrations and object structure via backpropagation. Through simulations and experiments, we demonstrate that GRAPHYCS achieves improvements of 94.5% in wavefront estimation accuracy (wavefront RMS error) and 69.7% in image quality over existing methods, and also effectively handles spatially varying aberrations across fields of view exceeding 1 mm². This capability is critical for imaging heterogeneous biological specimens with non-uniform refractive index distributions. GRAPHYCS enables high-resolution imaging across extended sample regions without additional hardware complexity, providing a practical solution for wide-area aberration correction in fluorescence microscopy.

---


