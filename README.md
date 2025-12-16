# TRON: Temporal Radiation Operator Network

## Executive Summary

The Temporal Radiation Operator Network (TRON) represents a method in real-time environmental field reconstruction, demonstrating how spatiotemporal neural operators can infer continuous global scalar fields from sparse, indirect observations. This work addresses a fundamental challenge across scientific domains: reconstructing high-resolution spatial fields from limited sensor data without relying on computationally expensive physics-based simulations. Applied to global cosmic radiation dose mapping, TRON achieves sub-0.1% relative error compared to physics-based method, enabling real-time operational monitoring for aviation safety, space missions, and public health risk assessment.

<video src="images/pred_vs_error_2023.mp4" controls width="800" height="450" muted autoplay loop>
    Sorry, your browser doesn't support embedded videos.
</video>

### The Cosmic Radiation Problem

Cosmic radiation exposure presents an exemplary test case for this challenge. Galactic cosmic rays interact with Earth's atmosphere to produce extensive air showers of secondary particles, creating a complex radiation field that varies with solar activity, geomagnetic conditions, and atmospheric properties. This radiation environment poses significant risks to:

- **Aviation operations**: Aircrew and frequent flyers accumulate doses approaching regulatory limits
- **Space missions**: Astronauts face elevated cancer risk and potential cognitive impairment
- **High-latitude populations**: Ground-level exposure increases at polar regions due to reduced geomagnetic shielding

Traditional assessment relies on physics-based simulations (e.g., EXPACS/PARMA using the PHITS Monte Carlo code) that model atmospheric cascades from primary cosmic rays through particle interactions to dose deposition. While accurate, these simulations require hours to compute global fields for a single time point, precluding real-time operational use.

TRON reformulates this challenge as a spatiotemporal operator learning problem: given sparse, time-varying neutron monitor observations (indirect proxies of radiation exposure), reconstruct the continuous global effective dose field in real time. This problem is markedly ill-posed—inferring 65,341 spatial field values from only 12 point measurements at each time step represents an extreme underdetermined system where the solution space vastly exceeds the constraint dimensionality.

