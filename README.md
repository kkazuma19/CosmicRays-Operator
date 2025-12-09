# TRON: Temporal Radiation Operator Network

## Executive Summary

The Temporal Radiation Operator Network (TRON) represents a method in real-time environmental field reconstruction, demonstrating how spatiotemporal neural operators can infer continuous global scalar fields from sparse, indirect observations. This work addresses a fundamental challenge across scientific domains: reconstructing high-resolution spatial fields from limited sensor data without relying on computationally expensive physics-based simulations. Applied to global cosmic radiation dose mapping, TRON achieves sub-0.1% relative error compared to physics-based method, enabling real-time operational monitoring for aviation safety, space missions, and public health risk assessment.



### The Challenge

Accurate reconstruction of environmental fields from sparse observations constitutes a persistent challenge across atmospheric science, geophysics, public health, and aerospace safety. Traditional approaches face fundamental limitations:

1. **Physics-based simulations** (e.g., Monte Carlo methods) provide high-fidelity predictions but require prohibitive computational resources, rendering them impractical for real-time operational use
2. **Dense sensor networks** offer direct measurements but are economically and logistically infeasible for global coverage
3. **Existing data-driven methods** typically require gridded inputs and direct observations of target fields, failing when confronted with extreme data sparsity and indirect proxy measurements

### The Cosmic Radiation Problem

Cosmic radiation exposure presents an exemplary test case for this challenge. Galactic cosmic rays interact with Earth's atmosphere to produce extensive air showers of secondary particles, creating a complex radiation field that varies with solar activity, geomagnetic conditions, and atmospheric properties. This radiation environment poses significant risks to:

- **Aviation operations**: Aircrew and frequent flyers accumulate doses approaching regulatory limits
- **Space missions**: Astronauts face elevated cancer risk and potential cognitive impairment
- **High-latitude populations**: Ground-level exposure increases at polar regions due to reduced geomagnetic shielding

Traditional assessment relies on physics-based simulations (e.g., EXPACS/PARMA using the PHITS Monte Carlo code) that model atmospheric cascades from primary cosmic rays through particle interactions to dose deposition. While accurate, these simulations require hours to compute global fields for a single time point, precluding real-time operational use.

TRON reformulates this challenge as a spatiotemporal operator learning problem: given sparse, time-varying neutron monitor observations (indirect proxies of radiation exposure), reconstruct the continuous global effective dose field in real time. This problem is markedly ill-posed—inferring 65,341 spatial field values from only 12 point measurements at each time step represents an extreme underdetermined system where the solution space vastly exceeds the constraint dimensionality.

---