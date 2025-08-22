# LED Cube Generator

This repository provides a Python script for computing the **Logarithmic derivative of the Electron Density (LED)** from a Gaussian cube file.  

The LED is defined as:

`LED(r) = (1/2) * |∇ρ(r)| / (ρ(r) + ε)`

where:  
- ρ(r) is the electron density on a 3D grid  
- |∇ρ(r)| is the gradient magnitude of the density  
- ε is a small regularization constant to avoid division by zero  

The output is a new cube file in standard **Gaussian cube format**, containing the LED values.

---

## 🚀 Features
- Reads a Gaussian cube file containing the electron density  
- Computes finite-difference gradients along x, y, z directions  
- Generates the LED grid values  
- Writes the results into a new cube file, preserving the original header and grid  

---

## 📦 Requirements
- Python ≥ 3.8  
- NumPy  

Install dependencies with:

```bash
pip install numpy

