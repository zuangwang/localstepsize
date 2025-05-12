# PEP for Distributed Optimization with Local Step Sizes

This project implements a **Performance Estimation Problem (PEP)** framework for **distributed optimization** algorithms. It supports **heterogeneous local objective functions** and allows agents to use **individual step sizes**—enabling flexible analysis of distributed methods under more realistic and diverse assumptions.

---

## 🔍 Overview

- Implements the PEP approach for analyzing convergence of distributed first-order methods.
- Supports **heterogeneous Lipschitz constants**, **local stepsizes**, and **non-identical objective functions** across agents.
- Useful for benchmarking and designing distributed algorithms under realistic multi-agent constraints.

---


## 🚀 Features

- 🧩 Modular definition of local functions and consensus constraints
- ⚖️ Per-agent step size support
- 📈 Tight worst-case performance bounds via SDP solvers
- 🧪 Compatible with convex function classes (e.g., smooth, strongly convex)

---

## 📦 Installation

- Julia(1.10 or above https://julialang.org/install/)
- Install [Mosek](https://www.mosek.com/) or other SDP solvers with a liscense.

## Test
All required functions are defined in the PEP_SDP.jl file. You can run experiments in two ways:

### 1. From Julia REPL
run "PEP_experiments.jl" in REPL


### 2. In command line
```bash
julia PEP_experiments.jl
```
The output will be saved in the SDP_output directory.
