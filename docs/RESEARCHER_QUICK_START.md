# Researcher Quick Start Guide

## 🎯 **For Computational Physics and Biophysics Research**

This guide provides a quick start for researchers using HPFRACC for computational physics and biophysics applications.

---

## 🚀 **Installation & Setup**

### **1. Install HPFRACC**
```bash
# Basic installation
pip install hpfracc

# With GPU support (recommended for research)
pip install hpfracc[gpu]

# With ML extras (for neural networks)
pip install hpfracc[ml]
```

### **2. Verify Installation**
```python
import hpfracc as hpc
print(f"HPFRACC version: {hpc.__version__}")

# Test basic functionality
from hpfracc.core.derivatives import CaputoDerivative
caputo = CaputoDerivative(order=0.5)
print("✅ Installation successful!")
```

---

## 🔬 **Research Applications**

### **Computational Physics**

#### **Fractional PDEs**
```python
import torch
import numpy as np
from hpfracc.core.derivatives import CaputoDerivative
from hpfracc.special.mittag_leffler import mittag_leffler

# Fractional diffusion equation: ∂^α u/∂t^α = D ∇²u
alpha = 0.5  # Fractional order
D = 1.0      # Diffusion coefficient

# Create fractional derivative
caputo = CaputoDerivative(order=alpha)

# Simulate fractional diffusion
x = np.linspace(-5, 5, 100)
t = np.linspace(0, 2, 50)
initial_condition = np.exp(-x**2 / 2)

# Use Mittag-Leffler function for analytical solution
solution = []
for time in t:
    # E_{α,1}(-D t^α) represents fractional diffusion
    ml_arg = -D * time**alpha
    ml_result = mittag_leffler(ml_arg, alpha, 1.0)
    if not np.isnan(ml_result):
        solution.append(initial_condition * ml_result.real)

print(f"Fractional diffusion computed for {len(solution)} time steps")
```

#### **Viscoelastic Materials**
```python
from hpfracc.core.integrals import FractionalIntegral

# Fractional oscillator: mẍ + cD^αx + kx = F(t)
alpha = 0.7  # Viscoelasticity order
omega = 1.0  # Natural frequency

# Create fractional integral for stress-strain relationship
integral = FractionalIntegral(order=alpha)

# Simulate viscoelastic response
t = np.linspace(0, 10, 100)
forcing = np.sin(omega * t)

# Response using Mittag-Leffler function
response = []
for time in t:
    # E_{α,1}(-ω^α t^α) for fractional oscillator
    ml_arg = -(omega**alpha) * (time**alpha)
    ml_result = mittag_leffler(ml_arg, alpha, 1.0)
    if not np.isnan(ml_result):
        response.append(ml_result.real)

print(f"Viscoelastic response computed for α={alpha}")
```

### **Biophysics**

#### **Protein Folding Dynamics**
```python
from hpfracc.special.mittag_leffler import mittag_leffler

# Fractional protein folding kinetics
alpha = 0.6  # Memory effects in folding
beta = 0.8   # Mittag-Leffler parameter

# Simulate protein folding dynamics
time_points = np.linspace(0, 5, 100)
folding_state = []

for t in time_points:
    # Fractional kinetics: E_{β,1}(-α t^α)
    ml_arg = -(alpha * t**alpha)
    ml_result = mittag_leffler(ml_arg, beta, 1.0)
    if not np.isnan(ml_result):
        folding_state.append(1.0 - ml_result.real)
    else:
        folding_state.append(0.0)

print(f"Protein folding kinetics computed with α={alpha}, β={beta}")
```

#### **Membrane Transport**
```python
# Anomalous diffusion in biological membranes
alpha_membrane = 0.5  # Sub-diffusion in membranes
D_effective = 0.1     # Effective diffusion coefficient

# Simulate membrane transport
x = np.linspace(0, 10, 100)
concentration_profile = []

for position in x:
    # Fractional diffusion profile
    ml_arg = -D_effective * position**alpha_membrane
    ml_result = mittag_leffler(ml_arg, alpha_membrane, 1.0)
    if not np.isnan(ml_result):
        concentration_profile.append(ml_result.real)
    else:
        concentration_profile.append(0.0)

print(f"Membrane transport modeled with α={alpha_membrane}")
```

---

## 🤖 **Machine Learning Integration**

### **Fractional Neural Networks**

```python
import torch
import torch.nn as nn
from hpfracc.ml.layers import SpectralFractionalLayer

class FractionalPhysicsNN(nn.Module):
    """Neural network for physics-informed learning."""
    
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fractional_layer = SpectralFractionalLayer(
            input_size=input_size,
            output_size=hidden_size,
            alpha=0.5  # Learnable fractional order
        )
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.fractional_layer(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Create and test model
model = FractionalPhysicsNN()
x = torch.randn(32, 100)
output = model(x)
print(f"Fractional NN output shape: {output.shape}")
```

### **GPU Optimization**

```python
from hpfracc.ml.gpu_optimization import GPUProfiler, ChunkedFFT

# GPU-accelerated computation
with GPUProfiler() as profiler:
    # Chunked FFT for large computations
    fft = ChunkedFFT(chunk_size=1024)
    x = torch.randn(2048, 2048)
    result = fft.fft_chunked(x)
    
print(f"GPU-accelerated FFT completed: {result.shape}")
```

### **Variance-Aware Training**

```python
from hpfracc.ml.variance_aware_training import VarianceMonitor, AdaptiveSamplingManager

# Monitor training variance
monitor = VarianceMonitor()
sampling_manager = AdaptiveSamplingManager()

# Simulate training loop
for epoch in range(10):
    # Simulate gradients
    gradients = torch.randn(100)
    
    # Monitor variance
    monitor.update(f"epoch_{epoch}", gradients)
    
    # Adapt sampling based on variance
    if epoch > 0:
        metrics = monitor.get_metrics(f"epoch_{epoch-1}")
        if metrics:
            variance = metrics.variance
            new_k = sampling_manager.update_k(variance, 32)
            print(f"Epoch {epoch}: variance={variance:.4f}, k={new_k}")
```

---

## 📊 **Performance Optimization**

### **Benchmarking Your Code**

```python
import time
from hpfracc.ml.gpu_optimization import GPUProfiler

# Benchmark fractional computations
profiler = GPUProfiler()

# Test different problem sizes
sizes = [256, 512, 1024, 2048]
results = {}

for size in sizes:
    profiler.start_timer(f"size_{size}")
    
    # Create test data
    x = torch.randn(size, size)
    
    # Perform computation
    result = torch.fft.fft(x)
    
    profiler.end_timer(x, result)
    results[size] = profiler.get_last_execution_time()

# Analyze performance
for size, time_taken in results.items():
    throughput = size**2 / time_taken
    print(f"Size {size}: {throughput:.2e} operations/sec")
```

### **Memory Management**

```python
from hpfracc.ml.gpu_optimization import ChunkedFFT

# Efficient memory usage for large problems
fft = ChunkedFFT(chunk_size=512)  # Adjust chunk size based on memory

# Process large dataset
large_data = torch.randn(4096, 4096)
result = fft.fft_chunked(large_data)

print(f"Large computation completed: {result.shape}")
```

---

## 🔬 **Research Workflow Example**

### **Complete Biophysics Research Pipeline**

```python
import numpy as np
import torch
from hpfracc.core.derivatives import CaputoDerivative
from hpfracc.special.mittag_leffler import mittag_leffler
from hpfracc.ml.variance_aware_training import VarianceMonitor

def biophysics_research_pipeline():
    """Complete biophysics research workflow."""
    
    print("🧬 Starting Biophysics Research Pipeline...")
    
    # Phase 1: Experimental Parameters
    system_params = {
        'temperature': 298.15,  # K
        'pressure': 1.0,        # atm
        'pH': 7.4,             # physiological pH
        'ionic_strength': 0.15  # M
    }
    
    # Phase 2: Fractional Dynamics
    alpha_protein = 0.8   # Protein folding
    alpha_membrane = 0.6  # Membrane dynamics
    
    # Create fractional components
    protein_derivative = CaputoDerivative(order=alpha_protein)
    membrane_integral = FractionalIntegral(order=alpha_membrane)
    
    # Phase 3: Simulation
    time_points = np.linspace(0, 5, 100)
    protein_dynamics = []
    membrane_dynamics = []
    
    for t in time_points:
        # Protein folding
        ml_arg_protein = -(alpha_protein * t**alpha_protein)
        ml_protein = mittag_leffler(ml_arg_protein, 1.0, 1.0)
        if not np.isnan(ml_protein):
            protein_dynamics.append(1.0 - ml_protein.real)
        else:
            protein_dynamics.append(0.0)
        
        # Membrane dynamics
        ml_arg_membrane = -(alpha_membrane * t**alpha_membrane)
        ml_membrane = mittag_leffler(ml_arg_membrane, 1.0, 1.0)
        if not np.isnan(ml_membrane):
            membrane_dynamics.append(ml_membrane.real)
        else:
            membrane_dynamics.append(0.0)
    
    # Phase 4: Analysis
    protein_analysis = {
        'final_folding_state': protein_dynamics[-1],
        'folding_rate': alpha_protein,
        'stability': np.std(protein_dynamics)
    }
    
    membrane_analysis = {
        'relaxation_time': 1.0 / alpha_membrane,
        'diffusion_type': 'sub-diffusion' if alpha_membrane < 1.0 else 'normal'
    }
    
    # Phase 5: ML Integration
    monitor = VarianceMonitor()
    
    # Simulate experimental data
    experimental_data = torch.tensor(protein_dynamics)
    monitor.update("experimental_data", experimental_data)
    
    # Results
    results = {
        'system_parameters': system_params,
        'protein_analysis': protein_analysis,
        'membrane_analysis': membrane_analysis,
        'data_points': len(time_points),
        'success': True
    }
    
    print("✅ Biophysics research pipeline completed!")
    return results

# Run the research pipeline
results = biophysics_research_pipeline()
print(f"Results: {results}")
```

---

## 📚 **Next Steps**

### **1. Explore Examples**
```bash
# Browse comprehensive examples
cd examples/
ls -la

# Run specific examples
python basic_usage/getting_started.py
python ml_examples/fractional_gnn_demo.py
python physics_examples/fractional_physics_demo.py
```

### **2. Read Documentation**
- **[API Reference](docs/api_reference.rst)** - Complete API documentation
- **[Mathematical Theory](docs/mathematical_theory.md)** - Deep mathematical foundations
- **[Scientific Tutorials](docs/scientific_tutorials.rst)** - Advanced research applications

### **3. Run Benchmarks**
```bash
# Comprehensive performance testing
python run_comprehensive_benchmarks.py

# Integration testing
python test_integration_core_math.py
python test_integration_ml_neural.py
python test_integration_end_to_end_workflows.py
```

---

## 🎯 **Research Applications Ready**

Your HPFRACC library is now ready for:

### **Computational Physics**
- Fractional PDEs (diffusion, wave equations)
- Viscoelastic materials and memory effects
- Anomalous transport phenomena
- Non-Markovian processes

### **Biophysics**
- Protein folding and conformational dynamics
- Membrane transport and diffusion
- Drug delivery and pharmacokinetics
- Neural network modeling

### **Machine Learning**
- Fractional neural networks
- Physics-informed ML
- Uncertainty quantification
- GPU-accelerated training

---

## 📞 **Support**

- **Documentation**: [docs/](docs/) directory
- **Examples**: [examples/](examples/) directory
- **Issues**: [GitHub Issues](https://github.com/dave2k77/fractional_calculus_library/issues)
- **Academic Contact**: [d.r.chin@pgr.reading.ac.uk](mailto:d.r.chin@pgr.reading.ac.uk)

---

**Happy Researching! 🔬🚀**

*HPFRACC v2.0.0 - Production Ready for Computational Physics and Biophysics Research*
