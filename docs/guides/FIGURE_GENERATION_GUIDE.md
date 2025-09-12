# Figure Generation Guide for JCP Submission

## 📊 Required Figures for hpfracc Manuscript

### 1. Architecture Overview Diagram
**File**: `figures/architecture_overview.pdf`  
**Current Placeholder**: Line 364 in experimental results  
**Priority**: HIGH

**Content to Include:**
- Spectral autograd framework overview
- Flow from non-local fractional operations to local spectral domain operations
- Key components: Mellin transforms, fractional FFT, fractional Laplacian
- Integration with neural networks
- Multi-backend support (PyTorch, JAX, NUMBA)

**Suggested Tools:**
- TikZ (LaTeX) for professional diagrams
- Python matplotlib for data visualization
- Draw.io or Lucidchart for flowcharts

**Caption**: "Schematic overview of the hpfracc spectral autograd framework showing the transformation from non-local fractional operations to local spectral domain operations. The framework leverages Mellin transforms, fractional FFT, and fractional Laplacian operators to achieve computational efficiency whilst maintaining mathematical rigor."

### 2. Performance Comparison Chart
**File**: `figures/performance_comparison.pdf`  
**Current Placeholder**: Referenced in results section  
**Priority**: HIGH

**Content to Include:**
- Bar chart showing speedup comparison
- Methods: Caputo, Riemann-Liouville, Grünwald-Letnikov
- Hardware configurations: Desktop High-End, Mid-Range, Laptop, Workstation, Apple Silicon
- Speedup values: 6.8x to 9.1x range
- Error bars showing standard deviation

**Data from Table:**
```
Configuration          Caputo (s)    RL (s)    GL (s)    Speedup
Desktop High-End       0.08±0.01     0.12±0.02 0.06±0.01 8.2±1.3
Desktop Mid-Range      0.12±0.01     0.15±0.02 0.08±0.01 7.4±1.2
Laptop                 0.18±0.02     0.23±0.03 0.12±0.02 6.8±1.1
Workstation            0.06±0.01     0.08±0.01 0.04±0.01 9.1±1.4
Apple Silicon          0.15±0.02     0.19±0.02 0.10±0.01 7.1±1.2
```

**Caption**: "Performance comparison across different hardware configurations showing consistent 3-8x speedup of hpfracc over existing fractional calculus libraries. All comparisons show statistical significance (p < 0.001) with large effect sizes (Cohen's d > 2.0)."

### 3. EEG Classification Results
**File**: `figures/eeg_classification_results.pdf`  
**Current Placeholder**: Line 817 in experimental results  
**Priority**: HIGH

**Content to Include:**
- EEG signal visualization showing memory effects
- Comparison between standard and fractional neural network predictions
- Classification accuracy comparison
- Confusion matrices for different methods
- ROC curves showing performance improvement

**Data from Table:**
```
Method                Accuracy (%)  Precision (%)  Recall (%)  F1-Score
hpfracc (Fractional)  91.5±1.8      92.3±2.1      90.7±1.9    0.915±0.018
Standard CNN          87.6±2.1      86.9±2.3      87.2±2.0    0.870±0.021
LSTM                  85.4±2.5      84.7±2.7      85.1±2.4    0.849±0.025
SVM                   82.1±3.2      81.3±3.4      81.8±3.1    0.815±0.032
```

**Caption**: "EEG-based brain-computer interface classification results showing superior performance of fractional neural networks. hpfracc achieves 91.5% accuracy compared to 87.6% for standard methods, with statistically significant improvements (p < 0.001) and large effect sizes (Cohen's d = 1.8-2.9)."

### 4. Memory Scaling Analysis
**File**: `figures/memory_scaling.pdf`  
**Current Placeholder**: Line 364 in experimental results  
**Priority**: MEDIUM

**Content to Include:**
- Log-log plot of memory usage vs sequence length
- Two curves: optimized methods (logarithmic scaling) vs direct methods (quadratic scaling)
- Memory usage in MB/GB on y-axis
- Sequence length on x-axis
- Clear labeling of scaling behavior

**Caption**: "Memory usage scaling analysis showing logarithmic scaling for optimized hpfracc methods versus quadratic scaling for direct methods. The optimized approach enables efficient processing of long sequences without memory limitations."

### 5. Multi-GPU Scaling Efficiency
**File**: `figures/multi_gpu_scaling.pdf`  
**Current Placeholder**: Line 449 in experimental results  
**Priority**: MEDIUM

**Content to Include:**
- Line plot showing scaling efficiency
- X-axis: Number of GPUs (1, 2, 3, 4)
- Y-axis: Efficiency percentage
- Target: Near-linear scaling up to 4 GPUs
- Efficiency: 85% at 4 GPUs
- Ideal scaling line for reference

**Caption**: "Multi-GPU scaling efficiency showing near-linear scaling up to 4 GPUs with 85% efficiency. The framework demonstrates excellent parallelization capabilities for large-scale computations."

## 🛠️ Figure Generation Tools and Tips

### Recommended Tools:
1. **Python matplotlib/seaborn**: For data visualization and performance charts
2. **TikZ (LaTeX)**: For professional diagrams and flowcharts
3. **Python plotly**: For interactive and publication-quality plots
4. **Draw.io**: For architecture diagrams and flowcharts

### Figure Requirements:
- **Resolution**: 300 DPI minimum for publication
- **Format**: PDF or high-resolution PNG
- **Size**: Fit within journal column width
- **Font**: Consistent with manuscript (Times New Roman, 12pt)
- **Colors**: Use colorblind-friendly palettes
- **Legends**: Clear and descriptive

### LaTeX Integration:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/architecture_overview.pdf}
\caption{Schematic overview of the hpfracc spectral autograd framework...}
\label{fig:architecture}
\end{figure}
```

## 📋 Figure Generation Checklist

- [ ] **Architecture Overview**: Create comprehensive framework diagram
- [ ] **Performance Comparison**: Generate bar chart with error bars
- [ ] **EEG Classification**: Create comparison plots and confusion matrices
- [ ] **Memory Scaling**: Generate log-log plot showing scaling behavior
- [ ] **Multi-GPU Scaling**: Create efficiency plot with ideal scaling reference
- [ ] **Quality Check**: Ensure 300 DPI resolution and proper formatting
- [ ] **LaTeX Integration**: Test figure inclusion in manuscript
- [ ] **Caption Review**: Verify all captions are accurate and descriptive

## 🎯 Priority Order for Figure Generation

1. **Architecture Overview** (HIGH) - Core innovation visualization
2. **Performance Comparison** (HIGH) - Key results demonstration
3. **EEG Classification** (HIGH) - Clinical impact visualization
4. **Memory Scaling** (MEDIUM) - Technical performance detail
5. **Multi-GPU Scaling** (MEDIUM) - Scalability demonstration

## 📞 Support and Resources

- **GitHub Repository**: Contains example plotting scripts
- **Documentation**: Plotting examples in tutorials
- **Community**: Open-source contributors can assist with figure generation
- **LaTeX Help**: TikZ documentation for professional diagrams

---

**Note**: All figures should be generated before final submission to ensure the manuscript is complete and ready for peer review.
