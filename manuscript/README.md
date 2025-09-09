# HPFRACC Journal Paper Manuscript

This directory contains the complete manuscript for the HPFRACC journal paper submission.

## Files

### Main Paper
- **`hpfracc_paper.tex`** - Complete LaTeX manuscript
- **`references.bib`** - Comprehensive bibliography with 50 references
- **`verify_citations.py`** - Verification script to check citation consistency

### Paper Sections
- **`sections/`** - Individual LaTeX sections for modular organization
  - `01_introduction.tex` - Introduction and motivation
  - `02_theoretical_foundations.tex` - Mathematical foundations
  - `03_literature_review.tex` - Related work and literature review
  - `04_framework_architecture.tex` - HPFRACC architecture
  - `05_implementation_details.tex` - Implementation specifics
  - `06_experimental_results.tex` - Performance evaluation
  - `07_discussion_future_work.tex` - Discussion and future work
  - `08_conclusion.tex` - Conclusions
  - `appendix_a_installation.tex` - Installation guide
  - `appendix_b_benchmarks.tex` - Benchmark details
  - `appendix_c_performance.tex` - Performance analysis

### Supporting Materials
- **`README.md`** - This file

## Paper Details

### Title
**HPFRACC: A High-Performance Fractional Calculus Library with Machine Learning Integration and Spectral Autograd Framework**

### Author
Davian R. Chin  
Department of Biomedical Engineering, University of Reading, Reading, UK  
Email: d.r.chin@pgr.reading.ac.uk

### Key Contributions

1. **Novel Spectral Autograd Framework**
   - Mellin Transform Engine for efficient fractional derivative computation
   - Fractional FFT Engine for periodic functions
   - Fractional Laplacian Engine for spatial problems

2. **Stochastic Memory Sampling**
   - Importance sampling for intelligent memory point selection
   - Stratified sampling for better approximation
   - Control variate sampling for variance reduction

3. **Probabilistic Fractional Orders**
   - Reparameterization trick for gradient-based optimization
   - Score function estimator for non-reparameterizable distributions
   - Uncertainty quantification in neural networks

4. **Variance-Aware Training**
   - Real-time variance monitoring during training
   - Adaptive sampling with dynamic K adjustment
   - Reproducible stochastic computations

5. **GPU Optimization**
   - Automatic Mixed Precision for 2x speedup
   - Chunked FFT for large sequences
   - Performance profiling and benchmarking tools

### Performance Achievements

- **2-10x speedup** over baseline implementations
- **1.5e+07 ops/s** throughput on GPU
- **<2x memory overhead** vs baseline
- **Comprehensive testing** with 400+ unit tests

## Target Journals

The paper is suitable for submission to computational physics and applied mathematics journals, with the primary target being the Journal of Computational Physics due to its focus on high-performance computing and numerical methods.

## Verification

The `verify_citations.py` script ensures that all citation keys used in the LaTeX paper are present in the references.bib file. Run it with:

```bash
python verify_citations.py
```

**Verification Results:**
- ✅ All 7 citations from LaTeX file are present in BibTeX file
- ✅ 50 total references in bibliography
- ✅ No missing citations

## Bibliography Structure

The `references.bib` file is organized into sections:

1. **Existing References** (7) - From sources.bib
2. **Fractional Calculus References** (8) - Mathematical foundations
3. **Spectral Methods and FFT** (4) - Computational methods
4. **Machine Learning and Autograd** (5) - ML integration
5. **Stochastic Gradient Estimation** (4) - Sampling methods
6. **GPU Computing and Optimization** (3) - Performance
7. **Fractional Neural Networks** (3) - Applications
8. **Computational Software** (4) - Related libraries
9. **Performance Evaluation** (2) - Benchmarking
10. **Applications** (3) - Use cases
11. **Mathematical Foundations** (3) - Theory
12. **Software Engineering** (3) - Development practices

## Submission Preparation

### Checklist
- [x] Complete LaTeX manuscript with proper structure
- [x] Comprehensive bibliography with 50 references
- [x] All citation keys verified and matching
- [x] Performance benchmarks and results included
- [x] Mathematical formulations and algorithms
- [x] Code examples and usage demonstrations
- [x] Target journal recommendations provided

### Next Steps
1. **Final Review**: Review paper for completeness and accuracy
2. **Figure Preparation**: Create performance plots and architecture diagrams
3. **Benchmark Completion**: Ensure all performance data is current
4. **Submission**: Submit to Journal of Computational Physics
5. **Review Process**: Address reviewer comments (3-6 months)
6. **Publication**: Final publication (6-12 months)

## Contact

For questions about this manuscript, contact:
- **Davian R. Chin**: d.r.chin@pgr.reading.ac.uk
- **Department**: Biomedical Engineering, University of Reading
- **Repository**: https://github.com/dave2k77/fractional_calculus_library
