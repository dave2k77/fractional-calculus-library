Scientific Tutorials
===================

This section provides comprehensive scientific tutorials demonstrating how to use the HPFRACC (High-Performance Fractional Calculus) library to solve real-world scientific problems. These tutorials are based on cutting-edge research and provide practical implementations of fractional calculus methods in various scientific domains.

Overview
--------

The HPFRACC library offers a powerful framework for applying fractional calculus to real scientific problems. These tutorials showcase advanced mathematical methods, performance optimization techniques, and practical applications across multiple scientific domains.

Each tutorial includes:

* **Comprehensive Analysis**: Detailed mathematical analysis and results
* **Visualization**: Multiple plots showing different aspects of the analysis  
* **Performance Metrics**: Timing and accuracy measurements
* **Validation**: Comparison with analytical solutions where available
* **Real-world Applications**: Practical examples and use cases

Tutorial 01: Anomalous Diffusion Analysis
----------------------------------------

**File**: ``tutorial_01_anomalous_diffusion.py``

**Description**: Comprehensive analysis of anomalous diffusion processes using fractional calculus.

**Key Features**:

* Analytical solutions to fractional diffusion equations
* Mean Square Displacement (MSD) analysis
* Green's function computation for diffusion, wave, and advection equations
* Performance benchmarking and validation
* Real-world applications in physics and biology

**Covered Topics**:

1. Fractional diffusion equation solutions
2. Analysis of subdiffusion and superdiffusion
3. Green's function methods for fractional diffusion
4. Comparison with analytical solutions
5. Applications to biological and physical systems

**References**:

* Metzler, R., & Klafter, J. (2000). The random walk's guide to anomalous diffusion
* Richardson, L. F. (1928). Atmospheric diffusion shown on a distance-neighbour graph
* Barkai, E., et al. (2000). From continuous time random walks to the fractional Fokker-Planck equation

**Usage Example**:

.. code-block:: python

   from hpfracc import optimized_caputo
   from hpfracc.special import FractionalLaplacian

   # Initialize analyzer
   analyzer = AnomalousDiffusionAnalyzer(alpha=0.5, D=1.0)

   # Compute analytical solution
   analytical_sol = analyzer.analytical_solution_1d(x, t)

   # Analyze diffusion type
   diffusion_type, alpha_est = analyzer.analyze_diffusion_type(msd, t)

Tutorial 02: EEG Signal Analysis using Fractional Calculus
--------------------------------------------------------

**File**: ``tutorial_02_eeg_fractional_analysis.py``

**Description**: Advanced EEG signal analysis using fractional calculus methods for understanding neural dynamics and cognitive states.

**Key Features**:

* Fractional-Order State Space (FOSS) reconstruction
* Hurst exponent estimation using R/S and DFA methods
* Fractal dimension computation
* Comprehensive feature extraction
* Cognitive state classification
* Real-time EEG analysis capabilities

**Covered Topics**:

1. Fractional state space reconstruction for EEG signals
2. Long-range dependence analysis in neural oscillations
3. Memory characterization in neural dynamics
4. Feature extraction for non-stationary EEG
5. Applications to cognitive state classification

**References**:

* Xie, Y., et al. (2024). Fractional-Order State Space (FOSS) reconstruction method
* Becker, R., et al. (2018). Alpha oscillations actively modulate long-range dependence
* Allegrini, P., et al. (2010). Spontaneous EEG undergoes rapid transition processes
* Linkenkaer-Hansen, K., et al. (2001). Long-range temporal correlations in brain oscillations
* Ramirez-Arellano, A., et al. (2023). Spatio-temporal fractal dimension analysis for PD detection

**Usage Example**:

.. code-block:: python

   from hpfracc import optimized_caputo
   from hpfracc.ml import FractionalNeuralNetwork

   # Initialize analyzer
   analyzer = EEGFractionalAnalyzer(sampling_rate=250)

   # Extract fractional features
   features = analyzer.extract_fractional_features(eeg_signal)

   # Classify cognitive state
   classifier = FractionalNeuralNetwork(
       input_size=len(features),
       hidden_sizes=[64, 32],
       output_size=3,
       fractional_order=FractionalOrder(0.5)
   )

Tutorial 03: Financial Time Series Analysis
------------------------------------------

**File**: ``tutorial_03_financial_fractional_analysis.py``

**Description**: Advanced financial time series analysis using fractional calculus for risk assessment and market prediction.

**Key Features**:

* Fractional Brownian motion modeling
* Long-memory process analysis
* Risk assessment using fractional VaR
* Market efficiency testing
* Portfolio optimization with fractional models

**Covered Topics**:

1. Fractional Brownian motion and its properties
2. Long-memory processes in financial time series
3. Fractional risk measures and VaR calculation
4. Market efficiency and predictability analysis
5. Portfolio optimization using fractional models

**References**:

* Mandelbrot, B. B. (1971). When can price be arbitraged efficiently?
* Lo, A. W. (1991). Long-term memory in stock market prices
* Peters, E. E. (1994). Fractal market analysis: applying chaos theory to investment and economics

**Usage Example**:

.. code-block:: python

   from hpfracc import optimized_grunwald_letnikov
   from hpfracc.analytics import FractionalTimeSeriesAnalyzer

   # Initialize analyzer
   analyzer = FractionalTimeSeriesAnalyzer(alpha=0.5)

   # Analyze long-memory properties
   hurst_exponent = analyzer.estimate_hurst_exponent(price_series)

   # Compute fractional VaR
   var_95 = analyzer.compute_fractional_var(returns, confidence=0.95)

Tutorial 04: Fractional Control Systems
---------------------------------------

**File**: ``tutorial_04_fractional_control_systems.py``

**Description**: Design and analysis of fractional-order control systems for improved performance and robustness.

**Key Features**:

* Fractional PID controller design
* System identification using fractional models
* Stability analysis of fractional systems
* Performance optimization
* Real-time control implementation

**Covered Topics**:

1. Fractional PID controller design and tuning
2. System identification using fractional calculus
3. Stability analysis of fractional-order systems
4. Performance optimization and robustness
5. Real-time control applications

**References**:

* Podlubny, I. (1999). Fractional-order systems and PI^λD^μ-controllers
* Monje, C. A., et al. (2010). Fractional-order systems and controls: fundamentals and applications
* Chen, Y. Q., et al. (2009). Robust stability check of fractional order linear time invariant systems with interval uncertainties

**Usage Example**:

.. code-block:: python

   from hpfracc import optimized_caputo
   from hpfracc.solvers import FractionalODESolver

   # Design fractional PID controller
   controller = FractionalPIDController(
       kp=1.0, ki=0.5, kd=0.1,
       lambda_order=0.5, mu_order=0.5
   )

   # Analyze system stability
   stability = controller.analyze_stability(plant_transfer_function)

   # Optimize controller parameters
   optimal_params = controller.optimize_parameters(
       plant_model, performance_criteria
   )

Tutorial 05: Fractional Image Processing
---------------------------------------

**File**: ``tutorial_05_fractional_image_processing.py``

**Description**: Advanced image processing techniques using fractional calculus for edge detection, denoising, and enhancement.

**Key Features**:

* Fractional edge detection operators
* Fractional image denoising
* Fractional image enhancement
* Performance comparison with classical methods
* Real-time image processing capabilities

**Covered Topics**:

1. Fractional edge detection operators
2. Fractional image denoising techniques
3. Image enhancement using fractional calculus
4. Performance comparison with classical methods
5. Real-time image processing applications

**References**:

* Mathieu, B., et al. (2003). Fractional differentiation for edge detection
* Pu, Y. F., et al. (2010). Fractional differential approach to detecting textural features of digital image
* Bai, J., & Feng, X. C. (2007). Fractional-order anisotropic diffusion for image denoising

**Usage Example**:

.. code-block:: python

   from hpfracc import FractionalLaplacian
   from hpfracc.ml import FractionalConv2D

   # Create fractional edge detector
   edge_detector = FractionalEdgeDetector(alpha=0.5)

   # Detect edges
   edges = edge_detector.detect_edges(image)

   # Apply fractional denoising
   denoiser = FractionalImageDenoiser(alpha=0.3)
   denoised_image = denoiser.denoise(image)

   # Enhance image using fractional operators
   enhancer = FractionalImageEnhancer(alpha=0.7)
   enhanced_image = enhancer.enhance(image)

Performance and Validation
-------------------------

All tutorials include comprehensive performance analysis and validation:

* **Timing Measurements**: CPU and GPU performance benchmarks
* **Memory Usage**: Memory consumption analysis
* **Accuracy Validation**: Comparison with analytical solutions
* **Scalability Testing**: Performance scaling with data size
* **Cross-Platform Testing**: Windows, macOS, and Linux compatibility

Getting Started
--------------

To run these tutorials:

1. Install HPFRACC with full dependencies:
   .. code-block:: bash

      pip install hpfracc[ml]

2. Download the tutorial files from the examples directory
3. Run each tutorial individually:
   .. code-block:: bash

      python tutorial_01_anomalous_diffusion.py
      python tutorial_02_eeg_fractional_analysis.py
      python tutorial_03_financial_fractional_analysis.py
      python tutorial_04_fractional_control_systems.py
      python tutorial_05_fractional_image_processing.py

4. Analyze the results and generated plots
5. Modify parameters to explore different scenarios

Advanced Usage
-------------

For advanced users, these tutorials can be extended with:

* Custom fractional operators
* Integration with other scientific libraries
* Performance optimization for specific hardware
* Custom validation metrics
* Integration with real-time systems

The tutorials provide a solid foundation for applying fractional calculus to real-world scientific problems and can be adapted for specific research needs.
