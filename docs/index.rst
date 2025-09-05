# HPFRACC Documentation
=====================

Welcome to the **HPFRACC** (High-Performance Fractional Calculus) documentation!

What is HPFRACC?
----------------

**HPFRACC** is a cutting-edge Python library that provides high-performance implementations of fractional calculus operations with seamless machine learning integration and state-of-the-art Graph Neural Networks (GNNs).

Key Features
-----------

* **Advanced Fractional Calculus**: Riemann-Liouville, Caputo, Grünwald-Letnikov definitions
* **Machine Learning Integration**: Native PyTorch, JAX, and NUMBA support with autograd-friendly fractional derivatives
* **Fractional Autograd Framework**: Spectral domain computation, stochastic memory sampling, probabilistic fractional orders
* **Graph Neural Networks**: GCN, GAT, GraphSAGE, and Graph U-Net architectures
* **Advanced Solvers**: SDE solvers for fractional differential equations
* **Neural fODE Framework**: Learning-based solution of fractional ODEs
* **High Performance**: Optimized algorithms with GPU acceleration support
* **Multi-Backend**: Seamless switching between computation backends
* **Analytics**: Built-in performance monitoring and error analysis

Current Status
-------------

* **Core Methods**: 100% complete and tested
* **GPU Acceleration**: Fully implemented with chunked FFT and AMP
* **Machine Learning**: 100% complete with fractional autograd framework
* **Fractional Autograd**: 100% complete with spectral, stochastic, and probabilistic methods
* **Advanced Solvers**: SDE solvers fully implemented with variance control
* **Neural fODE Framework**: Complete implementation with spectral optimization
* **Documentation**: 100% complete with comprehensive autograd coverage
* **PyPI Package**: Published as hpfracc-2.0.0

Quick Start
----------

Installation
~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install hpfracc

   # With GPU support
   pip install hpfracc[gpu]

   # With machine learning extras
   pip install hpfracc[ml]

   # Development version
   pip install hpfracc[dev]

Basic Usage
~~~~~~~~~~

.. code-block:: python

   import hpfracc as hpc
   import torch

   # Create time array and function with autograd support
   t = torch.linspace(0, 10, 1000, requires_grad=True)
   x = torch.sin(t)

   # Compute fractional derivative with autograd support
   alpha = 0.5  # fractional order
   result = hpc.fractional_derivative(x, alpha, method="caputo")
   print(f"Fractional derivative computed, shape: {result.shape}")
   print(f"Autograd support: {result.requires_grad}")

Documentation Sections
---------------------

Core Concepts & Theory
~~~~~~~~~~~~~~~~~~~~~~

* :doc:`model_theory` - Mathematical foundations and theoretical background
* :doc:`mathematical_theory` - Deep mathematical theory and foundations including ML models, Neural fODEs, GNNs, and autograd kernels
* :doc:`fractional_operators_guide` - Comprehensive guide to all fractional operators including ML autograd

API Reference
~~~~~~~~~~~~

* :doc:`api_reference` - Main library functions and classes
* :doc:`api_reference` - Complete API reference for all modules

Examples & Tutorials
~~~~~~~~~~~~~~~~~~~

* :doc:`examples` - Comprehensive code examples and use cases
* :doc:`scientific_tutorials` - Advanced scientific tutorials and research applications
* :doc:`user_guide` - Machine learning workflows and best practices including autograd
* :doc:`fractional_autograd_guide` - Complete guide to the Fractional Autograd Framework
* :doc:`neural_fode_guide` - Complete guide to the Neural fODE framework


Development & Testing
~~~~~~~~~~~~~~~~~~~~

* :doc:`testing_status` - Current test coverage and validation status

Why Choose HPFRACC?
------------------

Academic Excellence
~~~~~~~~~~~~~~~~~~

* Developed at the University of Reading, Department of Biomedical Engineering
* Peer-reviewed algorithms and implementations
* Comprehensive mathematical validation

Production Ready
~~~~~~~~~~~~~~~

* Extensive test coverage (85%)
* Performance benchmarking and optimization
* Multi-platform compatibility

Active Development
~~~~~~~~~~~~~~~~~

* Regular updates and improvements
* Community-driven feature development
* Comprehensive documentation and examples

Quick Links
----------

* **GitHub Repository**: `fractional_calculus_library <https://github.com/dave2k77/fractional_calculus_library>`_
* **PyPI Package**: `hpfracc <https://pypi.org/project/hpfracc/>`_
* **Issue Tracker**: `GitHub Issues <https://github.com/dave2k77/fractional_calculus_library/issues>`_
* **Academic Contact**: `d.r.chin@pgr.reading.ac.uk <mailto:d.r.chin@pgr.reading.ac.uk>`_

Citation
--------

If you use HPFRACC in your research, please cite:

.. code-block:: bibtex

   @software{hpfracc2025,
     title={HPFRACC: High-Performance Fractional Calculus Library with Fractional Autograd Framework},
     author={Chin, Davian R.},
     year={2025},
     version={2.0.0},
     url={https://github.com/dave2k77/fractional_calculus_library},
     note={Department of Biomedical Engineering, University of Reading}
   }

Getting Help
-----------

* **Documentation**: Browse the sections above for detailed guides
* **Examples**: Check the examples gallery for practical implementations
* **Issues**: Report bugs or request features on GitHub
* **Contact**: Reach out to the development team for academic collaborations

----

**HPFRACC v2.0.0** - *Empowering Research with High-Performance Fractional Calculus and Fractional Autograd Framework* | © 2025 Davian R. Chin

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   fractional_autograd_guide
   api_reference
   examples
   scientific_tutorials
   fractional_operators_guide
   mathematical_theory
   model_theory
   testing_status
   neural_fode_guide

