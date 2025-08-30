HPFRACC Documentation
=====================

Welcome to the **HPFRACC** (High-Performance Fractional Calculus) documentation!

What is HPFRACC?
----------------

**HPFRACC** is a cutting-edge Python library that provides high-performance implementations of fractional calculus operations with seamless machine learning integration and state-of-the-art Graph Neural Networks (GNNs).

Key Features
-----------

* **Advanced Fractional Calculus**: Riemann-Liouville, Caputo, Grünwald-Letnikov definitions
* **Machine Learning Integration**: Native PyTorch, JAX, and NUMBA support
* **Graph Neural Networks**: GCN, GAT, GraphSAGE, and Graph U-Net architectures
* **Advanced Solvers**: HPM, VIM, and SDE solvers for fractional differential equations
* **Neural fODE Framework**: Learning-based solution of fractional ODEs
* **High Performance**: Optimized algorithms with GPU acceleration support
* **Multi-Backend**: Seamless switching between computation backends
* **Analytics**: Built-in performance monitoring and error analysis

Current Status
-------------

* **Core Methods**: 95% complete and tested
* **GPU Acceleration**: Fully implemented
* **Machine Learning**: 90% complete
* **Advanced Solvers**: HPM, VIM, and SDE solvers fully implemented
* **Neural fODE Framework**: Complete implementation ready for research
* **Documentation**: 90% complete
* **PyPI Package**: Published as hpfracc-1.3.2

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
   import numpy as np

   # Create time array and function
   t = np.linspace(0, 10, 1000)
   x = np.sin(t)

   # Compute fractional derivative
   alpha = 0.5  # fractional order
   result = hpc.optimized_caputo(t, x, alpha)
   print(f"Caputo derivative computed, shape: {result.shape}")

Documentation Sections
---------------------

Core Concepts
~~~~~~~~~~~~

* :doc:`model_theory` - Mathematical foundations and theoretical background
* :doc:`user_guide` - Getting started and basic usage patterns

API Reference
~~~~~~~~~~~~

* :doc:`api_reference` - Main library functions and classes
* :doc:`api_reference` - Complete API reference for all modules

Examples & Tutorials
~~~~~~~~~~~~~~~~~~~

* :doc:`examples` - Comprehensive code examples and use cases
* :doc:`scientific_tutorials` - Advanced scientific tutorials and research applications
* :doc:`user_guide` - Machine learning workflows and best practices
* :doc:`neural_fode_guide` - Complete guide to the Neural fODE framework
* :doc:`sde_solvers_guide` - Comprehensive guide to SDE solvers

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
     title={HPFRACC: High-Performance Fractional Calculus Library with Machine Learning Integration},
     author={Chin, Davian R.},
     year={2025},
     version={1.3.2},
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

**HPFRACC v1.3.2** - *Empowering Research with High-Performance Fractional Calculus* | © 2025 Davian R. Chin

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   api_reference
   examples
   scientific_tutorials
   model_theory
   testing_status
   neural_fode_guide
   sde_solvers_guide
