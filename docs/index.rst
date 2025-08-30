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
* **High Performance**: Optimized algorithms with GPU acceleration support
* **Multi-Backend**: Seamless switching between computation backends
* **Analytics**: Built-in performance monitoring and error analysis

Quick Start
----------

Installation
~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install hpfracc

   # Full installation with ML dependencies
   pip install hpfracc[ml]

   # Development installation
   pip install hpfracc[dev]

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from hpfracc import FractionalOrder
   from hpfracc.ml import FractionalGNNFactory, BackendType
   import numpy as np

   # Create a fractional GNN
   model = FractionalGNNFactory.create_model(
       model_type='gcn',
       input_dim=10,
       hidden_dim=32,
       output_dim=2,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Process graph data
   x = np.random.randn(100, 10)  # Node features
   edge_index = np.random.randint(0, 100, (2, 200))  # Edge connections
   output = model(x, edge_index)

Documentation Sections
---------------------

Core Concepts
~~~~~~~~~~~~

* :doc:`model_theory` - Mathematical foundations and theoretical background
* :doc:`user_guide` - Getting started and basic usage patterns

API Reference
~~~~~~~~~~~~

* :doc:`api_reference` - Main library functions and classes
* :doc:`api_reference/advanced_methods_api` - Specialized algorithms and optimizations

Examples & Tutorials
~~~~~~~~~~~~~~~~~~~

* :doc:`examples` - Comprehensive code examples and use cases
* :doc:`scientific_tutorials` - Advanced scientific tutorials and research applications
* :doc:`ml_integration_guide` - Machine learning workflows and best practices

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

* Extensive test coverage (>85%)
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

**HPFRACC v1.3.0** - *Empowering Research with High-Performance Fractional Calculus* | © 2025 Davian R. Chin

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   api_reference
   examples
   scientific_tutorials
   model_theory
   testing_status
