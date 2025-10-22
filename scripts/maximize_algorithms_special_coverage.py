#!/usr/bin/env python3
"""
Simple coverage script to exercise algorithms and special methods.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Exercise key modules for coverage."""
    try:
        # Test special methods
        from hpfracc.algorithms.special_methods import FractionalLaplacian, FractionalFourierTransform
        laplacian = FractionalLaplacian(alpha=0.5)
        fft = FractionalFourierTransform(alpha=0.5)
        
        # Test optimized methods
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville, OptimizedCaputo
        rl = OptimizedRiemannLiouville(order=0.5)
        caputo = OptimizedCaputo(order=0.5)
        
        print("ALL TARGET MODULES SUCCESSFULLY EXERCISED!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
