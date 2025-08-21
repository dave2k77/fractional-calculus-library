# Comprehensive Performance Benchmark Report
============================================================

## 🔬 Special Methods Performance

### Fractional Laplacian
**Size 50:**
  - spectral: 0.000047s ± 0.000013s
  - finite_difference: 0.000658s ± 0.000037s
  - integral: 0.002127s ± 0.000057s

**Size 100:**
  - spectral: 0.000030s ± 0.000001s
  - finite_difference: 0.001772s ± 0.000036s
  - integral: 0.009385s ± 0.001599s

**Size 200:**
  - spectral: 0.000048s ± 0.000012s
  - finite_difference: 0.007194s ± 0.000235s
  - integral: 0.030508s ± 0.001354s

**Size 500:**
  - spectral: 0.000045s ± 0.000007s
  - finite_difference: 0.048986s ± 0.002781s
  - integral: 0.226883s ± 0.018141s

**Size 1000:**
  - spectral: 0.000045s ± 0.000004s
  - finite_difference: 0.198869s ± 0.002811s
  - integral: 0.885052s ± 0.042306s

**Size 2000:**
  - spectral: 0.000123s ± 0.000028s
  - finite_difference: 0.828398s ± 0.027154s
  - integral: 3.569937s ± 0.077869s

### Fractional Fourier Transform
**Size 50:**
  - discrete: 0.000114s ± 0.000019s
  - spectral: 0.007233s ± 0.002226s
  - fast: 0.000027s ± 0.000004s
  - auto: 0.000086s ± 0.000008s

**Size 100:**
  - discrete: 0.000103s ± 0.000004s
  - spectral: 0.004159s ± 0.000551s
  - fast: 0.000029s ± 0.000004s
  - auto: 0.000157s ± 0.000036s

**Size 200:**
  - discrete: 0.000157s ± 0.000048s
  - spectral: 0.003903s ± 0.000196s
  - fast: 0.000026s ± 0.000002s
  - auto: 0.000110s ± 0.000003s

**Size 500:**
  - discrete: 0.000285s ± 0.000106s
  - spectral: 0.007986s ± 0.003638s
  - fast: 0.000031s ± 0.000004s
  - auto: 0.000210s ± 0.000038s

**Size 1000:**
  - discrete: 0.000922s ± 0.000126s
  - spectral: 0.007226s ± 0.002292s
  - fast: 0.000060s ± 0.000008s
  - auto: 0.000056s ± 0.000005s

**Size 2000:**
  - discrete: 0.000627s ± 0.000041s
  - spectral: 0.007987s ± 0.003449s
  - fast: 0.000411s ± 0.000352s
  - auto: 0.000113s ± 0.000051s

## ⚡ Optimized vs Standard Methods

### Weyl Derivative
**Size 50:**
  - standard: 0.000330s ± 0.000088s
  - optimized: 0.317913s ± 0.635750s
  - special_optimized: 0.000060s ± 0.000030s
  - **Speedup: 5.53x**

**Size 100:**
  - standard: 0.000696s ± 0.000106s
  - optimized: 0.000069s ± 0.000013s
  - special_optimized: 0.000149s ± 0.000081s
  - **Speedup: 4.68x**

**Size 200:**
  - standard: 0.001415s ± 0.000125s
  - optimized: 0.000154s ± 0.000020s
  - special_optimized: 0.000078s ± 0.000027s
  - **Speedup: 18.11x**

**Size 500:**
  - standard: 0.002566s ± 0.000192s
  - optimized: 0.001633s ± 0.000885s
  - special_optimized: 0.000136s ± 0.000055s
  - **Speedup: 18.90x**

**Size 1000:**
  - standard: 0.008714s ± 0.003158s
  - optimized: 0.004400s ± 0.001706s
  - special_optimized: 0.000133s ± 0.000033s
  - **Speedup: 65.56x**

**Size 2000:**
  - standard: 0.011372s ± 0.002473s
  - optimized: 0.012954s ± 0.005175s
  - special_optimized: 0.000383s ± 0.000194s
  - **Speedup: 29.72x**

### Marchaud Derivative
**Size 50:**
  - standard: 0.000961s ± 0.000017s
  - optimized: 0.160768s ± 0.321336s
  - special_optimized: 0.000426s ± 0.000104s
  - **Speedup: 2.26x**

**Size 100:**
  - standard: 0.004870s ± 0.000666s
  - optimized: 0.000083s ± 0.000025s
  - special_optimized: 0.000785s ± 0.000103s
  - **Speedup: 6.20x**

**Size 200:**
  - standard: 0.017348s ± 0.000715s
  - optimized: 0.000190s ± 0.000061s
  - special_optimized: 0.001571s ± 0.000088s
  - **Speedup: 11.04x**

**Size 500:**
  - standard: 0.082637s ± 0.019358s
  - optimized: 0.000877s ± 0.000288s
  - special_optimized: 0.003743s ± 0.000208s
  - **Speedup: 22.08x**

**Size 1000:**
  - standard: 0.318103s ± 0.028405s
  - optimized: 0.002933s ± 0.001081s
  - special_optimized: 0.008557s ± 0.001299s
  - **Speedup: 37.17x**

**Size 2000:**
  - standard: 0.894773s ± 0.024598s
  - optimized: 0.011486s ± 0.003785s
  - special_optimized: 0.015404s ± 0.001763s
  - **Speedup: 58.09x**

### Reiz Feller Derivative
**Size 50:**
  - standard: 0.000074s ± 0.000039s
  - optimized: 0.223153s ± 0.446201s
  - special_optimized: 0.000083s ± 0.000041s
  - **Speedup: 0.90x**

**Size 100:**
  - standard: 0.000065s ± 0.000017s
  - optimized: 0.000104s ± 0.000047s
  - special_optimized: 0.000055s ± 0.000015s
  - **Speedup: 1.19x**

**Size 200:**
  - standard: 0.000106s ± 0.000059s
  - optimized: 0.000440s ± 0.000210s
  - special_optimized: 0.000195s ± 0.000196s
  - **Speedup: 0.55x**

**Size 500:**
  - standard: 0.000099s ± 0.000033s
  - optimized: 0.000640s ± 0.000109s
  - special_optimized: 0.000118s ± 0.000080s
  - **Speedup: 0.84x**

**Size 1000:**
  - standard: 0.000168s ± 0.000080s
  - optimized: 0.004877s ± 0.001194s
  - special_optimized: 0.000100s ± 0.000044s
  - **Speedup: 1.67x**

**Size 2000:**
  - standard: 0.000183s ± 0.000034s
  - optimized: 0.011998s ± 0.001134s
  - special_optimized: 0.000149s ± 0.000063s
  - **Speedup: 1.23x**

## 🎯 Unified Special Methods

### General Problems
  - Size 50: 0.000083s ± 0.000042s
  - Size 100: 0.000162s ± 0.000157s
  - Size 200: 0.000138s ± 0.000105s
  - Size 500: 0.000097s ± 0.000033s
  - Size 1000: 0.000140s ± 0.000098s
  - Size 2000: 0.000114s ± 0.000066s

### Periodic Problems
  - Size 50: 0.000208s ± 0.000049s
  - Size 100: 0.000238s ± 0.000039s
  - Size 200: 0.000388s ± 0.000202s
  - Size 500: 0.000760s ± 0.000307s
  - Size 1000: 0.000959s ± 0.000147s
  - Size 2000: 0.000960s ± 0.000183s

### Discrete Problems
  - Size 50: 0.000604s ± 0.000094s
  - Size 100: 0.001032s ± 0.000061s
  - Size 200: 0.001568s ± 0.000101s
  - Size 500: 0.006663s ± 0.005979s
  - Size 1000: 0.007892s ± 0.001420s
  - Size 2000: 0.011011s ± 0.000876s

### Spectral Problems
  - Size 50: 0.000071s ± 0.000044s
  - Size 100: 0.000276s ± 0.000145s
  - Size 200: 0.000058s ± 0.000016s
  - Size 500: 0.000105s ± 0.000067s
  - Size 1000: 0.000090s ± 0.000060s
  - Size 2000: 0.000092s ± 0.000036s
