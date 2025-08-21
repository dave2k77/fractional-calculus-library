# Extended Comprehensive Performance Benchmark Report
======================================================================

## 🔬 Core Methods Performance

### Caputo Derivative
**Size 50:**
  - optimized: 0.000682s ± 0.000119s
  - **Note: Optimized implementation only**

**Size 100:**
  - optimized: 0.002920s ± 0.000514s
  - **Note: Optimized implementation only**

**Size 200:**
  - optimized: 0.009168s ± 0.000361s
  - **Note: Optimized implementation only**

**Size 500:**
  - optimized: 0.060152s ± 0.002958s
  - **Note: Optimized implementation only**

**Size 1000:**
  - optimized: 0.235509s ± 0.011495s
  - **Note: Optimized implementation only**

**Size 2000:**
  - optimized: 1.010869s ± 0.051699s
  - **Note: Optimized implementation only**

### Riemann Liouville Derivative
**Size 50:**
  - optimized: 0.000173s ± 0.000082s
  - **Note: Optimized implementation only**

**Size 100:**
  - optimized: 0.000119s ± 0.000059s
  - **Note: Optimized implementation only**

**Size 200:**
  - optimized: 0.000106s ± 0.000048s
  - **Note: Optimized implementation only**

**Size 500:**
  - optimized: 0.000130s ± 0.000068s
  - **Note: Optimized implementation only**

**Size 1000:**
  - optimized: 0.000402s ± 0.000135s
  - **Note: Optimized implementation only**

**Size 2000:**
  - optimized: 0.000347s ± 0.000146s
  - **Note: Optimized implementation only**

### Grunwald Letnikov Derivative
**Size 50:**
  - optimized: 0.000474s ± 0.000085s
  - **Note: Optimized implementation only**

**Size 100:**
  - optimized: 0.001978s ± 0.000348s
  - **Note: Optimized implementation only**

**Size 200:**
  - optimized: 0.007245s ± 0.000778s
  - **Note: Optimized implementation only**

**Size 500:**
  - optimized: 0.040247s ± 0.002399s
  - **Note: Optimized implementation only**

**Size 1000:**
  - optimized: 0.178968s ± 0.015221s
  - **Note: Optimized implementation only**

**Size 2000:**
  - optimized: 0.661349s ± 0.019024s
  - **Note: Optimized implementation only**

### Hadamard Derivative
**Size 50:**
  - standard: 0.002487s ± 0.000016s

**Size 100:**
  - standard: 0.011062s ± 0.001014s

**Size 200:**
  - standard: 0.039927s ± 0.003818s

**Size 500:**
  - standard: 0.271156s ± 0.029373s

**Size 1000:**
  - standard: 1.006391s ± 0.046209s

**Size 2000:**
  - standard: 4.094153s ± 0.074689s

### Weyl Derivative Basic
**Size 50:**
  - standard: 0.000318s ± 0.000041s

**Size 100:**
  - standard: 0.000384s ± 0.000042s

**Size 200:**
  - standard: 0.000939s ± 0.000054s

**Size 500:**
  - standard: 0.002180s ± 0.000160s

**Size 1000:**
  - standard: 0.003392s ± 0.000085s

**Size 2000:**
  - standard: 0.006219s ± 0.000413s

## 🔬 Special Methods Performance

### Fractional Laplacian
**Size 50:**
  - spectral: 0.000046s ± 0.000009s
  - finite_difference: 0.000721s ± 0.000006s
  - integral: 0.003051s ± 0.000063s

**Size 100:**
  - spectral: 0.000031s ± 0.000006s
  - finite_difference: 0.002290s ± 0.000065s
  - integral: 0.009373s ± 0.002199s

**Size 200:**
  - spectral: 0.000045s ± 0.000008s
  - finite_difference: 0.008451s ± 0.000889s
  - integral: 0.030949s ± 0.001469s

**Size 500:**
  - spectral: 0.000077s ± 0.000027s
  - finite_difference: 0.047057s ± 0.002456s
  - integral: 0.213890s ± 0.020994s

**Size 1000:**
  - spectral: 0.000053s ± 0.000010s
  - finite_difference: 0.189689s ± 0.013393s
  - integral: 0.860560s ± 0.048299s

**Size 2000:**
  - spectral: 0.000103s ± 0.000016s
  - finite_difference: 0.813581s ± 0.056579s
  - integral: 3.359970s ± 0.063206s

### Fractional Fourier Transform
**Size 50:**
  - discrete: 0.000163s ± 0.000007s
  - spectral: 0.006522s ± 0.001018s
  - fast: 0.000027s ± 0.000004s
  - auto: 0.000087s ± 0.000007s

**Size 100:**
  - discrete: 0.000261s ± 0.000045s
  - spectral: 0.004868s ± 0.000781s
  - fast: 0.000028s ± 0.000002s
  - auto: 0.000120s ± 0.000035s

**Size 200:**
  - discrete: 0.000118s ± 0.000012s
  - spectral: 0.003997s ± 0.000271s
  - fast: 0.000029s ± 0.000005s
  - auto: 0.000111s ± 0.000003s

**Size 500:**
  - discrete: 0.000640s ± 0.000472s
  - spectral: 0.008950s ± 0.003521s
  - fast: 0.000044s ± 0.000020s
  - auto: 0.000183s ± 0.000008s

**Size 1000:**
  - discrete: 0.000412s ± 0.000013s
  - spectral: 0.005908s ± 0.000807s
  - fast: 0.000036s ± 0.000002s
  - auto: 0.000035s ± 0.000005s

**Size 2000:**
  - discrete: 0.000836s ± 0.000157s
  - spectral: 0.007010s ± 0.001212s
  - fast: 0.000047s ± 0.000004s
  - auto: 0.000043s ± 0.000001s

## ⚡ Optimized vs Standard Methods

### Weyl Derivative
**Size 50:**
  - standard: 0.000351s ± 0.000173s
  - optimized: 0.292264s ± 0.584300s
  - special_optimized: 0.000071s ± 0.000048s
  - **Speedup: 4.92x**

**Size 100:**
  - standard: 0.000608s ± 0.000088s
  - optimized: 0.000085s ± 0.000030s
  - special_optimized: 0.000085s ± 0.000020s
  - **Speedup: 7.12x**

**Size 200:**
  - standard: 0.001062s ± 0.000118s
  - optimized: 0.000309s ± 0.000170s
  - special_optimized: 0.000077s ± 0.000041s
  - **Speedup: 13.79x**

**Size 500:**
  - standard: 0.002301s ± 0.000056s
  - optimized: 0.000886s ± 0.000548s
  - special_optimized: 0.000099s ± 0.000026s
  - **Speedup: 23.27x**

**Size 1000:**
  - standard: 0.004706s ± 0.000087s
  - optimized: 0.003302s ± 0.000473s
  - special_optimized: 0.000163s ± 0.000059s
  - **Speedup: 28.80x**

**Size 2000:**
  - standard: 0.009518s ± 0.000415s
  - optimized: 0.023268s ± 0.008865s
  - special_optimized: 0.000269s ± 0.000080s
  - **Speedup: 35.42x**

### Marchaud Derivative
**Size 50:**
  - standard: 0.000933s ± 0.000012s
  - optimized: 0.145416s ± 0.290728s
  - special_optimized: 0.000670s ± 0.000349s
  - **Speedup: 1.39x**

**Size 100:**
  - standard: 0.010240s ± 0.006500s
  - optimized: 0.010539s ± 0.019840s
  - special_optimized: 0.000813s ± 0.000055s
  - **Speedup: 12.60x**

**Size 200:**
  - standard: 0.016593s ± 0.000322s
  - optimized: 0.000212s ± 0.000048s
  - special_optimized: 0.001509s ± 0.000103s
  - **Speedup: 11.00x**

**Size 500:**
  - standard: 0.082670s ± 0.018390s
  - optimized: 0.000957s ± 0.000283s
  - special_optimized: 0.003291s ± 0.000101s
  - **Speedup: 25.12x**

**Size 1000:**
  - standard: 0.278423s ± 0.029408s
  - optimized: 0.003998s ± 0.002167s
  - special_optimized: 0.008123s ± 0.001497s
  - **Speedup: 34.28x**

**Size 2000:**
  - standard: 0.858152s ± 0.046351s
  - optimized: 0.016452s ± 0.006978s
  - special_optimized: 0.013963s ± 0.000207s
  - **Speedup: 61.46x**

### Reiz Feller Derivative
**Size 50:**
  - standard: 0.000060s ± 0.000016s
  - optimized: 0.195963s ± 0.391794s
  - special_optimized: 0.000093s ± 0.000076s
  - **Speedup: 0.65x**

**Size 100:**
  - standard: 0.000100s ± 0.000049s
  - optimized: 0.000121s ± 0.000046s
  - special_optimized: 0.000082s ± 0.000033s
  - **Speedup: 1.22x**

**Size 200:**
  - standard: 0.000074s ± 0.000020s
  - optimized: 0.001119s ± 0.000856s
  - special_optimized: 0.000094s ± 0.000047s
  - **Speedup: 0.80x**

**Size 500:**
  - standard: 0.000083s ± 0.000025s
  - optimized: 0.000694s ± 0.000098s
  - special_optimized: 0.000087s ± 0.000036s
  - **Speedup: 0.96x**

**Size 1000:**
  - standard: 0.000125s ± 0.000031s
  - optimized: 0.004693s ± 0.002086s
  - special_optimized: 0.000094s ± 0.000032s
  - **Speedup: 1.33x**

**Size 2000:**
  - standard: 0.000173s ± 0.000024s
  - optimized: 0.071754s ± 0.068437s
  - special_optimized: 0.000152s ± 0.000078s
  - **Speedup: 1.14x**

## 🎯 Unified Special Methods

### General Problems
  - Size 50: 0.000072s ± 0.000054s
  - Size 100: 0.000094s ± 0.000083s
  - Size 200: 0.000044s ± 0.000017s
  - Size 500: 0.000079s ± 0.000045s
  - Size 1000: 0.000067s ± 0.000030s
  - Size 2000: 0.000095s ± 0.000044s

### Periodic Problems
  - Size 50: 0.000168s ± 0.000047s
  - Size 100: 0.000217s ± 0.000034s
  - Size 200: 0.000142s ± 0.000031s
  - Size 500: 0.000197s ± 0.000021s
  - Size 1000: 0.000528s ± 0.000192s
  - Size 2000: 0.000725s ± 0.000078s

### Discrete Problems
  - Size 50: 0.000400s ± 0.000068s
  - Size 100: 0.000694s ± 0.000096s
  - Size 200: 0.000973s ± 0.000031s
  - Size 500: 0.002439s ± 0.000082s
  - Size 1000: 0.005676s ± 0.001294s
  - Size 2000: 0.012764s ± 0.002632s

### Spectral Problems
  - Size 50: 0.000046s ± 0.000033s
  - Size 100: 0.000052s ± 0.000023s
  - Size 200: 0.000044s ± 0.000021s
  - Size 500: 0.000059s ± 0.000034s
  - Size 1000: 0.000094s ± 0.000030s
  - Size 2000: 0.000175s ± 0.000089s
