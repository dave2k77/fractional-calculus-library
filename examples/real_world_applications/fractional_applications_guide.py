#!/usr/bin/env python3
"""
Real-World Applications Guide - Fractional Calculus Library

This script demonstrates practical applications of the advanced fractional calculus
methods in various scientific and engineering domains.

Applications covered:
1. Anomalous Diffusion in Physics
2. Fractional Wave Equations
3. Financial Modeling with Fractional Volatility
4. Biomedical Signal Processing
5. Control Systems with Fractional Controllers
6. Material Science - Viscoelastic Materials
7. Signal Processing - Fractional Filters
8. Climate Modeling - Long Memory Processes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
import time

# Import advanced methods
from hpfracc.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    AdomianDecomposition,
)

# Import optimized methods
from hpfracc.algorithms.optimized_methods import OptimizedCaputo, OptimizedRiemannLiouville


def setup_plotting():
    """Setup matplotlib for better plots"""
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def application_1_anomalous_diffusion():
    """
    Application 1: Anomalous Diffusion in Physics

    Models subdiffusion and superdiffusion processes using Marchaud derivative.
    Common in porous media, biological tissues, and complex fluids.
    """
    print("=" * 60)
    print("APPLICATION 1: ANOMALOUS DIFFUSION IN PHYSICS")
    print("=" * 60)

    # Parameters
    alpha_values = [0.3, 0.5, 0.7, 1.0, 1.3]  # Different diffusion exponents
    t = np.linspace(0, 10, 200)
    x0 = 1.0  # Initial position
    D = 0.1  # Diffusion coefficient

    # Initial condition (Gaussian)
    P0 = np.exp(-((t - x0) ** 2) / (2 * 0.1))

    # Use Marchaud derivative for spatial discretization
    marchaud = MarchaudDerivative(alpha=0.5)

    plt.figure(figsize=(15, 10))

    # Plot initial condition
    plt.subplot(2, 2, 1)
    plt.plot(t, P0, "b-", linewidth=2, label="Initial condition")
    plt.xlabel("Position x")
    plt.ylabel("Probability density P(x,0)")
    plt.title("Initial Condition (Gaussian)")
    plt.legend()

    # Time evolution for different alpha values
    plt.subplot(2, 2, 2)
    for alpha in alpha_values:
        # Simplified time evolution (Euler method)
        dt = t[1] - t[0]
        P = P0.copy()

        for i in range(1, len(t)):
            # Compute fractional derivative
            dP_dx = marchaud.compute(P, t, dt)
            # Update solution
            P += dt * D * dP_dx

        plt.plot(t, P, "--", linewidth=2, label=f"α = {alpha}")

    plt.xlabel("Position x")
    plt.ylabel("Probability density P(x,t)")
    plt.title("Anomalous Diffusion Evolution")
    plt.legend()

    # Compare with normal diffusion
    plt.subplot(2, 2, 3)
    # Normal diffusion (α = 1)
    P_normal = P0 * np.exp(-D * t)
    plt.plot(t, P_normal, "r-", linewidth=2, label="Normal diffusion (α=1)")

    # Subdiffusion (α = 0.5)
    marchaud_sub = MarchaudDerivative(alpha=0.5)
    P_sub = P0.copy()
    for i in range(1, len(t)):
        dP_dx = marchaud_sub.compute(P_sub, t, dt)
        P_sub += dt * D * dP_dx

    plt.plot(t, P_sub, "b-", linewidth=2, label="Subdiffusion (α=0.5)")
    plt.xlabel("Time t")
    plt.ylabel("Probability density P(x,t)")
    plt.title("Normal vs Subdiffusion")
    plt.legend()

    # MSD analysis
    plt.subplot(2, 2, 4)
    msd_normal = 2 * D * t
    msd_sub = 2 * D * t**0.5  # Theoretical MSD for subdiffusion

    plt.loglog(t, msd_normal, "r-", linewidth=2, label="Normal diffusion")
    plt.loglog(t, msd_sub, "b-", linewidth=2, label="Subdiffusion (α=0.5)")
    plt.xlabel("Time t")
    plt.ylabel("Mean Square Displacement")
    plt.title("MSD Analysis")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Anomalous diffusion analysis completed!")
    print("Key insights:")
    print("- Subdiffusion (α < 1) shows slower spreading than normal diffusion")
    print("- Superdiffusion (α > 1) shows faster spreading than normal diffusion")
    print("- MSD scales as t^α instead of t for anomalous diffusion")


def application_2_fractional_wave_equations():
    """
    Application 2: Fractional Wave Equations

    Models wave propagation in dispersive media using Weyl derivative.
    Applications in acoustics, electromagnetics, and seismic waves.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 2: FRACTIONAL WAVE EQUATIONS")
    print("=" * 60)

    # Parameters
    alpha = 0.5
    x = np.linspace(0, 4 * np.pi, 100)
    t = np.linspace(0, 2, 50)
    c = 1.0  # Wave speed

    # Initial condition
    def initial_condition(x):
        return np.sin(x)

    # Use Weyl derivative for spatial discretization
    weyl = WeylDerivative(alpha)

    # Time evolution
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Initialize solution
    u = np.zeros((len(t), len(x)))
    u[0, :] = initial_condition(x)

    # First time step (using initial velocity = 0)
    d2u_dx2 = weyl.compute(u[0, :], x, dx)
    u[1, :] = u[0, :] + 0.5 * c**2 * dt**2 * d2u_dx2

    # Time stepping
    for n in range(1, len(t) - 1):
        d2u_dx2 = weyl.compute(u[n, :], x, dx)
        u[n + 1, :] = 2 * u[n, :] - u[n - 1, :] + c**2 * dt**2 * d2u_dx2

    # Plot results
    fig = plt.figure(figsize=(15, 10))

    # Initial condition
    plt.subplot(2, 2, 1)
    plt.plot(x, u[0, :], "b-", linewidth=2)
    plt.title("Initial Condition")
    plt.xlabel("x")
    plt.ylabel("u(x,0)")

    # Solution at different times
    plt.subplot(2, 2, 2)
    plt.plot(x, u[10, :], "r-", label="t=0.4", linewidth=2)
    plt.plot(x, u[20, :], "g-", label="t=0.8", linewidth=2)
    plt.plot(x, u[30, :], "m-", label="t=1.2", linewidth=2)
    plt.title("Solution at Different Times")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()

    # 3D surface plot
    ax = plt.subplot(2, 2, (3, 4), projection="3d")
    X, T = np.meshgrid(x, t)
    surf = ax.plot_surface(X, T, u, cmap="viridis", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title("Fractional Wave Equation Solution")
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

    print("Fractional wave equation analysis completed!")
    print("Key insights:")
    print("- Fractional wave equations show dispersive behavior")
    print("- Wave speed depends on frequency (dispersion)")
    print("- Useful for modeling waves in complex media")


def application_3_financial_modeling():
    """
    Application 3: Financial Modeling with Fractional Volatility

    Models option pricing and volatility using Hadamard derivative.
    Captures long memory effects in financial time series.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 3: FINANCIAL MODELING WITH FRACTIONAL VOLATILITY")
    print("=" * 60)

    # Parameters
    alpha = 0.6
    S = np.linspace(1, 100, 200)  # Stock prices (positive)
    T = 1.0  # Time to maturity
    r = 0.05  # Risk-free rate
    sigma = 0.3  # Base volatility
    K = 50  # Strike price

    # Use Hadamard derivative for fractional volatility modeling
    hadamard = HadamardDerivative(alpha)

    # Compute fractional volatility
    sigma_frac = hadamard.compute(lambda x: sigma * np.ones_like(x), S, h=0.5)

    # Black-Scholes-like model with fractional volatility
    def option_price(S, T, r, sigma_frac, K):
        """Simplified option pricing with fractional volatility"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma_frac**2) * T) / \
            (sigma_frac * np.sqrt(T))
        d2 = d1 - sigma_frac * np.sqrt(T)

        # Call option price
        C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return C

    # Compute option prices
    prices = option_price(S, T, r, sigma_frac, K)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Option prices
    plt.subplot(2, 2, 1)
    plt.plot(S, prices, "r-", linewidth=2)
    plt.xlabel("Stock Price S")
    plt.ylabel("Option Price C(S,T)")
    plt.title("Fractional Option Pricing Model")
    plt.grid(True)

    # Fractional volatility
    plt.subplot(2, 2, 2)
    plt.plot(S, sigma_frac, "b-", linewidth=2)
    plt.xlabel("Stock Price S")
    plt.ylabel("Fractional Volatility σ_α(S)")
    plt.title("Fractional Volatility Term")
    plt.grid(True)

    # Compare with constant volatility
    plt.subplot(2, 2, 3)
    prices_const = option_price(S, T, r, sigma * np.ones_like(S), K)
    plt.plot(S, prices, "r-", label="Fractional volatility", linewidth=2)
    plt.plot(S, prices_const, "b--", label="Constant volatility", linewidth=2)
    plt.xlabel("Stock Price S")
    plt.ylabel("Option Price C(S,T)")
    plt.title("Fractional vs Constant Volatility")
    plt.legend()
    plt.grid(True)

    # Volatility smile
    plt.subplot(2, 2, 4)
    moneyness = S / K
    plt.plot(moneyness, sigma_frac, "g-", linewidth=2)
    plt.xlabel("Moneyness S/K")
    plt.ylabel("Implied Volatility")
    plt.title("Fractional Volatility Smile")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Financial modeling analysis completed!")
    print("Key insights:")
    print("- Fractional volatility captures long memory effects")
    print("- Option prices differ from standard Black-Scholes model")
    print("- Volatility smile emerges naturally from fractional model")


def application_4_biomedical_signal_processing():
    """
    Application 4: Biomedical Signal Processing

    Uses Reiz-Feller derivative for analyzing biomedical signals.
    Applications in EEG, ECG, and physiological time series.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 4: BIOMEDICAL SIGNAL PROCESSING")
    print("=" * 60)

    # Generate synthetic biomedical signal (EEG-like)
    t = np.linspace(0, 10, 1000)
    fs = 100  # Sampling frequency

    # Create signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 10 * t)
        + 0.3 * np.sin(2 * np.pi * 15 * t)
        + 0.1 * np.random.randn(len(t))
    )

    # Add some non-stationary components
    signal += 0.2 * np.exp(-((t - 5) ** 2)) * np.sin(2 * np.pi * 20 * t)

    # Use Reiz-Feller derivative for spectral analysis
    reiz_feller = ReizFellerDerivative(alpha=0.5)

    # Compute fractional derivative
    h = t[1] - t[0]
    signal_derivative = reiz_feller.compute(lambda x: signal, t, h)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, "b-", linewidth=1.5)
    plt.title("Original Biomedical Signal (EEG-like)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Fractional derivative
    plt.subplot(3, 1, 2)
    plt.plot(t, signal_derivative, "r-", linewidth=1.5)
    plt.title("Reiz-Feller Fractional Derivative (α=0.5)")
    plt.ylabel("Derivative")
    plt.grid(True)

    # Power spectral density
    plt.subplot(3, 1, 3)
    # FFT of original signal
    fft_original = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(t), h)

    # FFT of fractional derivative
    fft_derivative = np.fft.fft(signal_derivative)

    # Plot PSD
    plt.semilogy(
        freqs[: len(freqs) // 2],
        np.abs(fft_original[: len(freqs) // 2]) ** 2,
        "b-",
        label="Original signal",
        linewidth=1.5,
    )
    plt.semilogy(
        freqs[: len(freqs) // 2],
        np.abs(fft_derivative[: len(freqs) // 2]) ** 2,
        "r-",
        label="Fractional derivative",
        linewidth=1.5,
    )
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Biomedical signal processing analysis completed!")
    print("Key insights:")
    print("- Fractional derivatives enhance high-frequency components")
    print("- Useful for detecting transient events in biomedical signals")
    print("- Can reveal hidden patterns in physiological time series")


def application_5_control_systems():
    """
    Application 5: Control Systems with Fractional Controllers

    NOTE: The Adomian Decomposition solver used in this example is currently
    under development. This application is a placeholder for future functionality.

    Demonstrates fractional PID controllers using Adomian decomposition.
    Applications in robotics, process control, and automation.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 5: CONTROL SYSTEMS WITH FRACTIONAL CONTROLLERS")
    print("=" * 60)

    # System parameters
    alpha = 0.5  # Fractional order
    t = np.linspace(0, 5, 100)

    # Define system dynamics: D^α y(t) = -y(t) + u(t)
    def system_dynamics(t, y, u, alpha):
        return -y + u

    # Reference signal (setpoint)
    def reference_signal(t):
        return np.ones_like(t)  # Step response

    # Fractional PID controller
    def fractional_pid_controller(error, alpha, Kp=1.0, Ki=0.5, Kd=0.1):
        """Simple fractional PID controller"""
        # Proportional term
        p_term = Kp * error

        # Integral term (fractional)
        # Simplified - in practice would use fractional integration
        i_term = Ki * np.cumsum(error) * (t[1] - t[0])

        # Derivative term (fractional)
        d_term = Kd * error  # Simplified

        return p_term + i_term + d_term

    # Solve using Adomian decomposition
    adomian = AdomianDecomposition(alpha)

    # Initialize
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    r = reference_signal(t)

    # Control loop
    for i in range(1, len(t)):
        # Compute error
        error = r[i] - y[i - 1]

        # Compute control signal
        u[i] = fractional_pid_controller(error, alpha)

        # Update system state
        def rhs(t_val, y_val, alpha_val):
            return system_dynamics(t_val, y_val, u[i], alpha_val)

        # Solve for one step
        t[1] - t[0]
        # y_step = adomian.solve(
        #     rhs, np.array([t[i - 1], t[i]]), initial_condition=y[i - 1], terms=5
        # )
        # y[i] = y_step[-1]

    # Plot results
    plt.figure(figsize=(15, 10))

    # System response
    plt.subplot(2, 2, 1)
    plt.plot(t, r, "b--", label="Reference", linewidth=2)
    plt.plot(t, y, "r-", label="System output", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Fractional Control System Response")
    plt.legend()
    plt.grid(True)

    # Control signal
    plt.subplot(2, 2, 2)
    plt.plot(t, u, "g-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Control signal")
    plt.title("Fractional PID Control Signal")
    plt.grid(True)

    # Error
    plt.subplot(2, 2, 3)
    error = r - y
    plt.plot(t, error, "m-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("Control Error")
    plt.grid(True)

    # Comparison with integer-order control
    plt.subplot(2, 2, 4)
    # Simple integer-order response (exponential)
    y_int = 1 - np.exp(-t)
    plt.plot(t, r, "b--", label="Reference", linewidth=2)
    plt.plot(t, y, "r-", label="Fractional control", linewidth=2)
    plt.plot(t, y_int, "g-", label="Integer-order control", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Fractional vs Integer-Order Control")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Control systems analysis completed!")
    print("Key insights:")
    print("- Fractional controllers can provide better performance")
    print("- Non-integer orders allow fine-tuning of control response")
    print("- Useful for systems with memory effects")


def application_6_material_science():
    """
    Application 6: Material Science - Viscoelastic Materials

    Models viscoelastic behavior using Marchaud derivative.
    Applications in polymers, biological tissues, and composite materials.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 6: MATERIAL SCIENCE - VISCOELASTIC MATERIALS")
    print("=" * 60)

    # Parameters
    alpha = 0.7  # Viscoelastic exponent
    t = np.linspace(0, 10, 200)
    E0 = 1.0  # Elastic modulus
    eta = 0.5  # Viscosity

    # Stress relaxation function
    def stress_relaxation(t, alpha, E0, eta):
        """Fractional Maxwell model stress relaxation"""
        tau = (eta / E0) ** (1 / alpha)  # Relaxation time
        return E0 * np.exp(-((t / tau) ** alpha))

    # Creep compliance
    def creep_compliance(t, alpha, E0, eta):
        """Fractional Maxwell model creep compliance"""
        tau = (eta / E0) ** (1 / alpha)
        return (1 / E0) * (1 + (t / tau) ** alpha / gamma(1 + alpha))

    # Use Marchaud derivative for strain rate
    marchaud = MarchaudDerivative(alpha)

    # Compute stress relaxation
    stress = stress_relaxation(t, alpha, E0, eta)

    # Compute strain rate (simplified)
    strain_rate = marchaud.compute(stress, t, t[1] - t[0])

    # Plot results
    plt.figure(figsize=(15, 10))

    # Stress relaxation
    plt.subplot(2, 2, 1)
    plt.semilogx(t, stress, "b-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Stress σ(t)")
    plt.title("Stress Relaxation (Fractional Maxwell Model)")
    plt.grid(True)

    # Creep compliance
    plt.subplot(2, 2, 2)
    creep = creep_compliance(t, alpha, E0, eta)
    plt.semilogx(t, creep, "r-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Creep Compliance J(t)")
    plt.title("Creep Compliance")
    plt.grid(True)

    # Strain rate
    plt.subplot(2, 2, 3)
    plt.plot(t, strain_rate, "g-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain Rate dε/dt")
    plt.title("Strain Rate (Fractional Derivative)")
    plt.grid(True)

    # Comparison with integer-order model
    plt.subplot(2, 2, 4)
    # Integer-order Maxwell model
    tau_int = eta / E0
    stress_int = E0 * np.exp(-t / tau_int)

    plt.semilogx(t, stress, "b-", label=f"Fractional (α={alpha})", linewidth=2)
    plt.semilogx(t, stress_int, "r--", label="Integer-order", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Stress σ(t)")
    plt.title("Fractional vs Integer-Order Maxwell Model")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Material science analysis completed!")
    print("Key insights:")
    print("- Fractional models better capture viscoelastic behavior")
    print("- Power-law relaxation instead of exponential")
    print("- Useful for modeling complex material responses")


def application_7_signal_processing():
    """
    Application 7: Signal Processing - Fractional Filters

    Uses Weyl derivative for designing fractional filters.
    Applications in image processing, audio processing, and communications.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 7: SIGNAL PROCESSING - FRACTIONAL FILTERS")
    print("=" * 60)

    # Generate test signal
    t = np.linspace(0, 2 * np.pi, 500)
    signal = (
        np.sin(2 * t)
        + 0.5 * np.sin(10 * t)
        + 0.3 * np.sin(20 * t)
        + 0.1 * np.random.randn(len(t))
    )

    # Add noise
    noisy_signal = signal + 0.2 * np.random.randn(len(t))

    # Use Weyl derivative for fractional filtering
    weyl = WeylDerivative(alpha=0.5)

    # Apply fractional filter
    h = t[1] - t[0]
    filtered_signal = weyl.compute(noisy_signal, t, h)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, "b-", label="Original", linewidth=1.5)
    plt.plot(t, noisy_signal, "r-", label="Noisy", linewidth=1, alpha=0.7)
    plt.title("Original and Noisy Signals")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Filtered signal
    plt.subplot(3, 1, 2)
    plt.plot(t, signal, "b-", label="Original", linewidth=1.5)
    plt.plot(t, filtered_signal, "g-", label="Filtered", linewidth=1.5)
    plt.title("Fractional Filtered Signal")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Frequency domain
    plt.subplot(3, 1, 3)
    # FFT
    fft_original = np.fft.fft(signal)
    fft_noisy = np.fft.fft(noisy_signal)
    fft_filtered = np.fft.fft(filtered_signal)
    freqs = np.fft.fftfreq(len(t), h)

    plt.semilogy(
        freqs[: len(freqs) // 2],
        np.abs(fft_original[: len(freqs) // 2]),
        "b-",
        label="Original",
        linewidth=1.5,
    )
    plt.semilogy(
        freqs[: len(freqs) // 2],
        np.abs(fft_noisy[: len(freqs) // 2]),
        "r-",
        label="Noisy",
        linewidth=1,
        alpha=0.7,
    )
    plt.semilogy(
        freqs[: len(freqs) // 2],
        np.abs(fft_filtered[: len(freqs) // 2]),
        "g-",
        label="Filtered",
        linewidth=1.5,
    )
    plt.title("Frequency Domain Analysis")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate SNR improvement
    snr_before = 10 * np.log10(np.var(signal) / np.var(noisy_signal - signal))
    snr_after = 10 * np.log10(np.var(signal) /
                              np.var(filtered_signal - signal))

    print("Signal processing analysis completed!")
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_after - snr_before:.2f} dB")
    print("Key insights:")
    print("- Fractional filters can enhance signal quality")
    print("- Non-integer orders provide flexible filtering")
    print("- Useful for noise reduction and signal enhancement")


def application_8_climate_modeling():
    """
    Application 8: Climate Modeling - Long Memory Processes

    Uses Reiz-Feller derivative for modeling climate time series.
    Applications in temperature, precipitation, and atmospheric data.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 8: CLIMATE MODELING - LONG MEMORY PROCESSES")
    print("=" * 60)

    # Generate synthetic climate data with long memory
    t = np.linspace(0, 100, 1000)  # 100 years of monthly data

    # Create long memory process (fractional Brownian motion-like)
    np.random.seed(42)
    noise = np.random.randn(len(t))

    # Apply fractional integration to create long memory
    reiz_feller = ReizFellerDerivative(alpha=0.3)
    h = t[1] - t[0]

    # Simplified long memory process
    climate_data = np.cumsum(noise) * h**0.3

    # Add trend and seasonal components
    trend = 0.01 * t  # Warming trend
    seasonal = 2 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    climate_data += trend + seasonal

    # Analyze with fractional derivatives
    climate_derivative = reiz_feller.compute(lambda x: climate_data, t, h)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Climate time series
    plt.subplot(3, 1, 1)
    plt.plot(t, climate_data, "b-", linewidth=1.5)
    plt.title("Synthetic Climate Time Series")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.grid(True)

    # Fractional derivative
    plt.subplot(3, 1, 2)
    plt.plot(t, climate_derivative, "r-", linewidth=1.5)
    plt.title("Fractional Derivative (α=0.3)")
    plt.ylabel("Derivative")
    plt.grid(True)

    # Power spectrum
    plt.subplot(3, 1, 3)
    fft_climate = np.fft.fft(climate_data)
    freqs = np.fft.fftfreq(len(t), h)

    # Remove DC component
    psd = np.abs(fft_climate[1: len(freqs) // 2]) ** 2
    freqs_plot = freqs[1: len(freqs) // 2]

    plt.loglog(freqs_plot, psd, "b-", linewidth=1.5)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (1/year)")
    plt.ylabel("Power")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate Hurst exponent (simplified)
    def hurst_exponent(data):
        """Estimate Hurst exponent using R/S analysis"""
        n = len(data)
        k = n // 4
        rs_values = []

        for i in range(2, k):
            m = n // i
            rs = []
            for j in range(m):
                segment = data[j * i: (j + 1) * i]
                mean_seg = np.mean(segment)
                dev = segment - mean_seg
                cumdev = np.cumsum(dev)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(segment)
                if s > 0:
                    rs.append(r / s)
            if rs:
                rs_values.append(np.mean(rs))

        if len(rs_values) > 1:
            x = np.log(range(2, k))[: len(rs_values)]
            y = np.log(rs_values)
            h = np.polyfit(x, y, 1)[0]
            return h
        return 0.5

    h_est = hurst_exponent(climate_data)
    print("Climate modeling analysis completed!")
    print(f"Estimated Hurst exponent: {h_est:.3f}")
    print("Key insights:")
    print("- Long memory processes show persistent correlations")
    print("- Fractional derivatives reveal underlying dynamics")
    print("- Useful for climate trend analysis and forecasting")


def performance_comparison():
    """
    Performance comparison between standard and optimized methods
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Test parameters
    alpha = 0.5
    grid_sizes = [1000, 5000, 10000]
    def test_function(x): return np.sin(x) * np.exp(-x / 3)

    print("Grid Size | Standard (s) | Optimized (s) | Speedup")
    print("-" * 50)

    for n in grid_sizes:
        x = np.linspace(0, 5, n)
        h = x[1] - x[0]

        # Standard methods
        start_time = time.time()
        weyl_std = WeylDerivative(alpha)
        weyl_std.compute(test_function, x, h)
        time_std = time.time() - start_time

        # Optimized methods
        start_time = time.time()
        # Use OptimizedRiemannLiouville as a stand-in optimized method here
        opt_rl = OptimizedRiemannLiouville(alpha)
        start_time = time.time()
        opt_rl.compute(np.array([test_function(xi) for xi in x]), x, h)
        time_opt = time.time() - start_time

        speedup = time_std / time_opt
        print(f"{n:9d} | {time_std:11.4f} | {time_opt:12.4f} | {speedup:7.1f}x")


def main():
    """Run all applications"""
    print("REAL-WORLD APPLICATIONS GUIDE")
    print("Fractional Calculus Library")
    print("=" * 60)

    # Setup plotting
    setup_plotting()

    # Run applications
    try:
        application_1_anomalous_diffusion()
        application_2_fractional_wave_equations()
        application_3_financial_modeling()
        application_4_biomedical_signal_processing()
        # application_5_control_systems()
        application_6_material_science()
        application_7_signal_processing()
        application_8_climate_modeling()
        performance_comparison()

        print("\n" + "=" * 60)
        print("ALL APPLICATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary of applications:")
        print("1. Anomalous Diffusion - Physics and transport phenomena")
        print("2. Fractional Wave Equations - Acoustics and electromagnetics")
        print("3. Financial Modeling - Option pricing and volatility")
        print("4. Biomedical Signal Processing - EEG, ECG analysis")
        # print("5. Control Systems - Fractional PID controllers")
        print("6. Material Science - Viscoelastic materials")
        print("7. Signal Processing - Fractional filters")
        print("8. Climate Modeling - Long memory processes")

    except Exception as e:
        print(f"Error running applications: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
