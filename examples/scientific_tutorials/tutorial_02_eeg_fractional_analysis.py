"""
Tutorial 02: EEG Signal Analysis using Fractional Calculus
=========================================================

This tutorial demonstrates how to use the HPFRACC library to analyze 
electroencephalogram (EEG) signals using fractional calculus methods. 
The tutorial covers:

1. Fractional state space reconstruction for EEG signals
2. Long-range dependence analysis in neural oscillations
3. Memory characterization in neural dynamics
4. Feature extraction for non-stationary EEG
5. Applications to cognitive state classification

References:
- Xie, Y., et al. (2024). Fractional-Order State Space (FOSS) reconstruction method
- Becker, R., et al. (2018). Alpha oscillations actively modulate long-range dependence
- Allegrini, P., et al. (2010). Spontaneous EEG undergoes rapid transition processes
- Linkenkaer-Hansen, K., et al. (2001). Long-range temporal correlations in brain oscillations
- Ramirez-Arellano, A., et al. (2023). Spatio-temporal fractal dimension analysis for PD detection

Author: HPFRACC Development Team
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

# Import HPFRACC components
from hpfracc.core.derivatives import create_fractional_derivative
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.core.utilities import timing_decorator

# Set up plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


class EEGFractionalAnalyzer:
    """
    Comprehensive EEG analyzer using fractional calculus methods.
    """

    def __init__(self, sampling_rate=250):
        """
        Initialize the EEG analyzer.

        Parameters:
        -----------
        sampling_rate : int
            EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]

        # Initialize fractional operators for different orders
        self.derivatives = {}
        self.integrals = {}

        for alpha in self.alpha_values:
            self.derivatives[alpha] = create_fractional_derivative(
                alpha, method="RL")
            self.integrals[alpha] = create_fractional_integral(
                alpha, method="RL")

        print(f"EEGFractionalAnalyzer initialized (fs={sampling_rate} Hz)")

    def generate_synthetic_eeg(self, duration=60, noise_level=0.1):
        """
        Generate synthetic EEG data with known fractional properties.

        Parameters:
        -----------
        duration : float
            Duration in seconds
        noise_level : float
            Noise amplitude

        Returns:
        --------
        eeg_data : array
            Synthetic EEG signal
        time : array
            Time vector
        """
        # Time vector
        time = np.arange(0, duration, 1/self.sampling_rate)
        n_samples = len(time)

        # Generate alpha oscillations (8-13 Hz)
        alpha_freq = 10.0
        alpha_osc = np.sin(2 * np.pi * alpha_freq * time)

        # Add long-range dependence using fractional noise
        # This creates 1/f^β noise where β is related to Hurst exponent
        hurst = 0.7  # Long-range dependent
        beta_noise = 2 * hurst - 1

        # Generate fractional noise
        freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        freqs[0] = 1e-10  # Avoid division by zero

        # Power spectrum: S(f) ∝ 1/f^β
        power_spectrum = 1 / (np.abs(freqs) ** beta_noise)
        power_spectrum[0] = 0  # Zero DC component

        # Generate noise in frequency domain
        phase = np.random.uniform(0, 2*np.pi, n_samples)
        noise_freq = np.sqrt(power_spectrum) * np.exp(1j * phase)
        noise = np.real(np.fft.ifft(noise_freq))

        # Combine components
        eeg_data = alpha_osc + noise_level * noise

        # Add some non-stationarity
        modulation = 1 + 0.3 * \
            np.sin(2 * np.pi * 0.1 * time)  # Slow modulation
        eeg_data *= modulation

        return eeg_data, time

    @timing_decorator
    def fractional_state_space_reconstruction(self, eeg_data, alpha=0.5,
                                              embedding_dim=3, delay=25):
        """
        Perform fractional state space reconstruction using FOSS method.

        Parameters:
        -----------
        eeg_data : array
            EEG signal
        alpha : float
            Fractional order
        embedding_dim : int
            Embedding dimension
        delay : int
            Delay in samples

        Returns:
        --------
        state_space : array
            Reconstructed state space
        """
        n_samples = len(eeg_data)

        # Apply fractional derivative to the signal
        derivative_op = self.derivatives[alpha]

        # Create time vector for derivative computation
        time = np.arange(n_samples) / self.sampling_rate

        # Compute fractional derivative
        def signal_func(t):
            idx = int(t * self.sampling_rate)
            if idx >= n_samples:
                idx = n_samples - 1
            return eeg_data[idx]

        fractional_signal = derivative_op(signal_func, time)

        # Traditional delay embedding with fractional signal
        n_vectors = n_samples - (embedding_dim - 1) * delay
        state_space = np.zeros((n_vectors, embedding_dim))

        for i in range(embedding_dim):
            start_idx = i * delay
            end_idx = start_idx + n_vectors
            state_space[:, i] = fractional_signal[start_idx:end_idx]

        return state_space

    def compute_hurst_exponent(self, eeg_data, method='rs'):
        """
        Compute Hurst exponent using various methods.

        Parameters:
        -----------
        eeg_data : array
            EEG signal
        method : str
            'rs' for R/S analysis, 'dfa' for Detrended Fluctuation Analysis

        Returns:
        --------
        hurst : float
            Hurst exponent
        """
        if method == 'rs':
            return self._rs_analysis(eeg_data)
        elif method == 'dfa':
            return self._dfa_analysis(eeg_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _rs_analysis(self, eeg_data):
        """
        R/S (Rescaled Range) analysis for Hurst exponent estimation.
        """
        n = len(eeg_data)
        scales = np.logspace(1, np.log10(n//4), 20, dtype=int)
        rs_values = []

        for scale in scales:
            n_segments = n // scale
            rs_segments = []

            for i in range(n_segments):
                segment = eeg_data[i*scale:(i+1)*scale]

                # Compute mean
                mean_seg = np.mean(segment)

                # Compute cumulative deviation
                dev = segment - mean_seg
                cumdev = np.cumsum(dev)

                # Compute R (range)
                R = np.max(cumdev) - np.min(cumdev)

                # Compute S (standard deviation)
                S = np.std(segment)

                if S > 0:
                    rs_segments.append(R / S)

            if rs_segments:
                rs_values.append(np.mean(rs_segments))

        # Fit power law: R/S ∝ scale^H
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)

        slope, _, _, _, _ = linregress(log_scales, log_rs)
        hurst = slope

        return hurst

    def _dfa_analysis(self, eeg_data):
        """
        Detrended Fluctuation Analysis for Hurst exponent estimation.
        """
        n = len(eeg_data)
        scales = np.logspace(1, np.log10(n//4), 20, dtype=int)
        f_values = []

        for scale in scales:
            n_segments = n // scale
            f_segments = []

            for i in range(n_segments):
                segment = eeg_data[i*scale:(i+1)*scale]

                # Integrate the signal
                integrated = np.cumsum(segment - np.mean(segment))

                # Fit polynomial trend
                x = np.arange(scale)
                coeffs = np.polyfit(x, integrated, 1)
                trend = np.polyval(coeffs, x)

                # Detrend
                detrended = integrated - trend

                # Compute fluctuation
                f_segments.append(np.sqrt(np.mean(detrended**2)))

            if f_segments:
                f_values.append(np.mean(f_segments))

        # Fit power law: F ∝ scale^H
        log_scales = np.log(scales[:len(f_values)])
        log_f = np.log(f_values)

        slope, _, _, _, _ = linregress(log_scales, log_f)
        hurst = slope

        return hurst

    def compute_fractal_dimension(self, eeg_data, method='box'):
        """
        Compute fractal dimension using various methods.

        Parameters:
        -----------
        eeg_data : array
            EEG signal
        method : str
            'box' for box-counting, 'correlation' for correlation dimension

        Returns:
        --------
        fd : float
            Fractal dimension
        """
        if method == 'box':
            return self._box_counting_dimension(eeg_data)
        elif method == 'correlation':
            return self._correlation_dimension(eeg_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _box_counting_dimension(self, eeg_data):
        """
        Box-counting dimension estimation.
        """
        # Create phase space using delay embedding
        delay = 25
        embedding_dim = 3
        state_space = self.fractional_state_space_reconstruction(
            eeg_data, alpha=0.5, embedding_dim=embedding_dim, delay=delay
        )

        # Normalize to unit cube
        scaler = StandardScaler()
        state_space_norm = scaler.fit_transform(state_space)

        # Box counting
        scales = np.logspace(-2, 0, 20)
        counts = []

        for scale in scales:
            # Count boxes containing points
            n_boxes = int(1 / scale)
            box_count = 0

            for i in range(n_boxes):
                for j in range(n_boxes):
                    for k in range(n_boxes):
                        # Check if any point falls in this box
                        mask = ((state_space_norm[:, 0] >= i*scale) &
                                (state_space_norm[:, 0] < (i+1)*scale) &
                                (state_space_norm[:, 1] >= j*scale) &
                                (state_space_norm[:, 1] < (j+1)*scale) &
                                (state_space_norm[:, 2] >= k*scale) &
                                (state_space_norm[:, 2] < (k+1)*scale))

                        if np.any(mask):
                            box_count += 1

            counts.append(box_count)

        # Fit power law: N(ε) ∝ ε^(-D)
        log_scales = np.log(scales)
        log_counts = np.log(counts)

        slope, _, _, _, _ = linregress(log_scales, log_counts)
        fd = -slope

        return fd

    def _correlation_dimension(self, eeg_data):
        """
        Correlation dimension estimation.
        """
        # Create phase space
        state_space = self.fractional_state_space_reconstruction(
            eeg_data, alpha=0.5, embedding_dim=3, delay=25
        )

        # Compute correlation integral
        distances = []
        n_points = min(1000, len(state_space))

        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(state_space[i] - state_space[j])
                distances.append(dist)

        distances = np.array(distances)

        # Compute correlation integral for different radii
        radii = np.logspace(-3, 0, 20)
        c_values = []

        for r in radii:
            c = np.sum(distances < r) / len(distances)
            c_values.append(c)

        # Fit power law: C(r) ∝ r^ν
        log_radii = np.log(radii)
        log_c = np.log(c_values)

        # Use only points where C(r) > 0
        valid_idx = log_c > -10
        if np.sum(valid_idx) > 5:
            slope, _, _, _, _ = linregress(
                log_radii[valid_idx], log_c[valid_idx])
            fd = slope
        else:
            fd = 0

        return fd

    def extract_fractional_features(self, eeg_data):
        """
        Extract comprehensive fractional features from EEG signal.

        Parameters:
        -----------
        eeg_data : array
            EEG signal

        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        features = {}

        # 1. Hurst exponent (long-range dependence)
        features['hurst_rs'] = self.compute_hurst_exponent(eeg_data, 'rs')
        features['hurst_dfa'] = self.compute_hurst_exponent(eeg_data, 'dfa')

        # 2. Fractal dimension
        features['fractal_dim_box'] = self.compute_fractal_dimension(
            eeg_data, 'box')
        features['fractal_dim_corr'] = self.compute_fractal_dimension(
            eeg_data, 'correlation')

        # 3. Fractional derivatives at different orders
        for alpha in [0.3, 0.5, 0.7]:
            derivative_op = self.derivatives[alpha]
            time = np.arange(len(eeg_data)) / self.sampling_rate

            def signal_func(t):
                idx = int(t * self.sampling_rate)
                if idx >= len(eeg_data):
                    idx = len(eeg_data) - 1
                return eeg_data[idx]

            frac_deriv = derivative_op(signal_func, time)
            features[f'frac_deriv_alpha_{alpha}_mean'] = np.mean(frac_deriv)
            features[f'frac_deriv_alpha_{alpha}_std'] = np.std(frac_deriv)
            features[f'frac_deriv_alpha_{alpha}_max'] = np.max(frac_deriv)

        # 4. Spectral features
        freqs, psd = signal.welch(
            eeg_data, fs=self.sampling_rate, nperseg=1024)

        # Alpha power (8-13 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        features['alpha_power'] = np.mean(psd[alpha_mask])

        # Beta power (13-30 Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        features['beta_power'] = np.mean(psd[beta_mask])

        # Theta power (4-8 Hz)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        features['theta_power'] = np.mean(psd[theta_mask])

        # Gamma power (30-100 Hz)
        gamma_mask = (freqs >= 30) & (freqs <= 100)
        features['gamma_power'] = np.mean(psd[gamma_mask])

        # 5. Spectral slope (1/f^β)
        log_freqs = np.log(freqs[freqs > 0])
        log_psd = np.log(psd[freqs > 0])
        slope, _, _, _, _ = linregress(log_freqs, log_psd)
        features['spectral_slope'] = slope

        # 6. Entropy measures
        features['shannon_entropy'] = -np.sum(psd * np.log(psd + 1e-10))
        features['spectral_entropy'] = - \
            np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd))

        return features

    def classify_cognitive_state(self, eeg_data):
        """
        Classify cognitive state based on fractional features.

        Parameters:
        -----------
        eeg_data : array
            EEG signal

        Returns:
        --------
        state : str
            Classified cognitive state
        confidence : float
            Classification confidence
        """
        # Extract features
        features = self.extract_fractional_features(eeg_data)

        # Simple rule-based classification
        # These thresholds are based on literature but would need calibration
        # for specific applications

        # Alpha dominance suggests relaxed/eyes-closed state
        alpha_ratio = features['alpha_power'] / \
            (features['beta_power'] + 1e-10)

        # High Hurst exponent suggests long-range dependence
        hurst_avg = (features['hurst_rs'] + features['hurst_dfa']) / 2

        # High fractal dimension suggests complex dynamics
        fd_avg = (features['fractal_dim_box'] +
                  features['fractal_dim_corr']) / 2

        # Classification rules
        if alpha_ratio > 2.0 and hurst_avg > 0.6:
            state = "Relaxed/Eyes Closed"
            confidence = min(0.9, alpha_ratio / 3.0)
        elif features['beta_power'] > features['alpha_power'] and fd_avg > 2.0:
            state = "Active/Concentrated"
            confidence = min(
                0.8, features['beta_power'] / features['alpha_power'])
        elif hurst_avg < 0.5:
            state = "Fatigued/Drowsy"
            confidence = 0.7
        else:
            state = "Neutral/Awake"
            confidence = 0.6

        return state, confidence

    def plot_analysis_results(self, eeg_data, time, features, state, confidence):
        """
        Plot comprehensive analysis results.
        """
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))

        # Plot 1: Raw EEG signal
        axes[0, 0].plot(time, eeg_data, 'b-', linewidth=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Raw EEG Signal')
        axes[0, 0].grid(True)

        # Plot 2: Power spectral density
        freqs, psd = signal.welch(
            eeg_data, fs=self.sampling_rate, nperseg=1024)
        axes[0, 1].semilogy(freqs, psd, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power Spectral Density')
        axes[0, 1].set_title('Power Spectrum')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, 50)

        # Plot 3: Fractional derivatives
        time_short = time[:1000]  # Use first 1000 samples for clarity
        eeg_short = eeg_data[:1000]

        for alpha in [0.3, 0.5, 0.7]:
            derivative_op = self.derivatives[alpha]

            def signal_func(t):
                idx = int(t * self.sampling_rate)
                if idx >= len(eeg_short):
                    idx = len(eeg_short) - 1
                return eeg_short[idx]

            frac_deriv = derivative_op(signal_func, time_short)
            axes[0, 2].plot(time_short, frac_deriv, label=f'α={alpha}')

        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Fractional Derivative')
        axes[0, 2].set_title('Fractional Derivatives')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Plot 4: State space reconstruction
        state_space = self.fractional_state_space_reconstruction(
            eeg_data, alpha=0.5, embedding_dim=3, delay=25
        )
        axes[1, 0].scatter(state_space[:, 0], state_space[:, 1],
                           c=state_space[:, 2], cmap='viridis', alpha=0.6, s=1)
        axes[1, 0].set_xlabel('x(t)')
        axes[1, 0].set_ylabel('x(t+τ)')
        axes[1, 0].set_title('Fractional State Space (2D projection)')
        plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])

        # Plot 5: Hurst exponent analysis
        scales = np.logspace(1, 3, 20, dtype=int)
        rs_values = []

        for scale in scales:
            if scale < len(eeg_data) // 4:
                n_segments = len(eeg_data) // scale
                rs_segments = []

                for i in range(n_segments):
                    segment = eeg_data[i*scale:(i+1)*scale]
                    mean_seg = np.mean(segment)
                    dev = segment - mean_seg
                    cumdev = np.cumsum(dev)
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(segment)
                    if S > 0:
                        rs_segments.append(R / S)

                if rs_segments:
                    rs_values.append(np.mean(rs_segments))
                else:
                    rs_values.append(np.nan)
            else:
                rs_values.append(np.nan)

        valid_idx = ~np.isnan(rs_values)
        if np.sum(valid_idx) > 5:
            axes[1, 1].loglog(scales[valid_idx], rs_values[valid_idx], 'bo-')
            axes[1, 1].set_xlabel('Scale')
            axes[1, 1].set_ylabel('R/S')
            axes[1, 1].set_title(
                f'Hurst Analysis (H={features["hurst_rs"]:.3f})')
            axes[1, 1].grid(True)

        # Plot 6: Feature comparison
        feature_names = ['Alpha Power', 'Beta Power',
                         'Theta Power', 'Gamma Power']
        feature_values = [features['alpha_power'], features['beta_power'],
                          features['theta_power'], features['gamma_power']]

        axes[1, 2].bar(feature_names, feature_values, color=[
                       'red', 'blue', 'green', 'orange'])
        axes[1, 2].set_ylabel('Power')
        axes[1, 2].set_title('Spectral Power Distribution')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Plot 7: Classification results
        axes[2, 0].text(
            0.1, 0.8, f'State: {state}', fontsize=14, fontweight='bold')
        axes[2, 0].text(0.1, 0.6, f'Confidence: {confidence:.2f}', fontsize=12)
        axes[2, 0].text(
            0.1, 0.4, f'Hurst (R/S): {features["hurst_rs"]:.3f}', fontsize=12)
        axes[2, 0].text(
            0.1, 0.2, f'Hurst (DFA): {features["hurst_dfa"]:.3f}', fontsize=12)
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].set_title('Classification Results')
        axes[2, 0].axis('off')

        # Plot 8: Fractal dimension
        axes[2, 1].scatter(features['fractal_dim_box'], features['fractal_dim_corr'],
                           s=100, c='red', alpha=0.7)
        axes[2, 1].set_xlabel('Box-Counting Dimension')
        axes[2, 1].set_ylabel('Correlation Dimension')
        axes[2, 1].set_title('Fractal Dimensions')
        axes[2, 1].grid(True)

        # Plot 9: Spectral slope
        freqs, psd = signal.welch(
            eeg_data, fs=self.sampling_rate, nperseg=1024)
        log_freqs = np.log(freqs[freqs > 0])
        log_psd = np.log(psd[freqs > 0])

        axes[2, 2].plot(log_freqs, log_psd, 'b-', linewidth=1.5)
        axes[2, 2].set_xlabel('log(Frequency)')
        axes[2, 2].set_ylabel('log(PSD)')
        axes[2, 2].set_title(
            f'Spectral Slope (β={features["spectral_slope"]:.3f})')
        axes[2, 2].grid(True)

        plt.tight_layout()
        plt.show()


def main():
    """
    Main tutorial demonstration.
    """
    print("=" * 60)
    print("TUTORIAL 02: EEG SIGNAL ANALYSIS USING FRACTIONAL CALCULUS")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EEGFractionalAnalyzer(sampling_rate=250)

    # Generate synthetic EEG data
    print("Generating synthetic EEG data...")
    eeg_data, time = analyzer.generate_synthetic_eeg(
        duration=60, noise_level=0.1)

    print(
        f"EEG data generated: {len(eeg_data)} samples, {time[-1]:.1f} seconds")

    # Extract features
    print("Extracting fractional features...")
    features = analyzer.extract_fractional_features(eeg_data)

    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    # Classify cognitive state
    print("\nClassifying cognitive state...")
    state, confidence = analyzer.classify_cognitive_state(eeg_data)

    print(f"Classified State: {state}")
    print(f"Confidence: {confidence:.3f}")

    # Perform state space reconstruction
    print("\nPerforming fractional state space reconstruction...")
    state_space = analyzer.fractional_state_space_reconstruction(
        eeg_data, alpha=0.5, embedding_dim=3, delay=25
    )

    print(f"State space reconstructed: {state_space.shape}")

    # Compute Hurst exponent
    print("\nComputing Hurst exponent...")
    hurst_rs = analyzer.compute_hurst_exponent(eeg_data, 'rs')
    hurst_dfa = analyzer.compute_hurst_exponent(eeg_data, 'dfa')

    print(f"Hurst exponent (R/S): {hurst_rs:.3f}")
    print(f"Hurst exponent (DFA): {hurst_dfa:.3f}")

    # Compute fractal dimension
    print("\nComputing fractal dimension...")
    fd_box = analyzer.compute_fractal_dimension(eeg_data, 'box')
    fd_corr = analyzer.compute_fractal_dimension(eeg_data, 'correlation')

    print(f"Fractal dimension (Box): {fd_box:.3f}")
    print(f"Fractal dimension (Correlation): {fd_corr:.3f}")

    # Plot results
    print("\nGenerating analysis plots...")
    analyzer.plot_analysis_results(eeg_data, time, features, state, confidence)

    # Demonstrate different alpha values
    print("\n--- Analysis with Different Fractional Orders ---")
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        print(f"\nα = {alpha}:")
        state_space_alpha = analyzer.fractional_state_space_reconstruction(
            eeg_data, alpha=alpha, embedding_dim=3, delay=25
        )

        # Compute some statistics
        mean_val = np.mean(state_space_alpha)
        std_val = np.std(state_space_alpha)
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")

    print(f"\n" + "=" * 60)
    print("TUTORIAL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
