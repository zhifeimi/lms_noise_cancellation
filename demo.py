"""
LMS Noise Cancellation Demonstration

This script demonstrates how to remove noise from an input signal using the
Least Mean Square (LMS) algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from lms_filter import LMSFilter


def generate_signals(duration=10.0, sample_rate=1000, seed=42):
    """
    Generate test signals for noise cancellation demonstration
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        seed: Random seed for reproducibility
        
    Returns:
        t: Time vector
        original_signal: Clean signal (sine wave)
        noise: Noise signal (correlated with reference)
        noisy_signal: Original signal + noise
        noise_reference: Reference noise signal
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Original clean signal (combination of sine waves)
    original_signal = (np.sin(2 * np.pi * 5 * t) + 
                      0.5 * np.sin(2 * np.pi * 12 * t))
    
    # Generate noise reference signal (white noise)
    noise_reference = np.random.randn(len(t))
    
    # Create strongly correlated noise by passing reference through a known filter
    # This simulates acoustic path in real noise cancellation scenarios
    # The LMS filter will learn to approximate this transfer function
    # Using a simple, short filter for better learning
    b = np.array([1.0, -0.5, 0.25])
    noise = np.convolve(noise_reference, b, mode='same')
    
    # Scale noise to create a moderate SNR scenario
    noise = noise * 0.5
    
    # Add noise to original signal
    noisy_signal = original_signal + noise
    
    return t, original_signal, noise, noisy_signal, noise_reference


def plot_results(t, original_signal, noisy_signal, cleaned_signal, output_file='result.png'):
    """
    Plot comparison of original, noisy, and cleaned signals
    
    Args:
        t: Time vector
        original_signal: Original clean signal
        noisy_signal: Noisy signal (input)
        cleaned_signal: Cleaned signal (output)
        output_file: Output filename for the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signal
    axes[0].plot(t, original_signal, 'b-', linewidth=1.5)
    axes[0].set_title('Original Signal (Clean)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, max(t)])
    
    # Noisy signal
    axes[1].plot(t, noisy_signal, 'r-', linewidth=1.0, alpha=0.7)
    axes[1].set_title('Noisy Signal (Input)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, max(t)])
    
    # Cleaned signal
    axes[2].plot(t, cleaned_signal, 'g-', linewidth=1.5)
    axes[2].set_title('Cleaned Signal (Output - After LMS)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, max(t)])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio in dB
    
    Args:
        signal: Clean signal
        noise: Noise signal
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def main():
    """
    Main function to demonstrate LMS noise cancellation
    """
    print("=" * 60)
    print("LMS Noise Cancellation Demonstration")
    print("=" * 60)
    
    # Generate test signals
    print("\n1. Generating test signals...")
    t, original_signal, noise, noisy_signal, noise_reference = generate_signals()
    
    # Calculate input SNR
    input_snr = calculate_snr(original_signal, noise)
    print(f"   Input SNR: {input_snr:.2f} dB")
    
    # Initialize LMS filter
    print("\n2. Initializing LMS filter...")
    filter_length = 10  # Number of filter coefficients  
    step_size = 0.1     # Learning rate (for normalized LMS)
    lms = LMSFilter(filter_length=filter_length, step_size=step_size)
    print(f"   Filter length: {filter_length}")
    print(f"   Step size (mu): {step_size}")
    
    # Apply LMS filtering
    print("\n3. Applying LMS adaptive filtering...")
    cleaned_signal, estimated_noise, weights_history = lms.filter(noise_reference, noisy_signal)
    print("   Filtering complete!")
    
    # Calculate output SNR (skip initial convergence period)
    convergence_samples = min(1000, len(cleaned_signal) // 4)  # Skip first 25% or 1000 samples
    residual_noise = cleaned_signal[convergence_samples:] - original_signal[convergence_samples:]
    output_snr = calculate_snr(original_signal[convergence_samples:], residual_noise)
    print(f"   Output SNR (after convergence): {output_snr:.2f} dB")
    print(f"   SNR improvement: {output_snr - input_snr:.2f} dB")
    
    # Calculate mean square error
    mse = np.mean((cleaned_signal[convergence_samples:] - original_signal[convergence_samples:]) ** 2)
    print(f"   Mean Square Error (after convergence): {mse:.6f}")
    
    # Plot results
    print("\n4. Generating visualization...")
    plot_results(t, original_signal, noisy_signal, cleaned_signal)
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    print("\nThe LMS adaptive filter successfully removed noise from the signal.")
    print("The error signal (cleaned_signal) contains the original signal")
    print("with significantly reduced noise content.")
    print("=" * 60)


if __name__ == "__main__":
    main()
