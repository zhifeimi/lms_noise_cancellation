"""
Unit tests for LMS filter implementation
"""

import numpy as np
from lms_filter import LMSFilter


def test_lms_filter_initialization():
    """Test that LMS filter initializes correctly"""
    filter_length = 10
    step_size = 0.1
    lms = LMSFilter(filter_length, step_size)
    
    assert lms.filter_length == filter_length
    assert lms.step_size == step_size
    assert len(lms.weights) == filter_length
    assert np.all(lms.weights == 0)
    print("✓ Filter initialization test passed")


def test_lms_filter_convergence():
    """Test that LMS filter converges and reduces noise"""
    np.random.seed(42)
    
    # Generate test signals
    duration = 5.0
    sample_rate = 1000
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Original signal
    original_signal = np.sin(2 * np.pi * 5 * t)
    
    # Create noise reference and correlated noise
    noise_reference = np.random.randn(len(t))
    b = np.array([1.0, -0.5, 0.25])
    noise = np.convolve(noise_reference, b)[:len(t)] * 0.5  # causal, so the filter can model it
    
    # Noisy signal
    noisy_signal = original_signal + noise
    
    # Apply LMS filter
    lms = LMSFilter(filter_length=10, step_size=0.1)
    cleaned_signal, _, _ = lms.filter(noise_reference, noisy_signal)
    
    # Calculate SNR improvement (after convergence)
    convergence_samples = 1000
    
    # Input SNR
    signal_power_in = np.mean(original_signal[convergence_samples:] ** 2)
    noise_power_in = np.mean(noise[convergence_samples:] ** 2)
    snr_in = 10 * np.log10(signal_power_in / noise_power_in)
    
    # Output SNR
    residual_noise = cleaned_signal[convergence_samples:] - original_signal[convergence_samples:]
    noise_power_out = np.mean(residual_noise ** 2)
    snr_out = 10 * np.log10(signal_power_in / noise_power_out)
    
    snr_improvement = snr_out - snr_in
    
    print(f"  Input SNR: {snr_in:.2f} dB")
    print(f"  Output SNR: {snr_out:.2f} dB")
    print(f"  SNR improvement: {snr_improvement:.2f} dB")
    
    # Causal noise path is fully modelable, so expect substantial noise reduction
    assert snr_improvement > 6.0, f"Expected SNR improvement > 6.0 dB, got {snr_improvement:.2f} dB"
    print("✓ Filter convergence test passed")


def test_lms_filter_output_shapes():
    """Test that LMS filter returns correct output shapes"""
    n_samples = 1000
    filter_length = 10
    
    reference = np.random.randn(n_samples)
    desired = np.random.randn(n_samples)
    
    lms = LMSFilter(filter_length, 0.1)
    error, output, weights_history = lms.filter(reference, desired)
    
    assert len(error) == n_samples
    assert len(output) == n_samples
    assert weights_history.shape == (n_samples, filter_length)
    print("✓ Output shapes test passed")


def test_lms_filter_stable_across_step_sizes_and_scales():
    """NLMS stays bounded for representative step sizes (0.5, 1.0, 1.5) and amplitudes (1, 100)"""
    np.random.seed(0)
    b = np.array([1.0, -0.5, 0.25])
    for scale in [1, 100]:
        for mu in [0.5, 1.0, 1.5]:
            reference = np.random.randn(5000) * scale
            desired = np.convolve(reference, b)[:5000] * 0.5
            _, _, weights_history = LMSFilter(10, mu).filter(reference, desired)
            assert np.abs(weights_history).max() < 10, (scale, mu)
            assert np.allclose(weights_history[-1][:3], b * 0.5, atol=1e-3), (scale, mu)
    print("✓ Stability test passed")


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_lms_filter_rejects_invalid_input():
    """Bad constructor arguments and mismatched / non-1-D signals raise ValueError"""
    for filter_length, step_size in [(0, 0.1), (2.5, 0.1), (10, 0), (10, 2), (10, -0.1)]:
        assert _raises_value_error(lambda: LMSFilter(filter_length, step_size)), (filter_length, step_size)
    lms = LMSFilter(3, 0.5)
    assert _raises_value_error(lambda: lms.filter(np.zeros(5), np.zeros(4)))
    assert _raises_value_error(lambda: lms.filter(np.zeros((5, 1)), np.zeros((5, 1))))
    print("✓ Invalid input test passed")


if __name__ == "__main__":
    print("Running LMS Filter Tests...")
    print()
    
    test_lms_filter_initialization()
    test_lms_filter_output_shapes()
    test_lms_filter_convergence()
    test_lms_filter_stable_across_step_sizes_and_scales()
    test_lms_filter_rejects_invalid_input()

    print()
    print("All tests passed! ✓")
