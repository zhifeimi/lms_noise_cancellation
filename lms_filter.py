"""
LMS (Least Mean Square) Adaptive Filter Implementation

This module implements the LMS algorithm for adaptive filtering, commonly used
in noise cancellation applications.
"""

import numpy as np


class LMSFilter:
    """
    Least Mean Square (LMS) Adaptive Filter with Normalization
    
    The LMS algorithm adjusts filter coefficients to minimize the mean square error
    between the desired signal and the filter output.
    
    Parameters:
    -----------
    filter_length : int
        Number of filter coefficients (filter order)
    step_size : float
        Learning rate (mu), controls convergence speed and stability
        For normalized LMS, typical range: 0 < mu < 2
    """
    
    def __init__(self, filter_length, step_size):
        """
        Initialize the LMS filter
        
        Args:
            filter_length: Number of filter taps
            step_size: Learning rate (mu)
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length)
        
    def filter(self, reference, desired):
        """
        Apply Normalized LMS adaptive filtering
        
        Args:
            reference: Reference signal (noise reference)
            desired: Desired signal (noisy signal to be cleaned)
            
        Returns:
            error: Error signal (cleaned signal)
            output: Filter output (estimated noise)
            weights_history: Evolution of filter weights
        """
        n_samples = len(reference)
        error = np.zeros(n_samples)
        output = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.filter_length))
        
        # Buffer for reference signal
        reference_buffer = np.zeros(self.filter_length)
        
        # Small constant to avoid division by zero
        epsilon = 0.001
        
        # Leaky integrator for power estimation
        alpha = 0.01  # Smoothing factor
        power_estimate = 1.0
        
        for n in range(n_samples):
            # Update reference buffer (shift and add new sample)
            reference_buffer = np.roll(reference_buffer, 1)
            reference_buffer[0] = reference[n]
            
            # Filter output (estimated noise)
            output[n] = np.dot(self.weights, reference_buffer)
            
            # Error signal (desired - estimated noise = cleaned signal)
            error[n] = desired[n] - output[n]
            
            # Estimate power with leaky integrator for smoother adaptation
            current_power = np.dot(reference_buffer, reference_buffer)
            power_estimate = alpha * current_power + (1 - alpha) * power_estimate
            
            # Update weights using normalized LMS algorithm
            # w(n+1) = w(n) + (mu / (power + epsilon)) * e(n) * x(n)
            normalization = power_estimate + epsilon
            self.weights = self.weights + (self.step_size / normalization) * error[n] * reference_buffer
            
            # Store weights for analysis
            weights_history[n] = self.weights.copy()
            
        return error, output, weights_history
