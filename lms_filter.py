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
        reference = np.asarray(reference, dtype=float)
        desired = np.asarray(desired, dtype=float)
        if reference.shape != desired.shape or reference.ndim != 1:
            raise ValueError("reference and desired must be 1-D arrays of equal length")

        n_samples = len(reference)
        error = np.zeros(n_samples)
        output = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.filter_length))

        # Zero-pad so the tap vector at time n is a slice instead of an np.roll copy
        padded = np.concatenate([np.zeros(self.filter_length - 1), reference])

        # Small constant to avoid division by zero
        epsilon = 0.001

        for n in range(n_samples):
            # Tap vector [x(n), x(n-1), ..., x(n-L+1)]
            reference_buffer = padded[n:n + self.filter_length][::-1]

            # Filter output (estimated noise)
            output[n] = np.dot(self.weights, reference_buffer)

            # Error signal (desired - estimated noise = cleaned signal)
            error[n] = desired[n] - output[n]

            # Update weights using normalized LMS algorithm
            # w(n+1) = w(n) + (mu / (||x(n)||^2 + epsilon)) * e(n) * x(n)
            # Normalizing by the instantaneous tap power (not a smoothed estimate)
            # is what guarantees stability for any 0 < mu < 2.
            normalization = np.dot(reference_buffer, reference_buffer) + epsilon
            self.weights += (self.step_size / normalization) * error[n] * reference_buffer

            # Store weights for analysis
            weights_history[n] = self.weights
            
        return error, output, weights_history
