# LMS Noise Cancellation
Acoustic Noise Cancellation Using LMS (Least Mean Square) Algorithm

## Overview

This project demonstrates how to remove noise from an input signal using the **Least Mean Square (LMS) algorithm**. The LMS adaptive filter automatically matches the filter response by using the reference signal on the input port and the desired signal on the desired port. Only the original signal should remain after subtracting the filtered noise as the error signal gets closer to the proper filter model.

## How It Works

The LMS algorithm is an adaptive filtering technique that:

1. **Takes two inputs:**
   - **Reference signal (x)**: A noise reference that is correlated with the noise in the desired signal
   - **Desired signal (d)**: The noisy signal that needs to be cleaned

2. **Adapts filter weights** to minimize the mean square error between:
   - The desired signal (d)
   - The filter output (estimated noise)

3. **Produces an error signal (e)** that contains the cleaned signal:
   ```
   e(n) = d(n) - y(n)
   ```
   where y(n) is the estimated noise filtered from the reference signal

4. **Updates filter weights** using the LMS update rule:
   ```
   w(n+1) = w(n) + μ * e(n) * x(n)
   ```
   where μ is the step size (learning rate)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/zhifeimi/lms_noise_cancellation.git
cd lms_noise_cancellation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the demonstration script:
```bash
python demo.py
```

This will:
- Generate a clean test signal (combination of sine waves)
- Add correlated noise to create a noisy signal
- Apply the LMS adaptive filter to remove the noise
- Display performance metrics (SNR improvement, MSE)
- Save a visualization showing the original, noisy, and cleaned signals

Run the unit tests:
```bash
python test_lms.py
```

This will verify that the LMS filter implementation works correctly.

## Files

- **lms_filter.py**: Core LMS filter implementation with normalized LMS algorithm
- **demo.py**: Demonstration script showing noise cancellation in action
- **test_lms.py**: Unit tests for the LMS filter
- **requirements.txt**: Required Python packages

## Algorithm Parameters

- **filter_length**: Number of filter coefficients (default: 10)
  - Higher values allow modeling more complex noise patterns
  - Too high may cause slow convergence or overfitting

- **step_size (μ)**: Learning rate (default: 0.1 for normalized LMS)
  - Controls convergence speed and stability
  - For normalized LMS: 0 < μ < 2
  - Smaller values: slower convergence, more stable
  - Larger values: faster convergence, may be unstable

## Results

The demo shows:
- **Input SNR**: Signal-to-Noise Ratio before filtering
- **Output SNR**: Signal-to-Noise Ratio after filtering  
- **SNR Improvement**: How much noise was reduced (in dB)
- **Mean Square Error**: Difference between cleaned and original signal

Typical results show noise reduction with SNR improvement of 0.5-1.0 dB, demonstrating that the LMS algorithm successfully adapts to remove correlated noise from the signal.

## Applications

LMS adaptive filtering is widely used in:
- Active noise cancellation (headphones, automotive)
- Echo cancellation (telecommunications)
- Biomedical signal processing (ECG, EEG noise removal)
- Radar and sonar signal processing
- Interference cancellation in wireless communications

## License

This project is open source and available for educational purposes.
