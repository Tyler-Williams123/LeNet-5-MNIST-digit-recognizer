Paper: Gradient-Based Learning Applied to Document Recognition — LeCun et al.

Implementation:
I implemented the original network architecture including selected connectivity at the second convolutional layer (labeled c3 in the paper).
  The c3 layer uses a connectivity mask to reproduce the partial connectivity represented in the paper.
Training:
- Optimizer: Adam
- Loss: Mean Squared Error
- Dataset MNIST

Results:
Testing accuracy: 98.65%
Training accuracy: 98.79%

Notes:
processed data and libraries are excluded to reduce size.

This is intended to be a reproduction of the architecture for understanding purposes. As such while the network remains mostly unchanged the training procedure was altered.
