import numpy as np



# Function to generate synthetic data
def generate_synthetic_data(num_sequence_types=3, num_samples=100, sequence_length=7, num_output_types=4, input_noise=False):
    X_blocked = []
    y_blocked = []
    for _ in range(num_sequence_types):
        sequence = np.random.randint(num_output_types, size=(1, 1, sequence_length))
        sequence_out = np.zeros((1, sequence_length, num_output_types))
        for i in range(sequence_length):
            sequence_out[0, i, sequence[0, 0, i]] = 1
        sequence = np.repeat(sequence, num_samples, axis=0)
        sequence_out = np.repeat(sequence_out, num_samples, axis=0)
        X_blocked.append(sequence)
        y_blocked.append(sequence_out)
    X_blocked = np.concatenate(X_blocked, axis=0) + 1 # Add 1 to avoid 0s
    y_blocked = np.concatenate(y_blocked, axis=0)

    if input_noise:
        # change int32 to float64
        X_blocked = X_blocked.astype(np.float64)
        X_blocked += np.random.normal(0, 0.1, X_blocked.shape)
        # y_blocked = y_blocked.astype(np.float64)
        # y_blocked += np.random.normal(0, 0.1, y_blocked.shape)

    X_random = X_blocked.copy()
    y_random = y_blocked.copy()
    # permute the first axis
    perm = np.random.permutation(X_random.shape[0])
    X_random = X_random[perm]
    y_random = y_random[perm]
    return X_blocked, y_blocked, X_random, y_random
