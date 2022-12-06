import numpy as np

parameters_files = [
    "example_0_20_parameters.npy",
    "example_1_20_parameters.npy",
    "example_2_20_parameters.npy",
    "example_3_20_parameters.npy",
    "example_4_20_parameters.npy",
]

parameters = np.vstack([np.load(param_file) for param_file in parameters_files])

