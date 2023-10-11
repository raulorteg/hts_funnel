# Basis set and grid field used in preprocessing.
BASIS_SET = "6-31G"
RADIUS = 0.75
GRID_INTERVAL = 0.3

# Setting of a neural network architecture.
DIM = 500
LAYER_FUNCTIONAL = 6
HIDDEN_HK = 500
LAYER_HK = 6

# Operation for final layer.
# OPERATION="sum"  # For energy (i.e., a property proportional to the molecular size).
OPERATION = "mean"  # For homo and lumo (i.e., a property unrelated to the molecular size or the unit is e.g., eV/atom).

# Setting of optimization.
BATCH_SIZE = 8
LR = 1e-4
LR_DECAY = 0.5
STEP_SIZE = 200
ITERATION = 2000
