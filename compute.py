import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import frenel
import force
from concurrent.futures import ProcessPoolExecutor, as_completed

# Retrieve interpolated values
eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')

# Define constants
STOP = 45
dist = 20
angle = 25 * np.pi / 180
phase = 0
a = 0.5

# Define ranges
R = np.linspace(20, 110, 100)
wls = np.linspace(400, 900, 100)

# Initialize result arrays
F_x = np.empty((len(wls), len(R), 8))
F_y = np.empty((len(wls), len(R), 8))
F_z = np.empty((len(wls), len(R), 8))

# Function to compute force for given indices
def compute_force(indices):
    i1, i2 = indices
    point = np.array([0, 0, dist + R[i1]])
    f = force.F(wls[i2], eps_Au, point, R[i1], eps_Si, angle, 1, phase, a, STOP, full_output=True)
    return i1, i2, f

# Main execution block
if __name__ == "__main__":
    # Create a list of all index pairs
    index_pairs = [(i1, i2) for i1 in range(len(R)) for i2 in range(len(wls))]

    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor() as executor:
        # Map the compute_force function over index_pairs with a progress bar
        futures = {executor.submit(compute_force, idx_pair): idx_pair for idx_pair in index_pairs}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Computing forces'):
            i1, i2, f = future.result()
            F_x[i2, i1, :] = f[0]
            F_y[i2, i1, :] = f[1]
            F_z[i2, i1, :] = f[2]

    # Save the results to .npy files
    np.save('F_x_phase.npy', F_x)
    np.save('F_y_phase.npy', F_y)
    np.save('F_z_phase.npy', F_z)
