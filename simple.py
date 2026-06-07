import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# 1. THE NEW BLACK BOX FUNCTION (Linear Peak at 0,0,0)
def black_box_function(x1, x2, x3):
    # Maximum value is 0 at (0,0,0) and decreases linearly away from it
    return -(np.abs(x1) + np.abs(x2) + np.abs(x3))

# 2. THE MATÉRN 5/2 KERNEL (Optimized for diagonal variance calculations)
def matern_5_2_kernel(X_A, X_B, l1, l2, l3, diag_only=False):
    M = X_A.shape[0]
    if diag_only:
        # Fast path for calculating variance along the slice grid
        d = np.sqrt(
            ((X_A[:, 0] - X_B[:, 0])**2 / l1**2) +
            ((X_A[:, 1] - X_B[:, 1])**2 / l2**2) +
            ((X_A[:, 2] - X_B[:, 2])**2 / l3**2)
        )
        sqrt_5_d = np.sqrt(5) * d
        return (1 + sqrt_5_d + (5.0 / 3.0) * (d**2)) * np.exp(-sqrt_5_d)
    
    N = X_B.shape[0]
    K = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            d = np.sqrt(
                ((X_A[i, 0] - X_B[j, 0])**2 / l1**2) +
                ((X_A[i, 1] - X_B[j, 1])**2 / l2**2) +
                ((X_A[i, 2] - X_B[j, 2])**2 / l3**2)
            )
            sqrt_5_d = np.sqrt(5) * d
            K[i, j] = (1 + sqrt_5_d + (5.0 / 3.0) * (d**2)) * np.exp(-sqrt_5_d)
    return K

# 3. SETUP & INITIALIZATION
np.random.seed(42)
bounds = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
l1, l2, l3 = 0.6, 0.6, 0.6
sigma_n_sq = 1e-5

# Start with 5 random samples
X_past = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, 3))
Y_past = np.array([black_box_function(x[0], x[1], x[2]) for x in X_past]).reshape(-1, 1)

# Generate slice grid layout (x3 = 0.0 to clearly show the peak at origin)
grid_x1 = np.linspace(-2.0, 2.0, 50)
grid_x2 = np.linspace(-2.0, 2.0, 50)
MX1, MX2 = np.meshgrid(grid_x1, grid_x2)
X_slice = np.vstack([MX1.ravel(), MX2.ravel(), np.zeros_like(MX1.ravel())]).T
Y_true_slice = np.array([black_box_function(x[0], x[1], x[2]) for x in X_slice]).reshape(MX1.shape)

# Lists to cache values computed during the calculations phase
history_plots_data = []

print("Phase 1: Computing calculations for the first 5 iterations safely in background...")

# 4. SILENT CALCULATION PHASE (Run 5 iterations first)
for iteration in range(1, 20):
    # GP Calculations
    K_xx = matern_5_2_kernel(X_past, X_past, l1, l2, l3)
    K_xx_inv = np.linalg.inv(K_xx + sigma_n_sq * np.eye(len(X_past)))
    alpha = K_xx_inv @ Y_past
    
    K_xstar_x = matern_5_2_kernel(X_slice, X_past, l1, l2, l3)
    
    # Speed Optimization: Only grab the diagonal elements for variance calculations
    K_xstar_xstar_diag = matern_5_2_kernel(X_slice, X_slice, l1, l2, l3, diag_only=True)
    
    mu_star = K_xstar_x @ alpha
    
    # Compute variance efficiently without calculating the full massive covariance matrix
    sigma_bar_sq = K_xstar_xstar_diag - np.sum((K_xstar_x @ K_xx_inv) * K_xstar_x, axis=1)
    sigma_star = np.sqrt(np.maximum(sigma_bar_sq.reshape(-1, 1), 1e-9))
    
    # Expected Improvement
    y_best = np.max(Y_past)
    Z = (mu_star - y_best) / sigma_star
    EI = (mu_star - y_best) * norm.cdf(Z) + sigma_star * norm.pdf(Z)
    
    best_idx = np.argmax(EI)
    next_X = X_slice[best_idx].reshape(1, 3)
    next_Y = np.array([[black_box_function(next_X[0,0], next_X[0,1], next_X[0,2])]])
    
    # Save a snapshot of data state for this specific frame iteration
    history_plots_data.append({
        'X_past': np.copy(X_past),
        'mu_star': np.copy(mu_star),
        'sigma_star': np.copy(sigma_star),
        'next_X': np.copy(next_X),
        'iter_num': iteration
    })
    
    # Feed data forward to expand history
    X_past = np.vstack((X_past, next_X))
    Y_past = np.vstack((Y_past, next_Y))
    print(f" -> Completed calculation iteration {iteration}/5")

print("\nPhase 2: Displaying automated graphics slideshow. Window loading...")

# 5. AUTOMATED TIMED DISPLAY PHASE (Changed to 1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Initialize empty colorbar variables to track initialization
cbar1, cbar2 = None, None

for frame in history_plots_data:
    # Clear old plots from window
    for ax in axes: ax.clear()
    
    X_p = frame['X_past']
    mu_s = frame['mu_star']
    sig_s = frame['sigma_star'] 
    nxt_X = frame['next_X']
    it = frame['iter_num']
    
    # Render Panel 1: Truth Target Space
    axes[0].set_title(f"True Linear Function (Slice x3=0)\nIteration {it}")
    c1 = axes[0].contourf(MX1, MX2, Y_true_slice, levels=50, cmap='viridis')
    axes[0].scatter(X_p[:-1, 0], X_p[:-1, 1], color='white', edgecolors='black', s=60, label='Past Tests')
    axes[0].scatter(X_p[-1, 0], X_p[-1, 1], color='red', marker='*', s=150, label='Latest Test')
    axes[0].legend()
    
    # Render Panel 2: GP Cloud Space Model
    axes[1].set_title("GP Predicted Mean Map ($\mu_*$)")
    c2 = axes[1].contourf(MX1, MX2, mu_s.reshape(MX1.shape), levels=50, cmap='viridis')
    axes[1].contour(MX1, MX2, sig_s.reshape(MX1.shape), levels=[0.4, 0.7], colors='cyan', linestyles='dashed')
    axes[1].scatter(X_p[:, 0], X_p[:, 1], color='white', edgecolors='black', s=40)
    # Overlay the next chosen target onto the predictive map for context
    axes[1].scatter(nxt_X[0, 0], nxt_X[0, 1], color='lime', marker='x', s=120, lw=3, label='Next Target')
    axes[1].legend()
    
    # Draw side color intensity indicators once
    if cbar1 is None:
        cbar1 = fig.colorbar(c1, ax=axes[0])
        cbar2 = fig.colorbar(c2, ax=axes[1])
        
    plt.tight_layout()
    
    # Force drawing systems to flush images immediately to monitor grid
    plt.draw()
    plt.pause(0.01) 
    
    print(f"Showing Iteration {it} on screen for 5 seconds...")
    time.sleep(8.0) # Standard raw system wait delay duration

print("\nSlideshow finished successfully.")
plt.show(block=True)


jbqfjbwfw.fvakjfbwr.kjfb  .                                                                         