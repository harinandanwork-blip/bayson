import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# =====================================================================
# BLOCK 1: Define the Visible Target Function & Search Grid
# =====================================================================
def true_function(x):
    return x * np.sin(x)

# 1,000 dense points to evaluate the space and draw the true function
X_space = np.linspace(0, 10, 1000).reshape(-1, 1)
y_space = true_function(X_space)

# =====================================================================
# BLOCK 2: Define Strategy (Upper Confidence Bound Acquisition)
# =====================================================================
def upper_confidence_bound(X, gp, kappa=2.5):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)  # Avoid mathematical errors
    return mu + kappa * sigma

# =====================================================================
# BLOCK 3: Set up Initial History and Initialize the GP "Brain"
# =====================================================================
# Start with two manual guesses to kickstart the matrix math
X_sample = np.array([[2], [5]])
y_sample = true_function(X_sample).flatten()

# Initialize the Gaussian Process Regressor with a Matérn 2.5 kernel
gp = GaussianProcessRegressor(kernel=Matern(nu=10), alpha=1e-6, random_state=42)

# =====================================================================
# BLOCK 4: The Interactive Optimization Loop
# =====================================================================
total_iterations = 12

for i in range(total_iterations):
    # 1. Update the Surrogate Model with our current data history
    gp.fit(X_sample, y_sample)
    
    # 2. Extract current predictions across the 1,000 grid points
    mu, sigma = gp.predict(X_space, return_std=True)
    
    # 3. Compute Acquisition Scores and pinpoint the highest scoring index
    ucb_score = upper_confidence_bound(X_space, gp, kappa=2.5)
    next_index = np.argmax(ucb_score)
    
    # 4. Extract and calculate the winner's actual performance
    next_X = X_space[next_index].reshape(-1, 1)
    next_y = true_function(next_X).flatten()
    
    # 5. Render the Dual Live Plots (Surrogate Model + Acquisition Function)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Top Subplot: Surrogate Model ---
    ax1.plot(X_space, y_space, 'r--', label='True Function f(x)')
    ax1.plot(X_sample, y_sample, 'kx', markersize=10, label='Past Samples')
    ax1.plot(X_space, mu, 'b-', label='GP Mean ($\mu$)')
    ax1.fill_between(X_space.flatten(), 
                     mu - 1.96 * sigma, 
                     mu + 1.96 * sigma, 
                     color='blue', alpha=0.15, label='95% Uncertainty Cloud')
    ax1.axvline(x=next_X[0][0], color='green', linestyle=':', label='Next Best Guess')
    ax1.set_title(f"Bayesian Optimization Loop — Iteration {i+1} of {total_iterations}")
    ax1.set_ylabel("Objective Value")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Bottom Subplot: Acquisition Function ---
    ax2.plot(X_space, ucb_score, 'g-', label='UCB Acquisition Curve')
    ax2.axvline(x=next_X[0][0], color='green', linestyle=':', label='Max Acquisition Point')
    ax2.scatter(next_X, ucb_score[next_index], color='green', s=100, zorder=5, label='Selected Point')
    ax2.set_xlabel("X Space")
    ax2.set_ylabel("UCB Score")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()  # Opens up the window showing the current dual-plot snapshot
    
    # =====================================================================
    # THE FREEZE SWITCH: Wait for User Input to advance
    # =====================================================================
    # If the plot window blocks execution, close the plot window first,
    # then press enter in your terminal console to fire the next iteration.
    input(f"--> [Iteration {i+1} Rendered] Press Enter in terminal to evaluate the Green Line and run Iteration {i+2}...")
    print("-" * 80)
    
    # 6. Commit the evaluated coordinates into memory logs and repeat
    X_sample = np.vstack((X_sample, next_X))
    y_sample = np.append(y_sample, next_y)

print("\n Optimization Loop complete! You have successfully stepped through Bayesian Optimization manual execution.")