import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 1. Define the true function we want to maximize
def true_function(x):
    return x * np.sin(x)

# 2. Create a dense grid of 1,000 points between 0 and 10.
# We use this ONLY to plot the true function and to let the AI scan for its next guess.
X_space = np.linspace(0, 10, 1000).reshape(-1, 1)
y_space = true_function(X_space)

# 1. Manually pick 2 initial points to start the optimization
# We pick x = 2.0 and x = 8.0 to give the model a wide view
X_sample = np.array([[2.0], [8.0]])

# 2. Calculate the true y-values for these 2 points using our function
# .flatten() just turns the output back into a simple 1D array for easy handling
y_sample = true_function(X_sample).flatten()

# 3. Create our Surrogate Model (The "Brain")
# We use the Matern kernel, which controls how smooth or bumpy the model expects the function to be.
gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, random_state=42)

# Run the optimization loop for 5 iterations
for i in range(5):
    # 1. Train the Gaussian Process with our current samples
    gp.fit(X_sample, y_sample)
    
    # 2. Predict the Mean (mu) and Std Dev (sigma) for all 1,000 points in our space
    mu, sigma = gp.predict(X_space, return_std=True)
    
    # 3. Calculate the Acquisition Score (UCB) for all 1,000 points
    ucb_score = upper_confidence_bound(X_space, gp, kappa=2.5)
    
    # 4. Find the best next point (where UCB score is the highest)
    next_index = np.argmax(ucb_score)
    next_X = X_space[next_index].reshape(-1, 1)
    next_y = true_function(next_X).flatten()
    
    # --- PLOTTING THE MEAN AND VARIANCE LIVE ---
    plt.figure(figsize=(10, 5))
    
    # Plot the True Function (The hidden reality)
    plt.plot(X_space, y_space, 'r--', label='True Function f(x)')
    
    # Plot the past data points we have sampled
    plt.plot(X_sample, y_sample, 'kx', markersize=10, label='Past Samples')
    
    # Plot the GP Mean Line (The prediction line)
    plt.plot(X_space, mu, 'b-', label='GP Mean ($\mu$)')
    
    # Plot the Variance/Uncertainty Bounds (The shaded region)
    # We shade the region between (Mean - 1.96*sigma) and (Mean + 1.96*sigma) 
    # This represents a 95% confidence interval
    plt.fill_between(X_space.flatten(), 
                     mu - 1.96 * sigma, 
                     mu + 1.96 * sigma, 
                     color='blue', alpha=0.15, label='Uncertainty Interval (95%)')
    
    # Highlight the next point the AI decided to sample
    plt.axvline(x=next_X[0][0], color='green', linestyle=':', label='Next Best Guess')
    
    plt.title(f"Bayesian Optimization - Iteration {i+1}")
    plt.legend(loc='upper left')
    plt.show()
    
    # 5. Append the new sample to our memory and repeat the loop!
    X_sample = np.vstack((X_sample, next_X))
    y_sample = np.append(y_sample, next_y)