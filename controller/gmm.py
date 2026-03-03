import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def calculate_gaussian_cvar(mu, sigma, alpha=0.05):
    """
    Calculates the Conditional Value at Risk (Expected Shortfall)
    for the left tail (losses) of a Normal Distribution.
    """
    # Inverse CDF (quantile) for the alpha level
    q = norm.ppf(alpha)
    # PDF at that quantile
    pdf_val = norm.pdf(q)
    
    # Analytical Formula: mu - sigma * (pdf / alpha)
    # Note: This gives the expected value of the tail. 
    # If you want "positive loss", flip the sign. Here we keep the raw value (e.g., -8%).
    cvar = mu - sigma * (pdf_val / alpha)
    return cvar

# 1. Generate Synthetic Data (Returns)
# Cluster 1: Stable regime (Mean=2%, Low Volatility)
# Cluster 2: Crisis regime (Mean=-5%, High Volatility)
np.random.seed(1)
stable_data = np.random.normal(loc=2, scale=1.5, size=700)
crisis_data = np.random.normal(loc=-5, scale=4.0, size=300)
crisis_data2 = np.random.normal(loc=0, scale=4.0, size=500)
X = np.concatenate([stable_data, crisis_data, crisis_data2]).reshape(-1, 1)

# 2. Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)


# 3. Analyze Each Component (Modal)
print(f"{'Cluster':<10} | {'Weight':<10} | {'Mean':<10} | {'Volatility':<10} | {'CVaR (5%)':<10}")
print("-" * 65)

x_axis = np.linspace(-20, 10, 1000)
plt.figure(figsize=(10, 6))

colors = ['blue', 'red', 'green']

for i in range(gmm.n_components):
    # Extract parameters for this component
    weight = gmm.weights_[i]
    mu = gmm.means_[i, 0]
    sigma = np.sqrt(gmm.covariances_[i, 0, 0])
    
    # Calculate Risk
    cvar = calculate_gaussian_cvar(mu, sigma, alpha=0.05)
    
    # Print Stats
    print(f"{i:<10} | {weight:.2f}       | {mu:.2f}       | {sigma:.2f}       | {cvar:.2f}")

    # --- Visualization ---
    # Draw the Gaussian Curve for this component
    y_axis = weight * norm.pdf(x_axis, mu, sigma)
    plt.plot(x_axis, y_axis, label=f'Cluster {i} (Mean: {mu:.1f}, CVaR: {cvar:.1f})', color=colors[i], linewidth=2)
    
    # Shade the CVaR area (Worst 5% of this specific cluster)
    cutoff = norm.ppf(0.05, mu, sigma)
    x_tail = np.linspace(x_axis.min(), cutoff, 100)
    y_tail = weight * norm.pdf(x_tail, mu, sigma)
    plt.fill_between(x_tail, 0, y_tail, color=colors[i], alpha=0.3)
    plt.axvline(cvar, color=colors[i], linestyle='--', alpha=0.7, label=f'Cluster {i} CVaR Line')

# Plot the actual data histogram for comparison
plt.hist(X, bins=50, density=True, color='gray', alpha=0.2, label='Actual Data')

plt.title('Risk Analysis of Each GMM Modal (Cluster)')
plt.xlabel('Returns / Value')
plt.ylabel('Density (Probability)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()