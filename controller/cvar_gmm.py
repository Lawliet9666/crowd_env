import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar



def cvar_method_1_variational(weights, means, stds, alpha=0.05):
    # 1. 转换为损失分布 (Loss Distribution)
    # 收益 mu=5% -> 损失 mu=-5%
    # 收益 mu=-20% -> 损失 mu=+20%
    loss_means = -1 * np.array(means) 
    
    def objective_function(nu):
        # nu 代表 VaR (损失阈值，正数)
        # 我们要最小化: nu + (1/alpha) * sum( w * E[(L - nu)+] )
        
        excess_loss_sum = 0
        for i in range(len(weights)):
            mu_L = loss_means[i]
            sig = stds[i]
            w = weights[i]
            
            # 计算 '超额损失期望' E[(L - nu)+]
            # 即损失分布中，超过阈值 nu 的部分的期望
            z = (nu - mu_L) / sig
            
            # 积分公式: (mu - nu) * (1 - Phi(z)) + sigma * phi(z)
            # 这里的 (1 - norm.cdf(z)) 代表右尾概率
            term = (mu_L - nu) * (1 - norm.cdf(z)) + sig * norm.pdf(z)
            
            excess_loss_sum += w * term
            
        return nu + (1 / alpha) * excess_loss_sum

    # 2. 设定搜索范围
    # 损失的均值大约在 -5 到 +20 之间。VaR 应该在 10 左右。
    # 给一个安全的范围：[最小均值, 最大均值 + 6倍标准差]
    bound_min = np.min(loss_means)
    bound_max = np.max(loss_means) + 6 * np.max(stds)
    
    # 3. 求解最小化问题
    res = minimize_scalar(objective_function, bounds=(bound_min, bound_max), method='bounded')
    
    # res.fun 是最小化后的 Loss CVaR (比如 +10.02)
    # 我们需要返回收益 CVaR (即 -10.02)
    return -1 * res.fun

def cvar_method_2_exact(weights, means, stds, alpha=0.05):
    # GMM 半解析法 (Exact Semi-Analytical)
    def gmm_cdf(x):
        total_cdf = 0
        for i in range(len(weights)):
            total_cdf += weights[i] * norm.cdf((x - means[i]) / stds[i])
        return total_cdf - alpha

    try:
        bound_min = np.min(means) - 10 * np.max(stds)
        bound_max = np.max(means) + 10 * np.max(stds)
        var_threshold = brentq(gmm_cdf, bound_min, bound_max)
    except ValueError:
        return np.nan

    cvar = 0
    for i in range(len(weights)):
        mu, sigma, w = means[i], stds[i], weights[i]
        z = (var_threshold - mu) / sigma
        term = mu * norm.cdf(z) - sigma * norm.pdf(z)
        cvar += w * term
    return cvar / alpha

def cvar_method_3_sampling(weights, means, stds, alpha=0.05, n_samples=1000000):
    # 蒙特卡洛采样法
    np.random.seed(42)
    samples = []
    for i in range(len(weights)):
        n_count = int(n_samples * weights[i])
        samples.append(np.random.normal(means[i], stds[i], n_count))
    all_data = np.concatenate(samples)
    all_data.sort()
    cutoff_index = int(n_samples * alpha)
    return all_data[:cutoff_index].mean()

def cvar_method_3_optimization(weights, means, stds, alpha=0.05, n_samples=1000000):
    np.random.seed(42)
    samples = []
    for i in range(len(weights)):
        n_count = int(n_samples * weights[i])
        samples.append(np.random.normal(means[i], stds[i], n_count))
    all_data = np.concatenate(samples)
    def objective_function(nu):
        # 离散版本的 Rockafellar 公式
        # E[...] 变成了 mean()
        # [L - nu]+ 变成了 np.maximum(samples - nu, 0)
        excess_loss = np.maximum(all_data - nu, 0)
        return nu + (1 / alpha) * np.mean(excess_loss)

    # 寻找最优的 nu
    res = minimize_scalar(objective_function)
    return res.fun

# 假设 samples 是你生成的 loss 数据
# cvar_sort = 也就是之前的 method 3
# cvar_opt  = cvar_method_3_optimization(samples)
# print(cvar_sort == cvar_opt)  -> 结果会极其接近

def cvar_method_4_worst_component(weights, means, stds, alpha=0.05):
    # 最差单体法 (Worst-Case Robust) - 忽略权重
    worst_cvar = float('inf')
    for i in range(len(weights)):
        mu, sigma = means[i], stds[i]
        q = norm.ppf(alpha)
        cvar_single = mu - sigma * (norm.pdf(q) / alpha)
        if cvar_single < worst_cvar:
            worst_cvar = cvar_single
    return worst_cvar

# ==========================================
np.random.seed(1)
stable_data = np.random.normal(loc=2, scale=1.5, size=700)
crisis_data = np.random.normal(loc=-5, scale=4.0, size=300)
crisis_data2 = np.random.normal(loc=0, scale=4.0, size=500)
X = np.concatenate([stable_data, crisis_data, crisis_data2]).reshape(-1, 1)

# 训练模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# ==========================================
# 3. 关键步骤：从 GMM 对象提取并转换参数
# ==========================================

# A. 提取权重 (直接就是 1D array)
w_learned = gmm.weights_
# B. 提取均值 (形状是 (3, 1)，需要展平成 1D)
m_learned = gmm.means_.flatten()
# C. 提取协方差并转为标准差
# gmm.covariances_ 形状是 (3, 1, 1)，因为是 'full' 且只有 1 个特征
# 我们需要取出数值，开根号，得到 std
covs = gmm.covariances_.flatten() # 变成 [var1, var2, var3]
s_learned = np.sqrt(covs)         # 变成 [std1, std2, std3]

# 打印学习到的参数看看
print("\n--- GMM 学习到的参数 ---")
for i in range(3):
    print(f"Cluster {i}: 权重={w_learned[i]:.2%}, 均值={m_learned[i]:.2f}, 标准差={s_learned[i]:.2f}")

# ==========================================
# 4. 代入函数计算对比
# ==========================================
alpha = 0.05
print(f"\n计算 {alpha*100}% CVaR (基于学习到的参数):")
print("-" * 60)

# 1. 变分法
res1 = cvar_method_1_variational(w_learned, m_learned, s_learned, alpha)
print(f"1) 变分优化法 (Weighted Constraint):  {res1:.6f}")

# 2. 精确半解析
res2 = cvar_method_2_exact(w_learned, m_learned, s_learned, alpha)
print(f"2) GMM 直接求解 (Exact Semi-Analytical): {res2:.6f}")

# 3. Sampling
res3 = cvar_method_3_sampling(w_learned, m_learned, s_learned, alpha)
print(f"3) Sampling 采样法 (Monte Carlo):     {res3:.6f}")

# 3. Sampling
res3 = cvar_method_3_optimization(w_learned, m_learned, s_learned, alpha)
print(f"3) 优化法 (Optimization):     {res3:.6f}")


# 4. 最差单体
res4 = cvar_method_4_worst_component(w_learned, m_learned, s_learned, alpha)
print(f"4) 不带权最差单体 (Worst-Case Robust):   {res4:.6f}")

print("-" * 60)
print("结论分析:")
print(f"方法 1 和 2 的误差: {abs(res1 - res2):.8f}")
print(f"真实 GMM 风险 (方法1/2) vs 极端保守风险 (方法4): {abs(res2 - res4):.4f}")