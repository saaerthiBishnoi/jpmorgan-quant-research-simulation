import pandas as pd
import numpy as np

df = pd.read_csv('Task_3_and_4_Loan_Data.csv')
df = df.sort_values('fico_score').reset_index(drop=True)

scores = df['fico_score'].values
defaults = df['default'].values
unique_scores = np.sort(df['fico_score'].unique())
n_unique = len(unique_scores)

cum_n = np.zeros(n_unique + 1)
cum_k = np.zeros(n_unique + 1)
for i, s in enumerate(unique_scores):
    mask = scores == s
    cum_n[i + 1] = cum_n[i] + mask.sum()
    cum_k[i + 1] = cum_k[i] + defaults[mask].sum()

def bucket_loglik(i, j):
    n = cum_n[j] - cum_n[i]
    k = cum_k[j] - cum_k[i]
    if n == 0:
        return 0
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)

def optimal_fico_buckets(num_buckets):
    NEG_INF = -1e18
    DP = np.full((num_buckets + 1, n_unique + 1), NEG_INF)
    DP[0][0] = 0
    split = np.full((num_buckets + 1, n_unique + 1), -1, dtype=int)

    for b in range(1, num_buckets + 1):
        for j in range(b, n_unique + 1):
            best_val = NEG_INF
            best_i = -1
            for i in range(b - 1, j):
                if DP[b - 1][i] == NEG_INF:
                    continue
                val = DP[b - 1][i] + bucket_loglik(i, j)
                if val > best_val:
                    best_val = val
                    best_i = i
            DP[b][j] = best_val
            split[b][j] = best_i

    boundaries_idx = [n_unique]
    j = n_unique
    for b in range(num_buckets, 0, -1):
        i = split[b][j]
        boundaries_idx.append(i)
        j = i
    boundaries_idx = sorted(set(boundaries_idx))

    boundaries = [unique_scores[idx] if idx < n_unique else unique_scores[-1] + 1
                  for idx in boundaries_idx]
    return boundaries, DP[num_buckets][n_unique]

NUM_BUCKETS = 5
fico_boundaries, total_loglik = optimal_fico_buckets(NUM_BUCKETS)

print(f"Optimal {NUM_BUCKETS}-bucket FICO boundaries:", fico_boundaries)
print("Total log-likelihood:", round(total_loglik, 2))

df['bucket'] = pd.cut(df['fico_score'], bins=fico_boundaries, right=False, include_lowest=True)
summary = df.groupby('bucket', observed=True).agg(n=('default', 'size'), defaults=('default', 'sum'))
summary['default_rate'] = summary['defaults'] / summary['n']
print("\nDefault rate by bucket (should decrease as FICO increases):")
print(summary)

def fico_to_rating(fico_score, boundaries=fico_boundaries):
    """
    Maps a FICO score to a credit rating bucket.
    Rating 1 = best credit score (lowest default risk)
    Rating N = worst credit score (highest default risk)
    """
    num_buckets = len(boundaries) - 1
    for idx in range(num_buckets):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if lower <= fico_score < upper:
            # highest bucket index = worst score, so reverse it for the rating
            # bucket 0 = lowest scores = worst credit = highest rating number
            rating = num_buckets - idx
            return rating
    # score falls at or above the top boundary
    return 1

if __name__ == "__main__":
    print("\nSample FICO to rating mappings:")
    for test_score in [420, 550, 610, 670, 720, 800]:
        print(f"  FICO {test_score} -> Rating {fico_to_rating(test_score)}")
