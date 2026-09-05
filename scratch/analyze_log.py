import pandas as pd

df = pd.read_csv(r'c:\Users\Hp\Documents\Projects\Federated_learning\multi-agent\bdsf-run-analysis\logs\phase4_results\benchmarks\Proposed_BDSF_AFL_S2_MIMICRY_a0.1_f20_s42_updates.csv')
print("Total records:", len(df))

print("\nStatus counts:")
print(df['status'].value_counts())

zero_grad = df[df['reason'] == 'HARD_GUARD_ZERO_GRADIENT']
print("\nZero gradient rejections:", len(zero_grad))
print("Clients with zero gradient:", zero_grad['client_id'].unique())

print("\nReasons breakdown:")
print(df['reason'].value_counts())

print("\nRejections by client:")
rejections = df[df['status'] == 'REJECT']
print(rejections['client_id'].value_counts())

print("\nDownweights by client:")
downweights = df[df['status'] == 'DOWNWEIGHT']
print(downweights['client_id'].value_counts())

# Group by round to see if all rounds are present
print("\nRounds:", df['round'].min(), "-", df['round'].max())
