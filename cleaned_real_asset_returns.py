import pandas as pd
df = pd.read_csv("Finance Portfolio 2.csv", skiprows=3)

df_real = df[
    [
        'Year',
        'S&P 500 (includes dividends)2',
        'US Small cap (bottom decile)22',
        '3-month T. Bill (Real)',
        '!0-year T.Bonds',
        'Baa Corp Bonds',
        'Real Estate3',
        'Gold'
    ]
]

df_real.columns = [
    'Year',
    'S&P 500',
    'Small Cap',
    'T-Bill',
    '10Y Treasury',
    'Baa Corporate',
    'Real Estate',
    'Gold'
]

for col in df_real.columns[1:]:
    df_real[col] = df_real[col].str.replace('%', '').astype(float)
df_real.to_csv("cleaned_real_asset_returns.csv", index=False)
print("Cleaned data saved as cleaned_real_asset_returns.csv")
