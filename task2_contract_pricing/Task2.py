import pandas as pd
import numpy as np

df = pd.read_csv('Nat_Gas_2.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)
START_DATE = df['Dates'].min()
df['DayNum'] = (df['Dates'] - START_DATE).dt.days
m, c = np.polyfit(df['DayNum'], df['Prices'], 1)
df['Trend'] = m * df['DayNum'] + c
df['Residual'] = df['Prices'] - df['Trend']
df['Month'] = df['Dates'].dt.month
seasonal_adjustment = df.groupby('Month')['Residual'].mean()

def price_estimate(date_input):
    date = pd.to_datetime(date_input)
    day_num = (date - START_DATE).days
    trend_price = m * day_num + c
    month_adj = seasonal_adjustment.loc[date.month]
    return round(trend_price + month_adj, 2)

def price_contract(injection_dates, withdrawal_dates, volume_per_event,
                    max_storage_volume, storage_cost_per_month):
    total_value = 0
    current_storage = 0
    all_events = []

    for d in injection_dates:
        all_events.append((pd.to_datetime(d), 'inject'))
    for d in withdrawal_dates:
        all_events.append((pd.to_datetime(d), 'withdraw'))
    all_events.sort()

    for date, action in all_events:
        price = price_estimate(date)
        if action == 'inject':
            current_storage += volume_per_event
            if current_storage > max_storage_volume:
                raise ValueError(f"Storage exceeded on {date.date()}: "
                                  f"{current_storage} > {max_storage_volume}")
            total_value -= price * volume_per_event
        else:
            if volume_per_event > current_storage:
                raise ValueError(f"Not enough gas in storage to withdraw on {date.date()}")
            current_storage -= volume_per_event
            total_value += price * volume_per_event

    first_date = min(all_events)[0]
    last_date = max(all_events)[0]
    months_stored = (last_date - first_date).days / 30.44
    storage_fee = months_stored * storage_cost_per_month
    total_value -= storage_fee

    return round(total_value, 2)

result1 = price_contract(
    injection_dates=['2023-06-30'],
    withdrawal_dates=['2023-12-31'],
    volume_per_event=1_000_000,
    max_storage_volume=1_000_000,
    storage_cost_per_month=100_000
)
print("Test 1 (single inject/withdraw, summer->winter):", result1)

result2 = price_contract(
    injection_dates=['2023-06-30', '2023-07-31'],
    withdrawal_dates=['2023-12-31', '2024-01-31'],
    volume_per_event=500_000,
    max_storage_volume=1_000_000,
    storage_cost_per_month=50_000
)
print("Test 2 (multiple inject/withdraw dates):", result2)

result3 = price_contract(
    injection_dates=['2024-05-31'],
    withdrawal_dates=['2024-06-30'],
    volume_per_event=1_000_000,
    max_storage_volume=1_000_000,
    storage_cost_per_month=100_000
)
print("Test 3 (short storage, minimal seasonal gain):", result3)
