import pandas as pd
import numpy as np
import holidays
from datetime import date

def avo_holiday_assign(df, date_var, us_holidays):
    # Assign holidays
    df["holiday"] = 0
    unique_weeks = pd.unique(df[date_var])
    timestamp_interval = unique_weeks[1] - unique_weeks[0]
    holiday_present_offset = 14

    for ii in range(df.shape[0]):
        timestamp = df[date_var].iloc[ii]
        for jj in np.linspace(-1, 6, 8):
            if jj == -1:
                new_timestamp =  timestamp + timestamp_interval
            else:
                new_timestamp = timestamp + timestamp_interval / (14 / (jj + 1.01)) + timestamp_interval
            day = new_timestamp.day
            month = new_timestamp.month
            year = new_timestamp.year
            holiday_check = date(year, month, day) in us_holidays
            if (month == 5) & (day == 5):
                holiday_check = True
            if holiday_check:
                df["holiday"].iloc[ii] = 1
    return df

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100