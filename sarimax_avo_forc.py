import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import matplotlib
import holidays
import collections
import warnings
import statsmodels.api as sm
import os

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import date, datetime, timedelta
from holiday_assign import *

def time_series_func(df, df_products, df_data, plot_save_flag, fig_path, pred_window,
                     offseason_week_start, offseason_week_end, forc_year, fcst_type, m):
    # Initialize values
    unique_items = pd.unique(df["Item Number"])
    num_items = len(unique_items)
    date_var = "Fiscal Week Start Date"
    us_holidays = holidays.UnitedStates()
    curr_date = datetime.today().strftime("%Y_%m_%d")

    actual = "CasesSum"
    predicted = "predicted_mean"
    forc = "sum_forc"

    keep_cols = ["Fiscal Week Start Date"]

    pred_end = df["Fiscal Week Start Date"].max()
    pred_start = pred_end - timedelta(weeks=pred_window - 1)

    off_start = pred_end + timedelta(weeks=1)
    off_end = off_start + timedelta(weeks=104)

    if off_start.week > offseason_week_start and off_start.year == forc_year:
        off_start = off_start - timedelta(weeks=off_start.week - offseason_week_start)

    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    # Setting plot characteristics
    plt.ioff()

    # Sort dataframe by date
    df = df.sort_values(by=date_var)
    df["year"] = df[date_var].dt.year
    df["month"] = df[date_var].dt.month
    df["week"] = df[date_var].dt.isocalendar().week

    unique_weeks = pd.unique(df[date_var])

    # Assign holiday values to the sales data dates
    df = avo_holiday_assign(df, date_var, us_holidays)

    #####Do the TS Thing!###################################################################################################
    df_unique_items = pd.DataFrame(unique_items, columns=["Item Number"])
    item_family_df = df_products[["Item Number", "Family"]]
    df_unique_items = pd.merge(df_unique_items, item_family_df, on="Item Number", how="left")
    df_unique_items.rename(columns={'Item Number': "Item Name"}, inplace=True)

    naive_rmse = np.array([])
    full_predictions = pd.DataFrame()
    off_season_predictions = pd.DataFrame()

    for ii in range(num_items):
        try:
            item_name = unique_items[ii]
            df_item_temp = df.loc[(df["Item Number"] == item_name) & (df[date_var] < pred_start)]
            cases_sum = df_item_temp["Cases"].groupby(df_item_temp[date_var].dt.to_period("W-MON")).sum()
            df_forc = df_item_temp.loc[df_item_temp["Item Number"] == item_name]
            df_forc = df_forc[keep_cols].drop_duplicates()
            df_forc["CasesSum"] = cases_sum.to_numpy()
            df_forc = df_forc.set_index(date_var).asfreq('W-MON')
            df_forc = df_forc.fillna(0)

            # Get dp predictions
            data_sub = df_data.loc[df_data["Item"] == item_name]
            data_sub["date"] = data_sub["WkDt"]
            sum_forc = data_sub["total_forc"].groupby(data_sub["date"].dt.to_period("W-MON")).sum()
            sum_forc = sum_forc.to_numpy()
            data_sub = pd.DataFrame(data_sub["date"].drop_duplicates())
            data_sub["sum_forc"] = sum_forc

            # Generate ARIMA params
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], m) for x in list(itertools.product(p, d, q))]

            # Run through the grid search
            best_aic = -1
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(df_forc,
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_invertibility=False)
                        results = mod.fit(disp=0)
                        if best_aic == -1:
                            best_aic = results.aic
                            best_param = param
                            best_seasonal = param_seasonal
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_param = param
                            best_seasonal = param_seasonal
                    except:
                        continue

            mod = sm.tsa.statespace.SARIMAX(df_forc,
                                            order=best_param,
                                            seasonal_order=best_seasonal,
                                            enforce_invertibility=False)
            results = mod.fit(disp=0)

            pred = results.get_prediction(start=pred_start + timedelta(weeks=1), end=pred_end)
            df_pred = pd.DataFrame(pred.predicted_mean)
            df_pred = df_pred.reset_index()
            df_pred.rename(columns={'index': date_var}, inplace=True)

            # Get off season predictions
            off_pred = results.get_prediction(start=off_start, end=off_end, freq='W-MON')
            df_off_pred = pd.DataFrame(off_pred.predicted_mean)
            df_off_pred = df_off_pred.reset_index()
            df_off_pred["week"] = df_off_pred["index"].dt.isocalendar().week
            df_off_pred["year"] = df_off_pred["index"].dt.year

            df_off_pred = df_off_pred.loc[df_off_pred["year"] == forc_year].drop(columns="year")

            off_season_start_index = df_off_pred.loc[df_off_pred["week"] == offseason_week_start].index[0]
            off_season_end_index = df_off_pred.loc[df_off_pred["week"] == offseason_week_end].index[0]
            if off_season_start_index > off_season_end_index:
                off_season_start_index = off_season_forecast.loc[off_season_forecast["week"]
                                                                 == offseason_week_end].index[1]

            df_off_pred = df_off_pred.loc[off_season_start_index
                                           :off_season_end_index + 1].drop(columns="week")
            df_off_pred.rename(columns={'index': "date"}, inplace=True)
            df_off_pred["Item Name"] = item_name
            off_season_predictions = off_season_predictions.append(df_off_pred, ignore_index=True)
            off_season_predictions["predicted_mean"].loc[off_season_predictions["predicted_mean"] < 0] = 0

            # Create the forecast df
            forecast = df.loc[(df["Item Number"] == item_name) & (df[date_var] >= df_pred[date_var].min()) &
                              (df[date_var] <= df_pred[date_var].max())]
            cases_sum = forecast["Cases"].groupby(forecast[date_var].dt.to_period("W-MON")).sum()
            forecast = forecast[keep_cols].drop_duplicates()
            forecast["CasesSum"] = cases_sum.to_numpy()
            item_sub = pd.merge(forecast, df_pred, on=date_var, how="left")

            data_sub = data_sub.loc[(data_sub["date"] >= df_pred[date_var].min()) &
                                    (data_sub["date"] <= df_pred[date_var].max())]
            data_sub.rename(columns={"date": date_var}, inplace=True)
            item_sub = pd.merge(item_sub, data_sub, on=date_var, how="left")
            item_sub["Item Name"] = item_name

            full_predictions = full_predictions.append(item_sub, ignore_index=True)

            # Visualize
            amend_rmse = mean_squared_error(item_sub[actual], item_sub[predicted], squared=False)
            amend_pct_diff_to_actual = ((item_sub[predicted].sum() - item_sub[actual].sum()) / item_sub[
                actual].sum()) * 100
            amend_mape = mape(item_sub[actual], item_sub[predicted])

            dp_rmse = mean_squared_error(item_sub[actual], item_sub[forc], squared=False)
            dp_pct_diff_to_actual = ((item_sub[forc].sum() - item_sub[actual].sum()) / item_sub[actual].sum()) * 100
            dp_mape = mape(item_sub[actual], item_sub[forc])

            if abs(amend_pct_diff_to_actual) < abs(dp_pct_diff_to_actual):
                amend_better = 1
                curation_better = 0
            else:
                amend_better = 0
                curation_better = 1

            temp_array = [unique_items[ii], amend_rmse, dp_rmse,
                          amend_pct_diff_to_actual, dp_pct_diff_to_actual,
                          amend_mape, dp_mape,
                          amend_better, curation_better]
            naive_rmse = np.append(naive_rmse, temp_array)
            item_sub.plot(x=date_var, y=[predicted, actual, forc], style="-o", title=item_name)

            fig = plt.gcf()
            fig_name = fig_path + fcst_type + "\\" + "actual_vs_" + fcst_type + "_forc_data" + item_name + "_" + curr_date + ".png"
            if plot_save_flag:
                if not os.path.isdir(fig_path + fcst_type + "\\"):
                    os.mkdir(fig_path + fcst_type + "\\")
                fig.savefig(fig_name)
            plt.close()
        except Exception as ex:
            print("Item " + item_name + " did not work!\n" + str(ex))
            continue

    naive_rmse = np.reshape(naive_rmse, [int(len(naive_rmse) / len(temp_array)), len(temp_array)])
    naive_rmse = pd.DataFrame(naive_rmse, columns=["Item Name", "amend_rmse", "dp_rmse",
                                                   "amend_pct_diff", "dp_pct_diff",
                                                   "amend_mape", "dp_mape",
                                                   "amend_better", "curation_better"])
    naive_rmse["index"] = naive_rmse.index
    cols = naive_rmse.columns.drop("Item Name")
    naive_rmse[cols] = naive_rmse[cols].apply(pd.to_numeric, errors='coerce')
    naive_rmse = pd.merge(naive_rmse, df_unique_items, on="Item Name", how="left")

    return naive_rmse, full_predictions, off_season_predictions