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
import xgboost as xgb

def avo_gbr_func(df, df_products, df_data, fig_path, pred_window, plot_save_flag,
                 offseason_week_start, offseason_week_end, fcst_type="gbr"):
    # Initialize values
    date_var = "Fiscal Week Start Date"
    us_holidays = holidays.UnitedStates()

    curr_date = datetime.today().strftime("%Y_%m_%d")

    actual = "CasesSum"
    predicted = "predicted"
    forc = "sum_forc"

    keep_cols = ["Fiscal Year", "holiday",
                 "Fiscal Period", "Fiscal Week", "Fiscal Week Start Date"]
    pred_end = df["Fiscal Week Start Date"].max()
    pred_start = pred_end - timedelta(weeks=pred_window - 1)

    off_start = pred_end + timedelta(weeks=1)
    off_end = off_start + timedelta(weeks=104)

    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    # Setting plot characteristics
    plt.ioff()
    # sns.set()

    # Sort dataframe by date
    df = df.sort_values(by=date_var)
    df["year"] = df[date_var].dt.year
    df["month"] = df[date_var].dt.month
    df["week"] = df[date_var].dt.isocalendar().week

    # Assign holiday values to the sales data dates
    df = avo_holiday_assign(df, date_var, us_holidays)

    # subset the dataframe by item and possibly by year
    # df_item_sub = subset_by_item_and_year(df, num_items, unique_items, year_var, year_sep_flag, date_var)
    #####Do the GB Thing!###################################################################################################
    unique_items = pd.unique(df["Item Number"].loc[df[date_var] >= (pd.Timestamp(pred_end) - timedelta(weeks = 3))])
    df_unique_items = pd.DataFrame(unique_items, columns = ["Item Number"])
    item_family_df = df_products[["Item Number", "Family"]]
    df_unique_items = pd.merge(df_unique_items, item_family_df, on="Item Number", how="left")

    naive_rmse = np.array([])
    full_predictions = pd.DataFrame()
    off_season_predictions = pd.DataFrame()
    for ii in range(len(df_unique_items)):
        try:
            item_name = unique_items[ii]
            df_item_temp = df.loc[(df["Item Number"] == item_name) & (df[date_var] < pred_start)]
            cases_sum = df_item_temp["Cases"].groupby(df_item_temp[date_var].dt.to_period("W-MON")).sum()
            df_forc = df_item_temp.loc[df_item_temp["Item Number"] == item_name]
            df_forc = df_forc[keep_cols].drop_duplicates()
            df_forc["CasesSum"] = cases_sum.to_numpy()
            df_forc = df_forc.drop(date_var, axis=1)

            df_forc_wDate = df.loc[(df["Item Number"] == item_name) & (df[date_var] >= pred_start)]
            cases_sum = df_forc_wDate["Cases"].groupby(df_forc_wDate[date_var].dt.to_period("W-MON")).sum()
            df_forc_wDate = df_forc_wDate.loc[df_forc_wDate["Item Number"] == item_name]
            df_forc_wDate = df_forc_wDate[keep_cols].drop_duplicates()
            df_forc_wDate["CasesSum"] = cases_sum.to_numpy()


            y, X = df_forc.loc[:, "CasesSum"].values, df_forc.loc[:,
                                                      df_forc.columns[df_forc.columns != "CasesSum"]].values

            data_dmatrix = xgb.DMatrix(X, label=y)

            # Test Train Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

            # Regression Model
            reg_mod = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.08,
                subsample=0.75,
                colsample_bytree=1,
                max_depth=7,
                gamma=0,
                seed = 123
            )
            reg_mod.fit(X_train, y_train)
            xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                             importance_type='gain', interaction_constraints='',
                             learning_rate=0.08, max_delta_step=0, max_depth=7,
                             min_child_weight=1, missing=np.nan, monotone_constraints='()',
                             n_estimators=1000, n_jobs=8, num_parallel_tree=1, random_state=0,
                             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.75,
                             tree_method='exact', validate_parameters=1, verbosity=None, seed = 123)

            # Test Fit
            reg_mod.fit(X_train, y_train)
            predictions = reg_mod.predict(X_test)
            predictions[predictions < 0] = 0

            # Create future predictions
            forecast_predict = df_forc_wDate

            forecast_predict = forecast_predict.drop(["CasesSum"], axis=1)
            forecast = forecast_predict
            forecast["predicted"] = reg_mod.predict(forecast.drop(date_var, axis = 1))
            forecast["date"] = forecast[date_var]

            # Build an independent set of offseason predictions
            date_range = pd.date_range(start=off_start,
                                       end=off_end, freq='W-MON')
            off_season_forecast = pd.DataFrame(date_range, columns=[date_var])
            off_season_forecast["week"] = off_season_forecast[date_var].dt.isocalendar().week

            off_season_start_index = off_season_forecast.loc[off_season_forecast["week"] == offseason_week_start].index[0]
            off_season_end_index = off_season_forecast.loc[off_season_forecast["week"] == offseason_week_end].index[0]
            if off_season_start_index > off_season_end_index:
                off_season_start_index = off_season_forecast.loc[off_season_forecast["week"]
                                                                 == offseason_week_end].index[1]

            off_season_forecast = off_season_forecast.iloc[off_season_start_index
                                                           :off_season_end_index + 1]
            fy_info = df_data[["FWeek", "FYear", "FPeriod", "WkDt"]].loc[df_data["WkDt"] >=
                                                                         off_season_forecast[date_var].min()].drop_duplicates()
            fy_info = fy_info.sort_values(by="WkDt")
            fy_info.rename(columns = {"WkDt" : date_var}, inplace = True)

            off_season_forecast = pd.merge(off_season_forecast, fy_info, on=date_var, how="left")
            off_season_forecast["predicted"] = reg_mod.predict(off_season_forecast.drop(date_var, axis = 1))
            off_season_forecast.rename(columns = {date_var : "date"})
            off_season_forecast = avo_holiday_assign(off_season_forecast, date_var, us_holidays)
            off_season_forecast["Item Name"] = item_name
            off_season_forecast["predicted"].loc[off_season_forecast["predicted"] < 0] = 0
            off_season_predictions = off_season_predictions.append(off_season_forecast, ignore_index=True)

            # Get dp predictions
            data_sub = df_data.loc[df_data["Item"] == item_name]
            data_sub["date"] = data_sub["WkDt"]
            sum_forc = data_sub["total_forc"].groupby(data_sub["date"].dt.to_period("W-MON")).sum()
            sum_forc = sum_forc.to_numpy()
            data_sub = pd.DataFrame(data_sub["date"].drop_duplicates())
            data_sub["sum_forc"] = sum_forc

            # Get actual values
            item_sub = pd.merge(forecast, data_sub, on = "date", how = "left")

            df_forc_wDate["date"] = df_forc_wDate[date_var]
            item_sub = pd.merge(item_sub, df_forc_wDate, on="date", how="left")
            item_sub = item_sub.fillna(0)
            item_sub["Item Number"] = item_name
            item_sub = item_sub[["Item Number", "date","predicted", "sum_forc", "CasesSum"]]
            item_sub = item_sub.loc[(item_sub["date"] >= pred_start) & (item_sub["date"] <= pred_end)]
            full_predictions = full_predictions.append(item_sub, ignore_index=True)

            #Visualize
            amend_rmse = mean_squared_error(item_sub[actual], item_sub[predicted], squared=False)
            amend_pct_diff_to_actual = ((item_sub[predicted].sum() - item_sub[actual].sum()) / item_sub[actual].sum()) * 100
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
            item_sub.plot(x="date", y=[predicted, actual, forc], style="-o", title=item_name)

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

    naive_rmse = np.reshape(naive_rmse, [int(len(naive_rmse)/len(temp_array)), len(temp_array)])
    naive_rmse = pd.DataFrame(naive_rmse, columns=["Item Name", "amend_rmse", "dp_rmse",
                                                   "amend_pct_diff", "dp_pct_diff",
                                                   "amend_mape", "dp_mape",
                                                   "amend_better", "curation_better"])
    naive_rmse["index"] = naive_rmse.index
    cols = naive_rmse.columns.drop("Item Name")
    naive_rmse[cols] = naive_rmse[cols].apply(pd.to_numeric, errors='coerce')
    print(naive_rmse.median())
    print("AMEND is better " + str(naive_rmse["amend_better"].sum() / (len(naive_rmse))) + " percent of the time")

    naive_rmse.rename(columns = {"Item Name" : "Item Number"}, inplace=True)
    full_predictions = pd.merge(full_predictions, df_unique_items, on = "Item Number", how = "left")
    naive_rmse = pd.merge(naive_rmse, df_unique_items, on = "Item Number", how = "left")
    naive_rmse.rename(columns={"Item Number": "Item Name"}, inplace=True)

    return naive_rmse, full_predictions, off_season_predictions