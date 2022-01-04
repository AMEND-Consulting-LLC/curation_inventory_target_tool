import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import holidays
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta

from holiday_assign import *

def base_avo_model_v2_func(df, df_products, df_data, df_market, plot_save_flag, fig_path, pred_window,
                           offseason_week_start, offseason_week_end, fcst_type):
    # Initialize values
    unique_items = pd.unique(df["Item Number"])
    date_var = "Fiscal Week Start Date"
    year_var = "Fiscal Year"
    us_holidays = holidays.UnitedStates()

    curr_date = datetime.today().strftime("%Y_%m_%d")

    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    actual = "CasesSum"
    predicted = "predicted"
    forc = "sum_forc"

    keep_cols = ["Fiscal Week Start Date"]

    pred_end = df["Fiscal Week Start Date"].max()
    pred_start = pred_end - timedelta(weeks=pred_window - 1)

    off_start = pred_end + timedelta(weeks=1)
    off_end = off_start + timedelta(weeks=104)

    # Setting plot characteristics
    plt.ioff()
    # sns.set()

    # Sort dataframe by date
    df = df.sort_values(by=date_var)
    df["year"] = df[date_var].dt.year
    df["week"] = df[date_var].dt.isocalendar().week

    # Assign holiday values to the sales data dates
    df = avo_holiday_assign(df, date_var, us_holidays)

    ########################################################################################################################
    # Naive
    # Get yoy sales numbers for each item
    yoy_change = np.array([])
    sales_by_week = np.array([])
    for ii in range(len(unique_items)):
        # Get item data subset
        item_no = unique_items[ii]
        df_item = df.loc[df["Item Number"] == item_no]
        # Get min and max FY
        min_fy = df_item[year_var].iloc[0]
        max_fy = df_item[year_var].iloc[-1]
        year_range = np.linspace(min_fy, max_fy, max_fy - min_fy + 1)

        year_by_year_cases = np.array([])
        year_by_year_weeks = np.array([])
        if (max_fy - min_fy) == (df[year_var].max() - df[year_var].min()):
            yoy_change = np.append(yoy_change, item_no)
            sales_by_week = np.append(sales_by_week, item_no)
            # Get FY information
            for jj in range(len(year_range)):
                fy = year_range[jj]
                df_fy = df_item.loc[df_item[year_var] == fy]
                fy_item_weeks = len(pd.unique(df_fy["Fiscal Week"]))
                fy_weeks = len(pd.unique(df["Fiscal Week"].loc[df[year_var] == fy]))

                year_by_year_cases = np.append(year_by_year_cases, df_fy["Cases"].sum())
                year_by_year_weeks = np.append(year_by_year_weeks, fy_item_weeks)

            avg_cases_per_week = year_by_year_cases / year_by_year_weeks
            yoy_change = np.append(yoy_change,
                                   (avg_cases_per_week[1:] - avg_cases_per_week[:-1]) / avg_cases_per_week[:-1] * 100)
            sales_by_week = np.append(sales_by_week, avg_cases_per_week)

    #
    # Establish yoy change array
    yoy_change = np.reshape(yoy_change, (int(len(yoy_change) / (jj + 1)), jj + 1))
    df_yoy = pd.DataFrame(yoy_change, columns=["ItemNo", "fy21tofy20", "fy22tofy21"])
    cols = df_yoy.columns.drop("ItemNo")
    df_yoy[cols] = df_yoy[cols].apply(pd.to_numeric, errors='coerce')
    df_yoy["median"] = df_yoy[cols].median(axis=1).fillna(0)

    # Same for sales
    sales_by_week = np.reshape(sales_by_week, (int(len(sales_by_week) / (jj + 2)), jj + 2))
    sales_by_week = pd.DataFrame(sales_by_week, columns=["ItemNo", "fy20", "fy21", "fy22"])
    cols = sales_by_week.columns.drop("ItemNo")
    sales_by_week[cols] = sales_by_week[cols].apply(pd.to_numeric, errors='coerce')

    ########################################################################################################################
    # Get matrix of market data
    ########################################################################################################################
    market_array = df_market.to_numpy()
    market_cols = df_market.columns

    yoy_market_perc = (market_array[:, 1:] - market_array[:, :-1]) / market_array[:, :-1] * 100
    yoy_market_perc_linear = np.reshape(yoy_market_perc, [np.shape(yoy_market_perc)[0] * np.shape(yoy_market_perc)[1]],
                                        order="F")
    market_linear = np.reshape(market_array, [np.shape(market_array)[0] * np.shape(market_array)[1]], order="F")

    yoy_market_perc_linear = yoy_market_perc_linear[~np.isnan(yoy_market_perc_linear)]
    market_linear = market_linear[~np.isnan(market_linear)]

    # Create column labels
    label_array = np.array([])
    for ii in range(len(market_cols)):
        if ii > 0:
            label = market_cols[ii] + "to" + market_cols[ii - 1]
            label_array = np.append(label_array, label)

    df_yoy_market = pd.DataFrame(yoy_market_perc, columns=label_array)
    df_yoy_market = df_yoy_market.drop(label_array[-1], axis=1)
    df_yoy_market["median"] = np.nanmedian(df_yoy_market, axis=1)
    df_yoy_market["week"] = df_yoy_market.index + 1

    ########################################################################################################################
    # Apply and evaluate naive trends
    ########################################################################################################################
    unique_items = pd.unique(df_yoy["ItemNo"])
    df_unique_items = pd.DataFrame(unique_items, columns=["Item Number"])
    item_family_df = df_products[["Item Number", "Family"]]
    df_unique_items = pd.merge(df_unique_items, item_family_df, on="Item Number", how="left")
    df_unique_items.rename(columns={'Item Number': "Item Name"}, inplace=True)

    naive_rmse = np.array([])
    full_predictions = pd.DataFrame()
    off_season_predictions = pd.DataFrame()
    safety_factor = 0
    for ii in range(len(df_unique_items)):
    # for ii in range(14, 15):
        try:
            item_name = unique_items[ii]
            df_item_temp = df.loc[(df["Item Number"] == item_name)]
            cases_sum = df_item_temp["Cases"].groupby(df_item_temp[date_var].dt.to_period("W-MON")).sum()
            df_forc = df_item_temp.loc[df_item_temp["Item Number"] == item_name]
            df_forc = df_forc[keep_cols].drop_duplicates()
            df_forc["CasesSum"] = cases_sum.to_numpy()
            df_forc = df_forc.set_index(date_var).asfreq('W-MON')
            df_forc = df_forc.fillna(0)
            df_forc.reset_index(inplace=True)
            df_forc["week"] = df_forc[date_var].dt.isocalendar().week
            df_forc["factor"] = np.nan
            for jj in range(df_forc.shape[0]):
                market_factor = df_yoy_market["median"].loc[df_yoy_market["week"] == df_forc["week"].iloc[jj]].sum()
                item_factor = df_yoy["median"].loc[df_yoy["ItemNo"] == item_name].sum()
                df_forc["factor"].iloc[jj] = 1 + (market_factor + item_factor + safety_factor) / 100
            df_forc[predicted] = df_forc[actual] * df_forc["factor"]

            df_predict = df_forc.drop(["CasesSum", "week"], axis=1)
            df_predict[date_var] = df_predict[date_var] + timedelta(weeks=52)

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
            off_season_forecast["predicted"] = np.nan
            off_season_forecast["Item Name"] = item_name
            for mm in range(off_season_forecast.shape[0]):
                off_season_week = off_season_forecast["week"].iloc[mm]

                prev_year_val = df_forc["CasesSum"].loc[df_forc["week"] == off_season_week].iloc[-1]

                market_factor = df_yoy_market["median"].loc[df_yoy_market["week"] == off_season_week].sum()
                item_factor = df_yoy["median"].loc[df_yoy["ItemNo"] == item_name].sum()
                factor = 1 + (market_factor + item_factor + safety_factor) / 100

                off_season_forecast["predicted"].iloc[mm] = prev_year_val * factor
                off_season_forecast["predicted"].loc[off_season_forecast["predicted"] < 0] = 0

            off_season_predictions = off_season_predictions.append(off_season_forecast, ignore_index=True)
            # Get dp predictions
            data_sub = df_data.loc[df_data["Item"] == item_name]
            data_sub["date"] = data_sub["WkDt"]
            sum_forc = data_sub["total_forc"].groupby(data_sub["date"].dt.to_period("W-MON")).sum()
            sum_forc = sum_forc.to_numpy()
            data_sub = pd.DataFrame(data_sub["date"].drop_duplicates())
            data_sub["sum_forc"] = sum_forc

            data_sub.rename(columns={"date": date_var}, inplace=True)
            item_sub = pd.merge(df_predict, data_sub, on=date_var, how="left")
            item_sub = pd.merge(item_sub, df_forc[[date_var, "CasesSum"]], on=date_var, how="left")
            item_sub.dropna(inplace=True)

            test_sub = item_sub.loc[(item_sub[date_var] < pred_end) &
                                    (item_sub[date_var] >= pred_start)]
            prev_adj_factor = 1 - ((test_sub["predicted"].sum() - test_sub["CasesSum"].sum()) / test_sub["CasesSum"].sum())
            item_sub["predicted"] = item_sub["predicted"] * prev_adj_factor
            item_sub["Item Name"] = item_name

            item_sub = item_sub.loc[(item_sub[date_var] >= pred_start) &
                                    (item_sub[date_var] <= pred_end)]

            full_predictions = full_predictions.append(item_sub, ignore_index=True)

            # Visualize
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
    print(naive_rmse.median())
    print("AMEND is better " + str(naive_rmse["amend_better"].sum() / (len(naive_rmse))) + " percent of the time")

    # getting data of the histogram
    fig = plt.figure()
    sns.distplot(naive_rmse["amend_pct_diff"].loc[naive_rmse["amend_pct_diff"] < 10000],
                 bins=100,
                 kde=True,
                 color='skyblue',
                 hist_kws={"linewidth": 15, 'alpha': 1})
    sns.distplot(naive_rmse["dp_pct_diff"].loc[naive_rmse["dp_pct_diff"] < 10000],
                 bins=100,
                 kde=True,
                 color='darksalmon',
                 hist_kws={"linewidth": 15, 'alpha': 1})
    plt.xlim([-200, 200])
    plt.xlabel("Pct Diff (%)")
    plt.ylabel("Probability (%)")
    plt.title("Pct Diff Distribution for Gradient Boosted")
    # plt.show()

    threshold = 15
    pct_better_than_threshold = len(naive_rmse["dp_pct_diff"].loc[(naive_rmse["dp_pct_diff"] <= threshold) &
                                                                  (naive_rmse["dp_pct_diff"] >= -threshold)]) / len(
        naive_rmse)
    print(pct_better_than_threshold)

    # excel_filename = "curation_avocado_" + fcst_type + "_forecast_" + str(datetime.utcnow())
    # excel_filename = excel_filename.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_") + ".xlsx"
    # full_predictions.to_excel("C:\\Users\\JayCarroll\\Documents\\AMEND\\Curation\\Forecasts\\" + excel_filename)
    return naive_rmse, full_predictions, off_season_predictions