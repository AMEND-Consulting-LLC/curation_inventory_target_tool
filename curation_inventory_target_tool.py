# Import packages
import ctypes
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
import timeit
import time

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog as fd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import date, datetime, timedelta
from holiday_assign import *
import xlwings as xw
import easygui

from xgb_avo_forc import *
from sarimax_avo_forc import *
from naive_avo_forc import *

# Set plot styles
plt.style.use('seaborn-whitegrid')
matplotlib.use("Agg")

start_time = timeit.default_timer()
# # Read in excel file
base_path = "C:\\Users\\"
filetypes = (('All files', '*.*'),
             ('Excel files', '*.xlsx'))
# show the open file dialog
f = fd.askopenfile(initialdir=base_path, filetypes=filetypes)
f.close()
# Read in the parameters and assign them to their respective variables
df_constants = pd.read_excel(f.name, engine="openpyxl")

# f = __file__.rsplit("/", 2)[0] + "/inventory_target_constants.xlsx"
# if ".py" in f:
#     f = __file__.rsplit("\\", 2)[0] + "\\inventory_target_constants.xlsx"
# df_constants = pd.read_excel(f, engine="openpyxl")

# Read in parameters. Make sure that paths exist and notify user of missing parameter
try:
    filepath = df_constants["var"].loc[df_constants["var_name"] == "filepath"].iloc[0]
    market_path = df_constants["var"].loc[df_constants["var_name"] == "market_path"].iloc[0]
    demand_path = df_constants["var"].loc[df_constants["var_name"] == "demand_path"].iloc[0]
    case_conversion_path = df_constants["var"].loc[df_constants["var_name"] == "case_conversion_path"].iloc[0]

    fig_path = df_constants["var"].loc[df_constants["var_name"] == "fig_path"].iloc[0]
    excel_path = df_constants["var"].loc[df_constants["var_name"] == "excel_path"].iloc[0]
    if not os.path.isdir(fig_path) or not os.path.isdir(excel_path):
        ctypes.windll.user32.MessageBoxW(0, "You are missing required paths! Please check "
                                         + f.name + " and try again.", "Missing Parameters", 0)
        raise Exception("Bad Paths")

    plot_save_flag = df_constants["var"].loc[df_constants["var_name"] == "plot_save_flag"].iloc[0]
    gbr_flag = df_constants["var"].loc[df_constants["var_name"] == "gbr_flag"].iloc[0]
    wholesale_factor = df_constants["var"].loc[df_constants["var_name"] == "wholesale_factor"].iloc[0]
    lbs_to_kg = df_constants["var"].loc[df_constants["var_name"] == "lbs_to_kg"].iloc[0]
    kg_to_lbs = 1 / lbs_to_kg
    pallet_cost = df_constants["var"].loc[df_constants["var_name"] == "pallet_cost"].iloc[0]

    perc_start = df_constants["var"].loc[df_constants["var_name"] == "perc_start"].iloc[0]
    perc_end = df_constants["var"].loc[df_constants["var_name"] == "perc_end"].iloc[0]

    pred_window = df_constants["var"].loc[df_constants["var_name"] == "pred_window"].iloc[0]
    offseason_week_start = df_constants["var"].loc[df_constants["var_name"] == "offseason_week_start"].iloc[0]
    offseason_week_end = df_constants["var"].loc[df_constants["var_name"] == "offseason_week_end"].iloc[0]
    forc_year = df_constants["var"].loc[df_constants["var_name"] == "forc_year"].iloc[0]

    seasonality_factor = df_constants["var"].loc[df_constants["var_name"] == "seasonality_factor"].iloc[0]

    offseason_cost = df_constants["var"].loc[df_constants["var_name"] == "offseason_cost"].iloc[0]
    production_cost = df_constants["var"].loc[df_constants["var_name"] == "production_cost"].iloc[0]

    curation_env_flag = df_constants["var"].loc[df_constants["var_name"] == "curation_env_flag"].iloc[0]
except:
    ctypes.windll.user32.MessageBoxW(0, "You are missing a required parameter! Please check : "
                                     + f.name + " and try again.", "Missing Parameters", 0)
    raise Exception("Check parameter file")

# Check that the prediction window is reasonable
if pred_window > 30 or pred_window < 13:
    param_check = ctypes.windll.user32.MessageBoxW(0, "Your testing window is abnormal. "
                                                      "Are you sure you'd like to continue?", "Parameter Check", 4)
    if param_check == 7:
        pred_window = easygui.multenterbox("Please enter your new testing window:", "Test Window Entry",
                             ["Test Window Length (weeks)"], [23])
        pred_window = int(pred_window)

# Check that the user is okay with an abnormal offseason week
if forc_year == pd.Timestamp(date.today()).year and pd.Timestamp(date.today()).week >= offseason_week_start:
    param_check = ctypes.windll.user32.MessageBoxW(0, "Your offseason start week is before the current date. "
                                                      "Are you sure you'd like to continue?", "Parameter Check", 4)
    if param_check == 7:
        offseason_week_start = easygui.multenterbox("Please enter your new offseason start week:", "Test Window Entry",
                             ["Offseason Start Week"], [9])
        offseason_week_start = int(offseason_week_start[0])

# Make sure the offseason weeks are physically possible
if offseason_week_start > offseason_week_end or offseason_week_start < 1 or offseason_week_end > 53:
    ctypes.windll.user32.MessageBoxW(0, "Something is wrong with your offseason window. Please re-enter.",
                                     "Parameter Check", 0)

    offseason_week_start, offseason_week_end = easygui.multenterbox("Please enter your new offseason window:",
                                                                    "Offseason Window Entry",
                                                                    ["Offseason Window Start (week)",
                                                                     "Offseason Window End (week)"], [7, 36])
    offseason_week_start = int(offseason_week_start[0])
    offseason_week_end = int(offseason_week_end[0])

# Update the sales history excel workbook
if curation_env_flag:
    app_excel = xw.App(visible = False)

    wbk = xw.Book(filepath)
    wbk.api.RefreshAll()

    # two options to save
    wbk.save(filepath)
    # kill Excel process
    app_excel.kill()
    del app_excel

# Read in the excel workbooks
df = pd.read_excel(filepath, engine="openpyxl", sheet_name="fact_Sales_Reporting")
df =df.loc[df["Fiscal Week Start Date"] < pd.Timestamp(date.today())]
df_fruit = pd.read_excel(filepath, engine="openpyxl", sheet_name="avo_fruit_pricing")
df_market = pd.read_excel(market_path, engine="openpyxl", sheet_name="avocado_market_sales")
df_data = pd.read_excel(demand_path, engine="openpyxl", sheet_name="Data")
df_products = pd.read_excel(filepath, engine="openpyxl", sheet_name="dim_Products")
df_conv = pd.read_excel(case_conversion_path, engine="openpyxl")

df_data = df_data.drop(
    index=np.array(range(df_data.loc[df_data[df_data.columns[0]] == df_data.columns[0]].index[0] + 1)))
df_data = df_data[["FWeek", "WkDt", "Item", "Values", "Demand Plan", "Lag Fcst", "FPeriod", "FYear"]]
df_data = df_data.loc[df_data["Values"] == "Vol"]
df_data = df_data.fillna(0)
df_data["Item"] = "AP" + df_data["Item"]
df_data["WkDt"] = pd.to_datetime(df_data["WkDt"])
df_data["total_forc"] = df_data["Demand Plan"] + df_data["Lag Fcst"]

date_var = "Fiscal Week Start Date"
full_label = "full"
curr_date = str(date.today())

df = df.sort_values(by=date_var)
df["year"] = df[date_var].dt.year
df["week"] = df[date_var].dt.isocalendar().week

# Run the forecasting functions
if gbr_flag:
    gbr_rmse, gbr_predictions, gbr_off_season = avo_gbr_func(df, df_products, df_data, fig_path, pred_window, plot_save_flag,
                                                             offseason_week_start, offseason_week_end,
                                                             forc_year, fcst_type="gbr")

ts_rmse, ts_predictions, ts_off_season = time_series_func(df, df_products, df_data, plot_save_flag, fig_path,
                                                          pred_window, offseason_week_start, offseason_week_end, 
                                                          forc_year, fcst_type = "sarimax", m = seasonality_factor)

naive_rmse, naive_predictions, naive_off_season = base_avo_model_v2_func(df, df_products, df_data, df_market,
                                                                         plot_save_flag, fig_path, pred_window,
                                                                         offseason_week_start, offseason_week_end,
                                                                         forc_year, fcst_type = "naive")

plt.close('all')

########################################################################################################################
# Process forecasts
########################################################################################################################
# combine the rmse and pct difference from the forecast functions
combined_rmse = pd.DataFrame([ts_rmse["Item Name"], ts_rmse["amend_pct_diff"]]).transpose()
if gbr_flag:
    combined_rmse = pd.merge(combined_rmse, gbr_rmse[["Item Name", "amend_pct_diff"]], on="Item Name", how="left")
combined_rmse = pd.merge(combined_rmse, naive_rmse[["Item Name", "amend_pct_diff"]], on="Item Name", how="left")
if gbr_flag:
    combined_rmse.rename(columns={"amend_pct_diff_x": "ts_amend_pct_diff",
                                  "amend_pct_diff_y": "gbr_amend_pct_diff",
                                  "amend_pct_diff": "naive_amend_pct_diff"}, inplace=True)
else:
    combined_rmse.rename(columns={"amend_pct_diff_x": "ts_amend_pct_diff",
                                  "amend_pct_diff_y": "naive_amend_pct_diff"}, inplace=True)
combined_rmse = combined_rmse.fillna(np.inf)
combined_rmse["min"] = np.nan
for ii in range(combined_rmse.shape[0]):
    item_method_rmse = combined_rmse.loc[:, combined_rmse.columns[combined_rmse.columns != "Item Name"]].iloc[ii]
    combined_rmse["min"].iloc[ii] = item_method_rmse.loc[item_method_rmse.abs() == item_method_rmse.abs().min()]

# Use the combined rmse dataframe to determine which forecast performed the best during the testing period.
combined_off_season_forc = pd.DataFrame()
full_actuals = pd.DataFrame()
full_predictions = pd.DataFrame()
for ii in range(combined_rmse.shape[0]):
    item_name = combined_rmse["Item Name"].iloc[ii]
    min_value = combined_rmse["min"].iloc[ii]
    tf_array = (combined_rmse.iloc[ii] == min_value)
    best_column = combined_rmse.iloc[ii].index[tf_array][0]
    # Set the forecast to be the best forecast available
    if best_column.find("ts_") >= 0:
        df_best_forc = ts_off_season[["date", "Item Name", "predicted_mean"]].loc[
            ts_off_season["Item Name"] == item_name]
        df_best_forc.rename(columns={"predicted_mean": "predicted", "date": "Fiscal Week Start Date"}, inplace=True)
        df_best_forc["type"] = "ts"

        df_best_pred = ts_predictions[["Fiscal Week Start Date", "Item Name", "predicted_mean",
                                       "CasesSum"]].loc[ts_predictions["Item Name"] == item_name]
        df_best_pred.rename(columns={"predicted_mean": "predicted"}, inplace=True)
        df_best_pred["type"] = "ts"
    elif best_column.find("gbr_") >= 0:
        df_best_forc = gbr_off_season[["Fiscal Week Start Date", "Item Name", "predicted"]].loc[
            gbr_off_season["Item Name"] == item_name]
        df_best_forc["type"] = "gbr"

        df_best_pred = gbr_predictions[["date", "Item Number", "predicted",
                                        "CasesSum"]].loc[gbr_predictions["Item Number"] == item_name]
        df_best_pred.rename(columns={"Item Number": "Item Name", "date": "Fiscal Week Start Date"}, inplace=True)
        df_best_pred["type"] = "gbr"
    elif best_column.find("naive_") >= 0:
        df_best_forc = naive_off_season[["Fiscal Week Start Date", "Item Name", "predicted"]].loc[
            naive_off_season["Item Name"] == item_name]
        df_best_forc["type"] = "naive"

        df_best_pred = naive_predictions[["Fiscal Week Start Date", "Item Name", "predicted",
                                          "CasesSum"]].loc[naive_predictions["Item Name"] == item_name]
        df_best_pred["type"] = "naive"

    # Remove wholesale buys from the predictions
    df_best_forc["predicted"].loc[df_best_forc["predicted"] >= wholesale_factor * df_best_forc["predicted"].mean()] = \
        df_best_forc["predicted"].loc[
            df_best_forc["predicted"] < wholesale_factor * df_best_forc["predicted"].mean()].mean()

    # Add the best prediction to the prediction data frame, and the best forecast to the forecast data frame
    full_predictions = full_predictions.append(df_best_pred, ignore_index=True)
    combined_off_season_forc = combined_off_season_forc.append(df_best_forc, ignore_index=True)
    df_best_forc["week"] = df_best_forc[date_var].dt.isocalendar().week

    # Get actuals
    df_item_temp = df.loc[(df["Item Number"] == item_name)]
    cases_sum = df_item_temp["Cases"].groupby(df_item_temp[date_var].dt.to_period("W-MON")).sum()
    df_actuals = df_item_temp.loc[df_item_temp["Item Number"] == item_name]
    df_actuals = df_actuals[[date_var, "Item Number"]].drop_duplicates()
    df_actuals["CasesSum"] = cases_sum.to_numpy()
    df_actuals = df_actuals.set_index(date_var).asfreq('W-MON')
    df_actuals = df_actuals.fillna(0)
    df_actuals.reset_index(inplace=True)
    df_actuals["week"] = df_actuals[date_var].dt.isocalendar().week

    full_actuals = full_actuals.append(df_actuals, ignore_index=True)

    # Plot actuals vs forecast
    fig_name = fig_path + full_label + "\\" + "actual_vs_" + full_label + "_forc_data" + item_name + "_" + curr_date + ".png"
    fig = plt.figure()

    plt.plot(df_actuals["Fiscal Week Start Date"], df_actuals["CasesSum"], "-o", label="Sales")
    plt.plot(df_best_forc["Fiscal Week Start Date"], df_best_forc["predicted"], "-o", label="Forecast")

    actual_years = pd.unique(df_actuals[date_var].dt.year)
    for jj in range(len(actual_years)):
        sub = df_actuals.loc[(df_actuals[date_var].dt.year == actual_years[jj]) &
                             (df_actuals["week"] >= offseason_week_start) &
                             (df_actuals["week"] <= offseason_week_end)]
        if len(sub) != 0:
            med_case = sub["CasesSum"].mean()
            plt.plot([sub[date_var].min(), sub[date_var].max()],
                     [med_case, med_case], "k--")
    plt.plot([df_best_forc[date_var].min(), df_best_forc[date_var].max()],
             [df_best_forc["predicted"].mean(), df_best_forc["predicted"].mean()], "k--",
             label="Off-Season Mean")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.title("Sales Actuals vs Future Predictions for Item : " + item_name)
    plt.legend()

    if plot_save_flag:
        if not os.path.isdir(fig_path + full_label + "\\"):
            os.mkdir(fig_path + full_label + "\\")
        fig.savefig(fig_name)
    plt.close('all')

# Post process the data frames to line up column names and sorting
max_df_year = df_actuals[date_var].dt.year.max()
combined_off_season_forc["predicted"].loc[combined_off_season_forc["predicted"] < 0] = 0
actuals_by_item = full_actuals["CasesSum"].loc[full_actuals[date_var].dt.year == max_df_year].groupby(
    full_actuals["Item Number"]).sum()
actuals_by_item = actuals_by_item.reset_index()
actuals_by_item.rename(columns={"Item Number": "Item Name"}, inplace=True)
combined_rmse = pd.merge(combined_rmse, actuals_by_item, on="Item Name", how="left")
combined_rmse.sort_values(by="CasesSum", inplace=True, ascending=False)

########################################################################################################################
# Do a cost simulation
########################################################################################################################
# Get fruit pricing for cost optimization
df_fruit["mean"] = df_fruit.max(axis=1)
grouped_weeks = df["Fiscal Period"].groupby(by=df["week"])

perc_forc_range = np.linspace(perc_start, perc_end, perc_end - perc_start + 1)
string_range = [str(int(i)) for i in perc_forc_range]
perc_names = [s + "_perc_forc" for s in string_range]

item_cost = pd.DataFrame()
item_cost["Item Name"] = np.nan
item_cost[perc_names] = np.nan
predicted_items = pd.unique(full_predictions["Item Name"])

for ii in range(len(predicted_items)):
    try:
        item_name = predicted_items[ii]
        item_sub = full_predictions.loc[full_predictions["Item Name"] == item_name]
        item_sub["week"] = item_sub[date_var].dt.isocalendar().week
        item_sub["fiscal_period"] = np.nan
        item_sub["fruit_cost"] = np.nan
        for ll in range(item_sub.shape[0]):
            item_sub["fiscal_period"].iloc[ll] = pd.unique(grouped_weeks.get_group(item_sub["week"].iloc[ll]))[0]
            item_sub["fruit_cost"].iloc[ll] = df_fruit["mean"].iloc[int(item_sub["fiscal_period"].iloc[ll])]

        cost_sub = pd.DataFrame([item_name], columns=["Item Name"])
        conv_sub = df_conv.loc[(df_conv["Item #"] == item_name[2:]) |
                               (df_conv["Item #"] == int(item_name[2:]))]
        lbs_per_case = pd.unique(conv_sub["Lbs. per Case"])[0]
        kg_per_case = lbs_per_case * 1 / lbs_to_kg
        cases_per_pallet = pd.unique(conv_sub["Cases per Pallet"])[0]

        for jj in range(len(perc_forc_range)):
            if item_sub["predicted"].sum() == 0:
                tag = perc_names[jj]
                cost_sub[tag] = 0
                continue
            else:
                tag = perc_names[jj]
                factor = perc_forc_range[jj] / 100
                item_sub["predicted_adj"] = item_sub["predicted"] * factor

                item_sub["forc_diff"] = item_sub["predicted_adj"] - item_sub["CasesSum"]

                if item_sub["forc_diff"].sum() > 0:
                    total_cost = item_sub["forc_diff"].sum() * production_cost
                elif item_sub["forc_diff"].sum() < 0:
                    total_cost = abs(item_sub["forc_diff"].sum() * offseason_cost)

                cost_sub[tag] = total_cost
        item_cost = item_cost.append(cost_sub, ignore_index=True)
    except Exception as ex:
        print("Something went wrong….", ex, item_name)
        continue

item_cost = item_cost.append(item_cost.sum(), ignore_index=True)
item_cost["Item Name"].iloc[-1] = "sum"
min_sum = item_cost.min(axis=1).iloc[-1]
best_tag = item_cost.columns[item_cost.iloc[-1] == min_sum][0]
plot_cost = item_cost.drop(columns="Item Name")
best_index = plot_cost.columns.get_loc(best_tag)
pct_diff_array = np.diff(plot_cost.iloc[-1][best_index:])/plot_cost.iloc[-1][best_index + 1:]
knee_tag = pct_diff_array[pct_diff_array > pct_diff_array.median()].index[0]
best_index = plot_cost.columns.get_loc(knee_tag)
forc_factor = perc_forc_range[best_index] / 100

fig = plt.figure()
plt.plot(perc_forc_range, plot_cost.iloc[-1], "-o")
plt.plot(perc_forc_range[best_index], plot_cost.iloc[-1][best_index], "o")
plt.xlabel("Percentage of Forecast (%)")
plt.ylabel("Cost (Factored Cases)")
plt.title("Cost Curve for Forecast Percentages")

fig_name = fig_path + "all_item_compare" + "\\" + "cost_curve_oos_holding_cost_" + curr_date + ".png"
if plot_save_flag:
    if not os.path.isdir(fig_path + "all_item_compare" + "\\"):
        os.mkdir(fig_path + "all_item_compare" + "\\")
    fig.savefig(fig_name)

plt.close()

excel_filename = "curation_avocado_cost_totals" + str(datetime.utcnow())
excel_filename = excel_filename.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".",
                                                                                              "_") + ".xlsx"
item_cost.to_excel(excel_path + excel_filename)

########################################################################################################################
# Create a combined volume dataframe
########################################################################################################################
df_off_by_off = pd.DataFrame()
cases_per_week = np.array([])
combined_off_season_forc["predicted"] = combined_off_season_forc["predicted"] * forc_factor
for ii in range(combined_rmse.shape[0]):
    try:
        item_case_array = np.array([])
        item_name = combined_rmse["Item Name"].iloc[ii]
        forc_sub = combined_off_season_forc.loc[combined_off_season_forc["Item Name"] == item_name]

        min_forc_date = forc_sub["Fiscal Week Start Date"].min()
        min_forc_year = min_forc_date.min
        max_forc_date = forc_sub["Fiscal Week Start Date"].max()
        max_forc_year = max_forc_date.max

        df_sub = df.loc[(df["Item Number"] == item_name)]
        min_df_year = df_sub["Fiscal Week Start Date"].dt.year.min()
        max_df_year = df_sub["Fiscal Week Start Date"].dt.year.max()

        if pd.Timestamp(date.today()).week < offseason_week_start and pd.Timestamp(date.today()).year == max_df_year:
            max_df_year -= 1

        year_week_df = pd.DataFrame()
        fig = plt.figure()
        fig.set_size_inches(12, 6.8)
        plt_index = 0
        try:
            for year in range(min_df_year, max_df_year + 1):
                df_year_sub = df_sub.loc[(df_sub["year"] == year) &
                                         (df_sub["week"] >= offseason_week_start) &
                                         (df_sub["week"] <= offseason_week_end)]
                total_weeks = df_year_sub["week"].max() - df_year_sub["week"].min()
                total_cases = df_year_sub["Cases"].groupby(df_year_sub[date_var]).sum()

                df_cases = df_year_sub[["Fiscal Week Start Date", "Item Number"]].drop_duplicates()
                df_cases["CasesSum"] = total_cases.to_numpy()
                df_cases = df_cases.set_index(date_var).asfreq('W-MON')
                df_cases = df_cases.reset_index()
                df_cases["year_week"] = df_cases[date_var].dt.year.astype(dtype="string") + "_" + \
                                        df_cases[date_var].dt.isocalendar().week.astype(dtype="string")
                df_temp = pd.DataFrame(columns=df_cases["year_week"])
                df_temp.loc[0] = df_cases["CasesSum"].to_numpy()
                if not plot_save_flag:
                    df_temp[str("sum_" + str(year))] = df_cases["CasesSum"].sum()
                year_week_df = pd.concat([year_week_df, df_temp], axis=1)

                plt_range = list(range(plt_index, plt_index + df_temp.shape[1]))
                plt_index += df_temp.shape[1]
                plt.plot(plt_range, df_temp.iloc[0], "-o", label="Sales " + str(year))
                plt.plot([plt_range[0], plt_range[-1]], [df_temp.iloc[0].mean(), df_temp.iloc[0].mean()], "k--")

        except Exception as ex:
            print(item_name, " can't plot : ", ex)

        # Get the forecast
        forc_item_sub = combined_off_season_forc.loc[combined_off_season_forc["Item Name"] == item_name]
        forc_item_sub["year_week"] = forc_item_sub[date_var].dt.year.astype(dtype="string") + "_" + \
                                     forc_item_sub[date_var].dt.isocalendar().week.astype(dtype="string")
        year = forc_item_sub[date_var].dt.year.max()
        df_temp = pd.DataFrame(columns=forc_item_sub["year_week"])
        df_temp.loc[0] = forc_item_sub["predicted"].to_numpy()
        if not plot_save_flag:
            df_temp[str("sum_" + str(year))] = forc_item_sub["predicted"].sum()
        year_week_df = pd.concat([year_week_df, df_temp], axis=1)

        year_week_df.rename(index={0: item_name}, inplace=True)
        df_off_by_off = df_off_by_off.append(year_week_df)
        df_off_by_off = df_off_by_off.fillna(0)

        plt_range = list(range(plt_index, plt_index + df_temp.shape[1]))
        plt_index += df_temp.shape[1]
        plt.plot(plt_range, df_temp.iloc[0], "-o", label="Forecasted " + str(year))
        plt.plot([plt_range[0], plt_range[-1]], [df_temp.iloc[0].mean(), df_temp.iloc[0].mean()], "k--",
                 label="Mean")
        plt.xlabel("Index")
        plt.ylabel("Cases")
        plt.title("Comparison of Off Season Sales an Forecast for Item : " + item_name +
                  "\nWeeks : " + str(offseason_week_start) + " to " + str(offseason_week_end))
        plt.legend()

        fig_name = fig_path + "off_season_compare" + "\\" + "actual_vs_" + "off_season_compare" \
                   + "_forc_data" + item_name + "_" + curr_date + ".png"
        if plot_save_flag:
            if not os.path.isdir(fig_path + "off_season_compare" + "\\"):
                os.mkdir(fig_path + "off_season_compare" + "\\")
            fig.savefig(fig_name)
        plt.close()

        plt.close('all')
    except Exception as ex:
        print("Something went wrong….", ex, item_name)
    continue

########################################################
# Additional Visualization
########################################################
# Percent diff by sales volume
fig = plt.figure()
fig.set_size_inches(12, 6.8)
plt.plot(combined_rmse["CasesSum"], combined_rmse["min"], "-o")
plt.xlabel("Total Volume for Previous Year (Cases)")
plt.ylabel("Forecasted Percent Difference (%)")
plt.title("Percent Difference for Generated Avocado Product Forecasts")

fig_name = fig_path + "all_item_compare" + "\\" + "pct_diff_by_volume_" + curr_date + ".png"
if plot_save_flag:
    if not os.path.isdir(fig_path + "all_item_compare" + "\\"):
        os.mkdir(fig_path + "all_item_compare" + "\\")
    fig.savefig(fig_name)
plt.close()

# Histogram of Pct Diff
fig = plt.figure()
fig.set_size_inches(12, 6.8)
sns.distplot(combined_rmse["min"],
             bins=100,
             kde=True,
             color='skyblue',
             hist_kws={"linewidth": 15, 'alpha': 1})
# plt.xlim([-200, 200])
plt.xlabel("Percent Difference (%)")
plt.ylabel("Probability (%)")
plt.title("Distribution of Forecast Percent Difference by Item")

fig_name = fig_path + "all_item_compare" + "\\" + "pct_diff_hist_" + curr_date + ".png"
if plot_save_flag:
    if not os.path.isdir(fig_path + "all_item_compare" + "\\"):
        os.mkdir(fig_path + "all_item_compare" + "\\")
    fig.savefig(fig_name)
plt.close()

excel_filename = "curation_avocado_" + "off_season" + "_year_by_year_" + str(datetime.utcnow())
excel_filename = excel_filename.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_") + ".xlsx"
off_sub = df_off_by_off[df_off_by_off.columns[df_off_by_off.shape[1] - 1 - offseason_week_end + offseason_week_start:]]
off_sum = off_sub.sum(axis = 1)
df_off_by_off["sum"] = off_sum
df_off_by_off.to_excel(excel_path + excel_filename)

excel_filename = "curation_avocado_" + "full" + "_forecast_" + str(datetime.utcnow())
excel_filename = excel_filename.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_") + ".xlsx"
combined_off_season_forc.to_excel(excel_path + excel_filename)

print(timeit.default_timer() - start_time)

ctypes.windll.user32.MessageBoxW(0, "The Inventory Target program has run successfully", "Program Success", 0)
