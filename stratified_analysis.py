"""
Stratified analysis on model performance

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import sys
import pandas as pd
pd.set_option('display.precision', 3)
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import numpy as np
import torch
from utils.clf_metrics import acc, sen, spe



def stratify_on_country(df_in, train_csv_in, best_guids, mode = 'pat'):
    
    df = df_in.copy()
    train_csv = train_csv_in.copy()
    
    categs = [[] for _ in range(3)]
    # (a) All included
    categs[0] = train_csv['ICGUID'].tolist()
    # (b) US only
    categs[1] = train_csv[(train_csv['USorUK'] == 'US')
                          ]['ICGUID'].tolist()
    # (c) EU only
    categs[2] = train_csv[(train_csv['USorUK'] == 'UK')
                          ]['ICGUID'].tolist()
    
    res = pd.DataFrame()
    for ind, cur_guids in enumerate(categs):
        
        cur_df = df[df['ICGUID'].isin(cur_guids)].reset_index(drop = True)
        
        if mode == 'pat':
            num = len(cur_df[['subject_number', 'Visit Number']].drop_duplicates())
            pos = cur_df[['subject_number', 'Visit Number', 'GT']].drop_duplicates()['GT'].sum()
        elif mode == 'img':
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        else:
            cur_df = cur_df[cur_df['ICGUID'].isin(best_guids)].reset_index(drop = True)
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        
        if ind == 0: categ = "All included"
        elif ind == 1: categ = "US"
        else: categ = "EU"
        categ += f" - N = {num} || {pos} pos"
        
        if mode == 'pat':
            preds = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
            targets = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['GT'].mean().values)
        else:
            preds = torch.tensor(cur_df['Pred_Proba'].values)
            targets = torch.tensor(cur_df['GT'].values)
        
        info = {
            'group': categ,
            'acc': None,
            'sen': None,
            'spe': None
        }
        for metric_fn in [acc, sen, spe]:
            try:
                val, _ = metric_fn(preds, targets, thres = 0.5)
            except:
                val = np.nan
            info[metric_fn.__name__] = val
            
        res = pd.concat([res, pd.DataFrame([info])], axis = 0).reset_index(drop = True)
            
    return res



def stratify_on_ulcer_size(df_in, train_csv_in, country, best_guids, thres_list, mode = 'pat'):
    
    df = df_in.copy()
    train_csv = train_csv_in.copy()
    
    if country == 'US':
        train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
        name_ = 'US'
    elif country == 'EU':
        train_csv = train_csv[train_csv['USorUK'] == 'UK'].reset_index(drop = True)
        name_ = 'EU'
    else:
        name_ = 'US + EU'
    
    categs = [[] for _ in range(len(thres_list) - 1)]
    tags = [[] for _ in range(len(thres_list) - 1)]
    for idx in range(len(thres_list) - 1):
        if (idx == len(thres_list) - 2):
            lb, ub = thres_list[0], thres_list[idx + 1]
        else:
            lb, ub = thres_list[idx], thres_list[idx + 1]
        categs[idx] = train_csv[(train_csv['cm2_planar_area'] >= lb) & (train_csv['cm2_planar_area'] < ub)]['ICGUID'].tolist()
        tags[idx] = f"{name_} [{lb}, {ub})"
    
    res = pd.DataFrame()
    for ind, cur_guids in enumerate(categs):
        
        cur_df = df[df['ICGUID'].isin(cur_guids)].reset_index(drop = True)
        
        if mode == 'pat':
            num = len(cur_df[['subject_number', 'Visit Number']].drop_duplicates())
            pos = cur_df[['subject_number', 'Visit Number', 'GT']].drop_duplicates()['GT'].sum()
        elif mode == 'img':
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        else:
            cur_df = cur_df[cur_df['ICGUID'].isin(best_guids)].reset_index(drop = True)
            num = len(cur_df)
            pos = cur_df['GT'].sum()
            
        tag = tags[ind]
        tag += f" - N = {num} || {pos} pos"
        
        if mode == 'pat': 
            preds = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
            targets = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['GT'].mean().values)
        else:
            preds = torch.tensor(cur_df['Pred_Proba'].values)
            targets = torch.tensor(cur_df['GT'].values)

        info = {
            'group': tag,
            'acc': None,
            'sen': None,
            'spe': None
        }
        for metric_fn in [acc, sen, spe]:
            try:
                val, _ = metric_fn(preds, targets, thres = 0.5)
            except:
                val = np.nan
            info[metric_fn.__name__] = val
            
        res = pd.concat([res, pd.DataFrame([info])], axis = 0).reset_index(drop = True)
            
    return res



def stratify_on_ulcer_loc(df_in, train_csv_in, country, best_guids, mode = 'pat'):
    
    df = df_in.copy()
    train_csv = train_csv_in.copy()
    
    if country == 'US':
        train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
        name_ = 'US'
    elif country == 'EU':
        train_csv = train_csv[train_csv['USorUK'] == 'UK'].reset_index(drop = True)
        name_ = 'EU'
    else:
        name_ = 'US + EU'
    
    categs = [[] for _ in range(3)]
    # (a) All included
    categs[0] = train_csv['ICGUID'].tolist()
    # (b) Non-toe only
    categs[1] = train_csv[(train_csv['dfu_position'] != 'Toe')
                          ]['ICGUID'].tolist()
    # (c) Toe only
    categs[2] = train_csv[(train_csv['dfu_position'] == 'Toe')
                          ]['ICGUID'].tolist()
    
    res = pd.DataFrame()
    for ind, cur_guids in enumerate(categs):
        
        cur_df = df[df['ICGUID'].isin(cur_guids)].reset_index(drop = True)
        
        if mode == 'pat':
            num = len(cur_df[['subject_number', 'Visit Number']].drop_duplicates())
            pos = cur_df[['subject_number', 'Visit Number', 'GT']].drop_duplicates()['GT'].sum()
        elif mode == 'img':
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        else:
            cur_df = cur_df[cur_df['ICGUID'].isin(best_guids)].reset_index(drop = True)
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        
        if ind == 0: categ = "All included"
        elif ind == 1: categ = f"{name_} non-toe"
        else: categ = f"{name_} toe"
        categ += f" - N = {num} || {pos} pos"
        
        if mode == 'pat':
            preds = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
            targets = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['GT'].mean().values)
        else:
            preds = torch.tensor(cur_df['Pred_Proba'].values)
            targets = torch.tensor(cur_df['GT'].values)

        info = {
            'group': categ,
            'acc': None,
            'sen': None,
            'spe': None
        }
        for metric_fn in [acc, sen, spe]:
            try:
                val, _ = metric_fn(preds, targets, thres = 0.5)
            except:
                val = np.nan
            info[metric_fn.__name__] = val
            
        res = pd.concat([res, pd.DataFrame([info])], axis = 0).reset_index(drop = True)
            
    return res



if __name__ == '__main__':
    
    # Path to model predictions
    unified_csv = sys.argv[1]
    prediction_path = sys.argv[2]
    hs = int(sys.argv[3])
    use_bsv_only = sys.argv[4]
    early_val = sys.argv[5]
    
    out_excel_path = f"/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/out/stratified_analysis"
        
    # Load unified CSV used in training
    train_csv = pd.read_csv(unified_csv)
    if use_bsv_only.strip().lower() == 'true': # Keep only SV1 rows
        train_csv = train_csv[train_csv['Visit Number'] == 'DFU_SV1'].reset_index(drop = True) 
    train_csv = train_csv[train_csv['good_ori'] == 'Y'].reset_index(drop = True) # Keep only rows with good orientation
    train_csv = train_csv[~train_csv['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True) # Exclude cases that are not usable
    
    if early_val.strip().lower() == 'true':
        df_pred = pd.read_csv(f"{prediction_path}/hs_{hs}/predictions_test.csv") # early validation
        ds = 'test'
    else:
        df_pred = pd.read_csv(f"{prediction_path}/hs_{hs}/predictions_cv.csv") # cross validation
        ds = 'cv'
    
    # df_pred = df_pred[df_pred['Visit Number'] == 'DFU_SV1'].reset_index(drop = True)
    
    # Gather ImgCollGUIDs that yield best orientation for each subject
    train_csv['orientation_deg'].fillna(float('inf'), inplace = True)
    best_idx = train_csv.groupby(['subject_number', 'Visit Number'])['orientation_deg'].idxmin().values
    best_guids = train_csv.iloc[best_idx]['ICGUID'].values.tolist()
    
    # Stratified based on country
    strat_country_pat = stratify_on_country(df_pred, train_csv, best_guids, mode = 'pat')
    strat_country_img = stratify_on_country(df_pred, train_csv, best_guids, mode = 'img')
    strat_country_best = stratify_on_country(df_pred, train_csv, best_guids, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_country_pat.index, columns = [''])
    out_df_1 = pd.concat([strat_country_pat, blank_df.copy(), strat_country_img, blank_df.copy(), strat_country_best], axis = 1)
    
    # Stratified based on ulcer size
    # train_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404/src/data/WAUSI_unifiedv3_BSVp1-7_final1_20240411.csv")
    # train_csv = train_csv[train_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # train_csv = train_csv[~train_csv['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True)
    # train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
    # for q in np.linspace(0, 1, 6):
    #     thres = train_csv[['subject_number', 'cm2_planar_area']].drop_duplicates()['cm2_planar_area'].quantile(q)
    #     print(thres)
    # thres_list = [0, 0.52, 1.31, 2.61, 5.76, 65.81, float('inf')]
    thres_list = [0, 1, 1000, float('inf')]
    
    strat_area_both_pat = stratify_on_ulcer_size(df_pred, train_csv, '', best_guids, thres_list, mode = 'pat')
    strat_area_both_img = stratify_on_ulcer_size(df_pred, train_csv, '', best_guids, thres_list, mode = 'img')
    strat_area_both_best = stratify_on_ulcer_size(df_pred, train_csv, '', best_guids, thres_list, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_area_both_pat.index, columns = [''])
    out_df_2a = pd.concat([strat_area_both_pat, blank_df.copy(), strat_area_both_img, blank_df.copy(), strat_area_both_best], axis = 1)

    strat_area_us_pat = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, thres_list, mode = 'pat')
    strat_area_us_img = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, thres_list, mode = 'img')
    strat_area_us_best = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, thres_list, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_area_us_pat.index, columns = [''])
    out_df_2b = pd.concat([strat_area_us_pat, blank_df.copy(), strat_area_us_img, blank_df.copy(), strat_area_us_best], axis = 1)
    
    strat_area_eu_pat = stratify_on_ulcer_size(df_pred, train_csv, 'EU', best_guids, thres_list, mode = 'pat')
    strat_area_eu_img = stratify_on_ulcer_size(df_pred, train_csv, 'EU', best_guids, thres_list, mode = 'img')
    strat_area_eu_best = stratify_on_ulcer_size(df_pred, train_csv, 'EU', best_guids, thres_list, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_area_eu_pat.index, columns = [''])
    out_df_2c = pd.concat([strat_area_eu_pat, blank_df.copy(), strat_area_eu_img, blank_df.copy(), strat_area_eu_best], axis = 1)
    
    # Stratified based on ulcer position 
    strat_loc_both_pat = stratify_on_ulcer_loc(df_pred, train_csv, '', best_guids, mode = 'pat')
    strat_loc_both_img = stratify_on_ulcer_loc(df_pred, train_csv, '', best_guids, mode = 'img')
    strat_loc_both_best = stratify_on_ulcer_loc(df_pred, train_csv, '', best_guids, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_loc_both_pat.index, columns = [''])
    out_df_3a = pd.concat([strat_loc_both_pat, blank_df.copy(), strat_loc_both_img, blank_df.copy(), strat_loc_both_best], axis = 1)
    
    strat_loc_us_pat = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, mode = 'pat')
    strat_loc_us_img = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, mode = 'img')
    strat_loc_us_best = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_loc_us_pat.index, columns = [''])
    out_df_3b = pd.concat([strat_loc_us_pat, blank_df.copy(), strat_loc_us_img, blank_df.copy(), strat_loc_us_best], axis = 1)
    
    strat_loc_eu_pat = stratify_on_ulcer_loc(df_pred, train_csv, 'EU', best_guids, mode = 'pat')
    strat_loc_eu_img = stratify_on_ulcer_loc(df_pred, train_csv, 'EU', best_guids, mode = 'img')
    strat_loc_eu_best = stratify_on_ulcer_loc(df_pred, train_csv, 'EU', best_guids, mode = 'best')
    blank_df = pd.DataFrame(np.nan, index = strat_loc_eu_pat.index, columns = [''])
    out_df_3c = pd.concat([strat_loc_eu_pat, blank_df.copy(), strat_loc_eu_img, blank_df.copy(), strat_loc_eu_best], axis = 1)
    
    blank_row = pd.DataFrame(np.nan, index=[0, 1], columns = out_df_1.columns)    
    out_df = pd.concat([out_df_1, blank_row.copy(), out_df_2a, blank_row.copy(), out_df_2b, blank_row.copy(), out_df_2c, blank_row.copy(), out_df_3a, blank_row.copy(), out_df_3b, blank_row.copy(), out_df_3c], ignore_index = True).round(3)

    with pd.ExcelWriter(f"{out_excel_path}/{'_'.join(prediction_path.split('/')[-2:])}_hs_{hs}_{ds}.xlsx", engine = 'openpyxl') as writer:
        
        out_df.to_excel(writer, index = False, sheet_name = 'Sheet1')
        
        # Access the openpyxl workbook and sheet objects
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        for i, column in enumerate(out_df.columns, start=1):
            column_letter = get_column_letter(i)
            if column == 'group':
                worksheet.column_dimensions[column_letter].width = 30
            elif column in ['acc', 'sen', 'spe']:
                worksheet.column_dimensions[column_letter].width = 10
            else:
                pass