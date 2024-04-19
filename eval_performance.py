"""
(1) Find optimal epoch that yields the best performance across all validation folds 
(2) Export predictions on training and test set

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import os
import pandas as pd
import numpy as np
import torch
from utils.clf_metrics import acc, sen, spe, harmonic



class Config:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)



def get_best_epoch(config, results_path, strategy = 'acc'):
    
    stats = []
    for fold in range(1, config.K_FOLD + 1):
        val_records = pd.read_csv(f"{results_path}/predictions/train_val_curve/df_val_records{fold}.csv")
        stats.append(val_records.values)
    
    res_df = pd.DataFrame(np.mean(stats, axis = 0),
                          index = range(0, config.EPOCH),
                          columns = "mean_" + val_records.columns)
    
    row = res_df.sort_values(by = f"mean_{strategy}", ascending = False).iloc[[0]]
    best_epoch = row.index.values[0]
        
    return best_epoch



def evaluate(config, results_path, best_epoch, mode = 'cv'):
    
    if mode == 'cv':
        df = pd.DataFrame()
        for fold in range(1, config.K_FOLD + 1):
            cur_pred = pd.read_csv(f"{results_path}/predictions/fold_{fold}/pred_epoch_{best_epoch}.csv")
            df = pd.concat([df, cur_pred], axis = 0).reset_index(drop = True)
    else:
        df = pd.read_csv(f"{results_path}/predictions/fold_99/pred_epoch_{best_epoch}.csv")
        
    # Evaluation on a per-patient basis
    preds_pat = torch.tensor(df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
    targets_pat = torch.tensor(df.groupby(['subject_number', 'Visit Number'])["GT"].mean().values)
    assert(all(targets_pat.unique().numpy() == [0., 1.]))
    
    # Evaluation on a per-image basis
    preds_img, targets_img = torch.tensor(df['Pred_Proba'].values), torch.tensor(df['GT'].values)
    
    performance_pat, performance_img = {}, {}
    for metric_fn in [acc, sen, spe, harmonic]:
        
        val, _ = metric_fn(preds_pat, targets_pat, thres = 0.5)
        performance_pat[metric_fn.__name__] = val
        
        val, _ = metric_fn(preds_img, targets_img, thres = 0.5)
        performance_img[metric_fn.__name__] = val
        
    return df, performance_pat, performance_img



if __name__ == "__main__":
    
    strategy = 'harmonic' # Choose from ['acc', 'sen', 'spe', 'harmonic]
    
    cols = ['hs', 'best_epoch', 'pat_acc', 'pat_sen', 'pat_spe', 'pat_harmonic', 'img_acc', 'img_sen', 'img_spe', 'img_harmonic']
    cv_performance = pd.DataFrame(columns = cols)
    
    start, end = 571, 571
    for exp_id in range(start, end + 1):
        
        print(f"Progress: {exp_id} / {end} ...", end = "\r")

        results_path = f"/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/baseline/hs_{exp_id}"
        
        if not os.path.exists(f"{results_path}/model/config_CV.npy"): continue
        
        config = np.load(f"{results_path}/model/config_CV.npy", allow_pickle = True).item()
        config = Config(config)
        
        best_epoch = get_best_epoch(config, results_path, strategy)
        
        info = {
            "hs": f"hs_{exp_id}",
            "best_epoch": best_epoch
        }
        
        df_cv, performance_pat_cv, performance_img_cv = evaluate(config, results_path, best_epoch, mode = 'cv')
        df_cv.to_csv(f"{results_path}/predictions_cv.csv", index = False)
        
        entry = pd.concat([pd.DataFrame([info]), pd.DataFrame([performance_pat_cv]), pd.DataFrame([performance_img_cv])], axis = 1, ignore_index = True)
        entry.columns = cols
        cv_performance = pd.concat([cv_performance, entry], axis = 0).reset_index(drop = True)
        
        if os.path.exists(f"{results_path}/model/config_final_train.npy"):
            df_test, performance_pat_test, performance_img_test = evaluate(config, results_path, best_epoch, mode = 'test')
            df_test.to_csv(f"{results_path}/predictions_test.csv", index = False)

    sorted_df = cv_performance.sort_values(by = 'pat_harmonic', ascending = False)