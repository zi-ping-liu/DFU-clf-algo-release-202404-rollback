"""
Encode and imputate clinical features used in classification model training

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder



class PreProcessing():
    
    def __init__(self, config, df_train_in, df_val_in):
        
        self.config = config
        
        df_train = df_train_in.copy()
        df_val = df_val_in.copy()
        
        # Rename "subject_number" column since one patient can have multiple visits where the numerical features could be different
        # In such a scenario, each visit should be treated as a "different patient"
        df_train['subject_number'] = df_train[['subject_number', 'Visit Number']].apply(self.rename_subjectID, axis = 1)
        df_val['subject_number'] = df_val[['subject_number', 'Visit Number']].apply(self.rename_subjectID, axis = 1)
        
        self.train_pats = df_train[['ICGUID', 'subject_number']]
        self.val_pats = df_val[['ICGUID', 'subject_number']]
        
        # Keep only the feature columns and drop duplicate rows (due to multiple images per patient)
        cols_to_drop = ['ICGUID', 'Site', 'DE_phase', 'DS_phase', 'DS_split', 'good_ori', 'GT', 'Wound Location',
                        'USorUK', 'Visit Number', 'orientation_deg']
        self.df_train = df_train.drop(cols_to_drop, axis = 1).drop_duplicates().reset_index(drop = True)
        self.df_val = df_val.drop(cols_to_drop, axis = 1).drop_duplicates().reset_index(drop = True)
        
        # Categorical (binary) features 1: Missing values in these features are imputed with 0
        self.CAT_BIN_1 = [
            'smoking', 'anemia', 'dialysis', 'debridement_performed', 'additional_DFUs', 'skeletal_deformity', 'skeletal_study', 'diabetic_neuropathy', 
            'limb_loss_history', 'limb_loss_study_foot_binary', 'angio_stent_bypass', 'infection_binary', 
            'ckd', 'cirrhosis', 'DVT', 'heart_attack', 'stroke'
        ]
        
        # Categorical (binary) features 2: Missing values in these features are imputed based on mode value
        self.CAT_BIN_2 = [
            'sex'
        ]
        
        # Categorical features - one-hot encoded variables
        self.CAT_ONEHOT = {
            'race': ['white', 'black', "asian", 'other'],
            'dfu_position': ['Forefoot', 'Hindfoot', 'Toe', 'Midfoot']
        }
        
        # Ordinal features
        self.ORDINAL = {
            'tendon_bone': {
                'Not Applicable': 0,
                'Tendon or capsule': 1,
                'Bone or joint': 2
            },
            'exu_type': {
                'Not applicable - no exudate present': 0,
                'Serous: clear or light yellow, watery plasma': 1,
                'Serosanguinous: pink to light red watery plasma': 2,
                'Sanguinous: red with fresh bleeding': 3,
                'Purulent: thick and opaque exudate, thick yellow, green, white or tan color': 4
            },
            'exu_volume': {
                'Minimal amount': 1,
                'Light or small amount': 2,
                'Moderate amount': 3,
                'Heavy/large/copious amounts': 4
            }
        }
        
        # Numerical features 1: Missing values in these features are filled with K-NN imputer
        self.NUMERICAL_1 = [
            'age', 'dialysis_years', 'temp', 'heart_rate', 'ulcer_duration',
            'cm3_volume', 'cm2_surf_area', 'cm2_planar_area', 'cm_bbox_x', 'cm_bbox_y', 'cm_bbox_z'
        ]
        
        # Numerical features 2: Missing values in these features are filled also with K-NN imputer but after the above features
        # Two new columns are appended: "BMI", "mean_arterial_pressure"
        self.NUMERICAL_2 = [
            'height_inches', 'weight_pounds', 'systolic_blood_pressure', 'diastolic_blood_pressure'
        ]
        
        # Numerical features 3: Missing values in these features are filled with specific strategies
        self.NUMERICAL_3 = [
            'first_dfu_age'
        ]
        
    
    def preprocess_df(self):
        
        ##########################################################################################
        ### Process categorical (binary) features 1
        X_train_cat_1 = self.df_train[self.CAT_BIN_1].copy()
        X_val_cat_1 = self.df_val[self.CAT_BIN_1].copy()
        
        # Step 1. Replace yes/no with 1/0
        for col in X_train_cat_1.columns:
            X_train_cat_1.loc[:, col] = X_train_cat_1.loc[:, col].apply(self.replace_One_Zero)
            X_val_cat_1.loc[:, col] = X_val_cat_1.loc[:, col].apply(self.replace_One_Zero)
        
        # Step 2. Replace missing values with 0
        X_train_cat_1 = X_train_cat_1.fillna(0)
        X_val_cat_1 = X_val_cat_1.fillna(0)
        
        
        ##########################################################################################
        ### Process categorical (binary) features 2
        X_train_cat_2 = self.df_train[self.CAT_BIN_2].copy()
        X_val_cat_2 = self.df_val[self.CAT_BIN_2].copy()
        
        for col in X_train_cat_2.columns:
            
            # Step 1. Replace yes/no (male/female) with 1/0    
            X_train_cat_2.loc[:, col] = X_train_cat_2.loc[:, col].apply(self.replace_One_Zero)
            X_val_cat_2.loc[:, col] = X_val_cat_2.loc[:, col].apply(self.replace_One_Zero)
        
            # Step 2. Replace missing values with mode
            mode = X_train_cat_2[col].mode()[0]
            if (np.isnan(mode)):
                raise Exception(f"ERROR: mode value of feature '{col}' is NaN")
            X_train_cat_2.loc[:, col] = X_train_cat_2.loc[:, col].fillna(mode)
            X_val_cat_2.loc[:, col] = X_val_cat_2.loc[:, col].fillna(mode)
        
        
        ##########################################################################################
        ### Process categorical (one-hot encoded) features
        X_train_oh = self.df_train[list(self.CAT_ONEHOT.keys())].copy()
        X_val_oh = self.df_val[list(self.CAT_ONEHOT.keys())].copy()
        
        # Step 1. Impute each feature if necessary
        for col in X_train_oh.columns:
            
            mode = X_train_oh[col].mode()[0]
            if (str(mode).strip().lower() == 'nan'):
                raise Exception(f"ERROR: mode value of feature '{col}' is NaN")
            
            X_train_oh.loc[:, col] = X_train_oh.loc[:, col].fillna(mode)
            X_val_oh.loc[:, col] = X_val_oh.loc[:, col].fillna(mode)
        
        # Step 2. One-hot encode
        categories = [list(value) for value in self.CAT_ONEHOT.values()]
        encoder = OneHotEncoder(categories = categories)
        
        X_train_oh = pd.DataFrame(encoder.fit_transform(X_train_oh).toarray(), 
                                   columns = encoder.get_feature_names_out())
        X_val_oh = pd.DataFrame(encoder.transform(X_val_oh).toarray(), 
                                 columns = encoder.get_feature_names_out())
        
        
        ##########################################################################################
        ### Process ordinal features
        X_train_ord = self.df_train[list(self.ORDINAL.keys())].copy()
        X_val_ord = self.df_val[list(self.ORDINAL.keys())].copy()
        
        # Step 1. Ordinal encode
        for col, levels in self.ORDINAL.items():
            X_train_ord.loc[:, col] = X_train_ord.loc[:, col].map(levels)
            X_val_ord.loc[:, col] = X_val_ord.loc[:, col].map(levels)
            
        # Step 2. Replace missing values with 0
        X_train_ord = X_train_ord.fillna(0)
        X_val_ord = X_val_ord.fillna(0)
        
        
        ##########################################################################################
        ### Concatenate all categorical/ordinal features
        X_train_categorical = pd.concat([X_train_cat_1, X_train_cat_2, X_train_oh, X_train_ord], axis = 1).copy()
        X_val_categorical = pd.concat([X_val_cat_1, X_val_cat_2, X_val_oh, X_val_ord], axis = 1).copy()
        
        
        ##########################################################################################
        ### Process numerical features 1
        X_train_num_1 = self.df_train[self.NUMERICAL_1].copy()
        X_val_num_1 = self.df_val[self.NUMERICAL_1].copy()

        # Step 1. Normalize
        norm = self.config.NUMERICAL_PREPROCESS['NORMALIZATION']
        if (norm == 'min_max'):
            scaler = MinMaxScaler()
        elif (norm == 'standard'):
            scaler = StandardScaler()
        else:
            raise Exception("ERROR: invalid normalization specified for numerical features. Please select from ['min_max', 'standard']")
        scaler.fit(X_train_num_1)
        X_train_num_1 = pd.DataFrame(scaler.transform(X_train_num_1), columns = self.NUMERICAL_1)
        X_val_num_1 = pd.DataFrame(scaler.transform(X_val_num_1), columns = self.NUMERICAL_1)
        
        # Step 2. Concatenate with categorical features
        X_train_all = pd.concat([X_train_categorical, X_train_num_1], axis = 1)
        X_val_all = pd.concat([X_val_categorical, X_val_num_1], axis = 1)
        
        # Step 3. K-NN impute
        imputer = KNNImputer(n_neighbors = self.config.NUMERICAL_PREPROCESS['NUM_NEAREST_NEIGHBOURS'],
                             weights = 'distance')
        X_train_all = pd.DataFrame(imputer.fit_transform(X_train_all), columns = X_train_all.columns)
        X_val_all = pd.DataFrame(imputer.transform(X_val_all), columns = X_train_all.columns)
        
        # Step 4. Retreive processed numerical features 1
        X_train_num_1_final = X_train_all[self.NUMERICAL_1].copy()
        X_val_num_1_final = X_val_all[self.NUMERICAL_1].copy()
        
        
        ##########################################################################################
        ### Process numerical features 2
        X_train_num_2 = self.df_train[self.NUMERICAL_2].copy()
        X_val_num_2 = self.df_val[self.NUMERICAL_2].copy()
        
        # Step 1. Concatenate with X_train_all/X_val_all
        X_train_all = pd.concat([X_train_all, X_train_num_2], axis = 1)
        X_val_all = pd.concat([X_val_all, X_val_num_2], axis = 1)
        
        # Step 2. K-NN impute
        imputer = KNNImputer(n_neighbors = self.config.NUMERICAL_PREPROCESS['NUM_NEAREST_NEIGHBOURS'],
                             weights = 'distance')
        X_train_all = pd.DataFrame(imputer.fit_transform(X_train_all), columns = X_train_all.columns)
        X_val_all = pd.DataFrame(imputer.transform(X_val_all), columns = X_train_all.columns)
        
        # Step 3. Impute (1) BMI; (2) mean_arterial_pressure and retrieve processed numerical features 2
        X_train_num_2_final = X_train_all[self.NUMERICAL_2].copy()
        X_val_num_2_final = X_val_all[self.NUMERICAL_2].copy()
        
        tmp = X_train_num_2_final[['height_inches', 'weight_pounds']].apply(self.get_BMI, axis = 1)
        X_train_num_2_final = pd.concat([X_train_num_2_final, tmp.rename('BMI')], axis = 1)
        
        tmp = X_val_num_2_final[['height_inches', 'weight_pounds']].apply(self.get_BMI, axis = 1)
        X_val_num_2_final = pd.concat([X_val_num_2_final, tmp.rename('BMI')], axis = 1)
        
        tmp = X_train_num_2_final[['systolic_blood_pressure', 'diastolic_blood_pressure']].apply(self.get_mean_arterial_pressure, axis = 1)
        X_train_num_2_final = pd.concat([X_train_num_2_final, tmp.rename('mean_arterial_pressure')], axis = 1)     
        
        tmp = X_val_num_2_final[['systolic_blood_pressure', 'diastolic_blood_pressure']].apply(self.get_mean_arterial_pressure, axis = 1)
        X_val_num_2_final = pd.concat([X_val_num_2_final, tmp.rename('mean_arterial_pressure')], axis = 1)
        
        # Step 4. Normalize
        norm = self.config.NUMERICAL_PREPROCESS['NORMALIZATION']
        if (norm == 'min_max'):
            scaler = MinMaxScaler()
        elif (norm == 'standard'):
            scaler = StandardScaler()
        else:
            raise Exception("ERROR: invalid normalization specified for numerical features. Please select from ['min_max', 'standard']")
        scaler.fit(X_train_num_2_final)
        X_train_num_2_final = pd.DataFrame(scaler.transform(X_train_num_2_final), columns = X_train_num_2_final.columns)
        X_val_num_2_final = pd.DataFrame(scaler.transform(X_val_num_2_final), columns = X_train_num_2_final.columns)
        
        
        ##########################################################################################
        ### Process numerical features 3
        X_train_num_3_final = pd.DataFrame()
        X_val_num_3_final = pd.DataFrame()
        
        for col in self.NUMERICAL_3:
            
            if (col == 'first_dfu_age'):
                
                X_train_col = pd.DataFrame(self.df_train[['first_dfu_age', 'age']].apply(self.impute_first_dfu_age, axis = 1))
                X_val_col = pd.DataFrame(self.df_val[['first_dfu_age', 'age']].apply(self.impute_first_dfu_age, axis = 1))
                
                norm = self.config.NUMERICAL_PREPROCESS['NORMALIZATION']
                if (norm == 'min_max'):
                    scaler = MinMaxScaler()
                elif (norm == 'standard'):
                    scaler = StandardScaler()
                else:
                    raise Exception("ERROR: invalid normalization specified for numerical features. Please select from ['min_max', 'standard']")
                scaler.fit(X_train_col)
                X_train_col = pd.DataFrame(scaler.transform(X_train_col), columns = ['first_dfu_age'])
                X_val_col = pd.DataFrame(scaler.transform(X_val_col), columns = ['first_dfu_age'])
                
                X_train_num_3_final = pd.concat([X_train_num_3_final, X_train_col], axis = 1)
                X_val_num_3_final = pd.concat([X_val_num_3_final, X_val_col], axis = 1)
        
        ##########################################################################################
        ### Concatenate categorical and numerical features
        X_train_out = pd.concat([X_train_categorical, X_train_num_1_final, X_train_num_2_final, X_train_num_3_final], axis = 1)
        X_val_out = pd.concat([X_val_categorical, X_val_num_1_final, X_val_num_2_final, X_val_num_3_final], axis = 1)
        
        X_train_out = pd.merge(self.train_pats, pd.concat([self.df_train[['subject_number']], X_train_out], axis = 1), on = 'subject_number')
        X_val_out = pd.merge(self.val_pats, pd.concat([self.df_val[['subject_number']], X_val_out], axis = 1), on = 'subject_number')
        
        X_train_out.drop(['subject_number', 'ICGUID'], axis = 1, inplace = True)
        X_val_out.drop(['subject_number', 'ICGUID'], axis = 1, inplace = True)
        
        return X_train_out, X_val_out
    
    
    
    def rename_subjectID(self, x):
        """
        Helper function: Append visit number to subjectID (e.g., 201-001 -> 201-001_DFU_SV1)

        Args:
        --------
        x (pd.Series): 
            x[0] -> "subject_number"
            x[1] -> "Visit Number"
        
        Returns:
        --------
        Renamed "subject_number" column
        """
        
        return x[0] + '_' + x[1]
    
    
    
    def replace_One_Zero(self, val):
        """
        Helper function: Replace the value of binary variable with 1/0
        
        Args:
        --------
        val:
            Yes/No/NaN
        
        Returns:
        --------
        1/0
        """
        
        val_cp = str(val).strip().lower()
        
        if (val_cp == 'yes') or (val_cp == 'male'):
            return 1
        elif (val_cp == 'no') or (val_cp == 'female'):
            return 0
        elif (val_cp == 'nan'): # Not deal with missing values at this point
            return np.nan
        else:
            raise Exception(f"ERROR: unknown value found: {val} when processing binary variables")
    
    
    
    def get_BMI(self, x):
        """
        Helper function: Calculate patient body mass index (BMI) based on height and weight

        Args:
        --------
        x (pd.Series): 
            Row containing the "height" and "weight" columns
            x[0] -> height in inches
            x[1] -> weight in pounds

        Returns:
        --------
        bmi (float):
            Calculated BMI value
        """
        
        height = x[0] * 0.0254 # inch to meter
        weight = x[1] * 0.453592 # pound to kg
        
        bmi = weight / (height ** 2)
        
        return bmi
    
    
    
    def get_mean_arterial_pressure(self, x):
        """
        Helper function: Calculate mean arterial pressure based on systolic/diastolic blood pressure

        Args:
        --------
        x (pd.Series): 
            x[0] -> "systolic_blood_pressure"
            x[1] -> "diastolic_blood_pressure"    

        Returns:
        --------
        Mean artirial pressure
        """
        
        # mean arterial pressure = (SBP + (2 * DBP)) * 1/3
        return (x[0] + 2 * x[1]) * (1/3)
    
    
    
    def impute_first_dfu_age(self, x):
        """
        Helper function: Impute missing values in "first_dfu_age" feature

        Args:
        --------
        x (pd.Series):
            x[0] -> "first_dfu_age"
            x[1] -> "age"

        Returns:
        --------
        Imputed entries if applicable
        """
        
        if str(x[0]) == 'nan':
            return x[1] # if "first_dfu_age" is NaN, impute with patient's current age
        else:
            return x[0]