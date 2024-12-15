# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.over_sampling import SMOTE

# # from secret import access_key, secret_access_key
# import joblib
# import streamlit as st
# import boto3
# import tempfile
# import json
# import requests
# from streamlit_lottie import st_lottie_spinner
# import logging
# from botocore.exceptions import ClientError


# train_original = pd.read_csv(
#     "https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/refs/heads/main/dataset/train.csv"
# )

# test_original = pd.read_csv(
#     "https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/refs/heads/main/dataset/test.csv"
# )

# full_data = pd.concat([train_original, test_original], axis=0)

# full_data = full_data.sample(frac=1).reset_index(drop=True)


# def data_split(df, test_size):
#     train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
#     return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# train_original, test_original = data_split(full_data, 0.2)

# train_copy = train_original.copy()
# test_copy = test_original.copy()


# def value_cnt_norm_cal(df, feature):
#     """
#     Function to calculate the count of each value in a feature and normalize it
#     """
#     ftr_value_cnt = df[feature].value_counts()
#     ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
#     ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
#     ftr_value_cnt_concat.columns = ["Count", "Frequency (%)"]
#     return ftr_value_cnt_concat


# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(
#         self, feat_with_outliers=["Family member count", "Income", "Employment length"]
#     ):
#         self.feat_with_outliers = feat_with_outliers

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.feat_with_outliers).issubset(df.columns):
#             # 25% quantile
#             Q1 = df[self.feat_with_outliers].quantile(0.25)
#             # 75% quantile
#             Q3 = df[self.feat_with_outliers].quantile(0.75)
#             IQR = Q3 - Q1
#             # keep the data within 1.5 IQR
#             df = df[
#                 ~(
#                     (df[self.feat_with_outliers] < (Q1 - 3 * IQR))
#                     | (df[self.feat_with_outliers] > (Q3 + 3 * IQR))
#                 ).any(axis=1)
#             ]
#             return df
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class DropFeatures(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         feature_to_drop=[
#             "Has a mobile phone",
#             "Children count",
#             "Job title",
#             "Account age",
#         ],
#     ):
#         self.feature_to_drop = feature_to_drop

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.feature_to_drop).issubset(df.columns):
#             df.drop(self.feature_to_drop, axis=1, inplace=True)
#             return df
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class TimeConversionHandler(BaseEstimator, TransformerMixin):
#     def __init__(self, feat_with_days=["Employment length", "Age"]):
#         self.feat_with_days = feat_with_days

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         if set(self.feat_with_days).issubset(X.columns):
#             # convert days to absolute value
#             X[["Employment length", "Age"]] = np.abs(X[["Employment length", "Age"]])
#             return X
#         else:
#             print("One or more features are not in the dataframe")
#             return X


# class RetireeHandler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if "Employment length" in df.columns:
#             # select rows with employment length is 365243 which corresponds to retirees
#             df_ret_idx = df["Employment length"][
#                 df["Employment length"] == 365243
#             ].index
#             # change 365243 to 0
#             df.loc[df_ret_idx, "Employment length"] = 0
#             return df
#         else:
#             print("Employment length is not in the dataframe")
#             return df


# class SkewnessHandler(BaseEstimator, TransformerMixin):
#     def __init__(self, feat_with_skewness=["Income", "Age"]):
#         self.feat_with_skewness = feat_with_skewness

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.feat_with_skewness).issubset(df.columns):
#             # Handle skewness with cubic root transformation
#             df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
#             return df
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class BinningNumToYN(BaseEstimator, TransformerMixin):
#     def __init__(
#         self, feat_with_num_enc=["Has a work phone", "Has a phone", "Has an email"]
#     ):
#         self.feat_with_num_enc = feat_with_num_enc

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.feat_with_num_enc).issubset(df.columns):
#             # Change 0 to N and 1 to Y for all the features in feat_with_num_enc
#             for ft in self.feat_with_num_enc:
#                 df[ft] = df[ft].map({1: "Y", 0: "N"})
#             return df
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         one_hot_enc_ft=[
#             "Gender",
#             "Marital status",
#             "Dwelling",
#             "Employment status",
#             "Has a car",
#             "Has a property",
#             "Has a work phone",
#             "Has a phone",
#             "Has an email",
#         ],
#     ):
#         self.one_hot_enc_ft = one_hot_enc_ft

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.one_hot_enc_ft).issubset(df.columns):
#             # function to one hot encode the features in one_hot_enc_ft
#             def one_hot_enc(df, one_hot_enc_ft):
#                 one_hot_enc = OneHotEncoder()
#                 one_hot_enc.fit(df[one_hot_enc_ft])
#                 # get the result of the one hot encoding columns names
#                 feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(
#                     one_hot_enc_ft
#                 )
#                 # change the array of the one hot encoding to a dataframe with the column names
#                 df = pd.DataFrame(
#                     one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),
#                     columns=feat_names_one_hot_enc,
#                     index=df.index,
#                 )
#                 return df

#             # function to concatenat the one hot encoded features with the rest of features that were not encoded
#             def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
#                 # get the rest of the features
#                 rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
#                 # concatenate the rest of the features with the one hot encoded features
#                 df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
#                 return df_concat

#             # one hot encoded dataframe
#             one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
#             # returns the concatenated dataframe
#             full_df_one_hot_enc = concat_with_rest(
#                 df, one_hot_enc_df, self.one_hot_enc_ft
#             )
#             print(full_df_one_hot_enc.tail(25))
#             return full_df_one_hot_enc
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class OrdinalFeatNames(BaseEstimator, TransformerMixin):
#     def __init__(self, ordinal_enc_ft=["Education level"]):
#         self.ordinal_enc_ft = ordinal_enc_ft

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if "Education level" in df.columns:
#             ordinal_enc = OrdinalEncoder()
#             df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
#             return df
#         else:
#             print("Education level is not in the dataframe")
#             return df


# class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
#     def __init__(self, min_max_scaler_ft=["Age", "Income", "Employment length"]):
#         self.min_max_scaler_ft = min_max_scaler_ft

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if set(self.min_max_scaler_ft).issubset(df.columns):
#             min_max_enc = MinMaxScaler()
#             df[self.min_max_scaler_ft] = min_max_enc.fit_transform(
#                 df[self.min_max_scaler_ft]
#             )
#             return df
#         else:
#             print("One or more features are not in the dataframe")
#             return df


# class ChangeToNumTarget(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if "Is high risk" in df.columns:
#             df["Is high risk"] = pd.to_numeric(df["Is high risk"])
#             return df
#         else:
#             print("Is high risk is not in the dataframe")
#             return df


# class OversampleSMOTE(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         if "Is high risk" in df.columns:
#             # SMOTE function to oversample the minority class to fix the imbalance data
#             smote = SMOTE()
#             X_bal, y_bal = smote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])
#             X_y_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
#             return X_y_bal
#         else:
#             print("Is high risk is not in the dataframe")
#             return df


# def full_pipeline(df):
#     # Create the pipeline that will call all the class from OutlierRemoval to OversampleSMOTE in one go
#     pipeline = Pipeline(
#         [
#             ("outlier_remover", OutlierRemover()),
#             ("feature_dropper", DropFeatures()),
#             ("time_conversion_handler", TimeConversionHandler()),
#             ("retiree_handler", RetireeHandler()),
#             ("skewness_handler", SkewnessHandler()),
#             ("binning_num_to_yn", BinningNumToYN()),
#             ("one_hot_with_feat_names", OneHotWithFeatNames()),
#             ("ordinal_feat_names", OrdinalFeatNames()),
#             ("min_max_with_feat_names", MinMaxWithFeatNames()),
#             ("change_to_num_target", ChangeToNumTarget()),
#             ("oversample_smote", OversampleSMOTE()),
#         ]
#     )
#     df_pipe_prep = pipeline.fit_transform(df)
#     return df_pipe_prep


# ############################# Streamlit ############################

# st.write("""
# # Credit card approval prediction
# This app predicts if an applicant will be approved for a credit card or not. Just fill in the following information and click on the Predict button.
# """)

# # Gender input
# st.write("""
# ## Gender
# """)
# input_gender = st.radio("Select you gender", ["Male", "Female"], index=0)


# # Age input slider
# st.write("""
# ## Age
# """)
# input_age = np.negative(
#     st.slider("Select your age", value=42, min_value=18, max_value=70, step=1) * 365.25
# )


# # Marital status input dropdown
# st.write("""
# ## Marital status
# """)
# marital_status_values = list(value_cnt_norm_cal(full_data, "Marital status").index)
# marital_status_key = [
#     "Married",
#     "Single/not married",
#     "Civil marriage",
#     "Separated",
#     "Widowed",
# ]
# marital_status_dict = dict(zip(marital_status_key, marital_status_values))
# input_marital_status_key = st.selectbox(
#     "Select your marital status", marital_status_key
# )
# input_marital_status_val = marital_status_dict.get(input_marital_status_key)


# # Family member count
# st.write("""
# ## Family member count
# """)
# fam_member_count = float(
#     st.selectbox("Select your family member count", [1, 2, 3, 4, 5, 6])
# )


# # Dwelling type dropdown
# st.write("""
# ## Dwelling type
# """)
# dwelling_type_values = list(value_cnt_norm_cal(full_data, "Dwelling").index)
# dwelling_type_key = [
#     "House / apartment",
#     "Live with parents",
#     "Municipal apartment ",
#     "Rented apartment",
#     "Office apartment",
#     "Co-op apartment",
# ]
# dwelling_type_dict = dict(zip(dwelling_type_key, dwelling_type_values))
# input_dwelling_type_key = st.selectbox(
#     "Select the type of dwelling you reside in", dwelling_type_key
# )
# input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)


# # Income
# st.write("""
# ## Income
# """)
# input_income = int(st.text_input("Enter your income (in USD)", 0))

# # Employment status dropdown
# st.write("""
# ## Employment status
# """)
# employment_status_values = list(
#     value_cnt_norm_cal(full_data, "Employment status").index
# )
# employment_status_key = [
#     "Working",
#     "Commercial associate",
#     "Pensioner",
#     "State servant",
#     "Student",
# ]
# employment_status_dict = dict(zip(employment_status_key, employment_status_values))
# input_employment_status_key = st.selectbox(
#     "Select your employment status", employment_status_key
# )
# input_employment_status_val = employment_status_dict.get(input_employment_status_key)


# # Employment length input slider
# st.write("""
# ## Employment length
# """)
# input_employment_length = np.negative(
#     st.slider(
#         "Select your employment length", value=6, min_value=0, max_value=30, step=1
#     )
#     * 365.25
# )


# # Education level dropdown
# st.write("""
# ## Education level
# """)
# edu_level_values = list(value_cnt_norm_cal(full_data, "Education level").index)
# edu_level_key = [
#     "Secondary school",
#     "Higher education",
#     "Incomplete higher",
#     "Lower secondary",
#     "Academic degree",
# ]
# edu_level_dict = dict(zip(edu_level_key, edu_level_values))
# input_edu_level_key = st.selectbox("Select your education status", edu_level_key)
# input_edu_level_val = edu_level_dict.get(input_edu_level_key)


# # Car ownship input
# st.write("""
# ## Car ownship
# """)
# input_car_ownship = st.radio("Do you own a car?", ["Yes", "No"], index=0)

# # Property ownship input
# st.write("""
# ## Property ownship
# """)
# input_prop_ownship = st.radio("Do you own a property?", ["Yes", "No"], index=0)


# # Work phone input
# st.write("""
# ## Work phone
# """)
# input_work_phone = st.radio("Do you have a work phone?", ["Yes", "No"], index=0)
# work_phone_dict = {"Yes": 1, "No": 0}
# work_phone_val = work_phone_dict.get(input_work_phone)

# # Phone input
# st.write("""
# ## Phone
# """)
# input_phone = st.radio("Do you have a phone?", ["Yes", "No"], index=0)
# work_dict = {"Yes": 1, "No": 0}
# phone_val = work_dict.get(input_phone)

# # Email input
# st.write("""
# ## Email
# """)
# input_email = st.radio("Do you have an email?", ["Yes", "No"], index=0)
# email_dict = {"Yes": 1, "No": 0}
# email_val = email_dict.get(input_email)

# st.markdown("##")
# st.markdown("##")
# # Button
# predict_bt = st.button("Predict")

# # list of all the input variables
# profile_to_predict = [
#     0,  # ID
#     input_gender[:1],  # gender
#     input_car_ownship[:1],  # car ownership
#     input_prop_ownship[:1],  # property ownership
#     0,  # Children count (which will be dropped in the pipeline)
#     input_income,  # Income
#     input_employment_status_val,  # Employment status
#     input_edu_level_val,  # Education level
#     input_marital_status_val,  # Marital status
#     input_dwelling_type_val,  # Dwelling type
#     input_age,  # Age
#     input_employment_length,  # Employment length
#     1,  # Has a mobile phone (which will be dropped in the pipeline)
#     work_phone_val,  # Work phone
#     phone_val,  # Phone
#     email_val,  # Email
#     "to_be_droped",  # Job title (which will be dropped in the pipeline)
#     fam_member_count,  # Family member count
#     0.00,  # Account age (which will be dropped in the pipeline)
#     0,  # target set to 0 as a placeholder
# ]


# profile_to_predict_df = pd.DataFrame([profile_to_predict], columns=train_copy.columns)


# # add the profile to predict as a last row in the train data
# train_copy_with_profile_to_pred = pd.concat(
#     [train_copy, profile_to_predict_df], ignore_index=True
# )


# # whole dataset prepared
# train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)

# # Get the row with the ID = 0, and drop the ID, and target(placeholder) column
# profile_to_pred_prep = train_copy_with_profile_to_pred_prep[
#     train_copy_with_profile_to_pred_prep["ID"] == 0
# ].drop(columns=["ID", "Is high risk"])


# import logging
# import joblib
# import streamlit as st
# from streamlit_lottie import st_lottie_spinner
# import requests
# import time

# # Load Lottie animation
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_loading_an = load_lottieurl(
#     "https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json"
# )

import logging
import time
import streamlit as st
import numpy as np

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# def make_prediction(age, income, debt, marital_status, family_member_count, 
#                     dwelling_type, employment_status, employment_length, 
#                     education_level, car_ownership, property_ownership, 
#                     work_phone, phone, email, dept_amount):
#     try:
#         logger.info("Simulating model prediction based on manual logic")
        
#         logger.info(f"Input Values: Age={age}, Income={income}, Debt={debt}, "
#                      f"Marital Status={marital_status}, Family Count={family_member_count}, "
#                      f"Dwelling Type={dwelling_type}, Employment Status={employment_status}, "
#                      f"Employment Length={employment_length}, Education={education_level}, "
#                      f"Car Ownership={car_ownership}, Property Ownership={property_ownership}, "
#                      f"Work Phone={work_phone}, Phone={phone}, Email={email}, Debt Amount={dept_amount}") 
         
#         # Rejection condition based on income and employment length (new condition)
#         if income == 0 and employment_length <= 7:  # Low income and short employment length
#             logger.warning(f"Rejected: Income is {income} and employment length is {employment_length} years.")
#             return "Rejected: You can't apply for a credit card."

#         # Approval condition based on the income with zero debt
#         if income >= 843 and dept_amount == 0:
#             logger.info("Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # Rejection condition based on debt-to-income ratio
#         if dept_amount > income * 0.5:  # Debt-to-Income ratio greater than 50%
#             logger.warning(f"Debt amount {dept_amount} is too high for the income {income}. Rejected.")
#             return "Rejected: You can't apply for a credit card."

#         # Approval condition if not car or not property, income > 900, and debt <= 90
#         if (car_ownership == "No" or property_ownership == "No") and income > 900 and dept_amount <= 90:
#             logger.info("Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # Approval condition based on car and property ownership with income and debt checks
#         if car_ownership == "Yes" and property_ownership == "Yes":
#             if income > 500 and dept_amount < 400:
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."
#             else:
#                 logger.warning(f"Rejected: Income {income} is <= 500 or Debt {dept_amount} is >= 400.")
#                 return "Rejected: You can't apply for a credit card."

#         # Approval condition based on income and employment length
#         if income >= 50000 and employment_length >= 2:  # Example: high income and stable job
#             logger.info("Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # Additional checks for approval
#         if marital_status == "Married" and family_member_count > 1:  # Married with dependents
#             logger.info(f"Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # If none of the above conditions are met, it's a neutral case, but let's lean towards rejection
#         logger.info("Rejected: You can't apply for a credit card.")
#         return "Rejected: You can't apply for a credit card."
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         st.error(f"An unexpected error occurred: {str(e)}")
#         return "Error: Unable to process the request."



# def make_prediction(age, income, debt, marital_status, family_member_count, 
#                     dwelling_type, employment_status, employment_length, 
#                     education_level, car_ownership, property_ownership, 
#                     work_phone, phone, email, dept_amount):
#     try:
#         logger.info("Simulating model prediction based on manual logic")
        
#         logger.info(f"Input Values: Age={age}, Income={income}, Debt={debt}, "
#                      f"Marital Status={marital_status}, Family Count={family_member_count}, "
#                      f"Dwelling Type={dwelling_type}, Employment Status={employment_status}, "
#                      f"Employment Length={employment_length}, Education={education_level}, "
#                      f"Car Ownership={car_ownership}, Property Ownership={property_ownership}, "
#                      f"Work Phone={work_phone}, Phone={phone}, Email={email}, Debt Amount={dept_amount}") 


#         # Check if all contact methods are "No"
#         if work_phone == "No" and phone == "No" and email == "No":
#             logger.warning("Rejected: No contact information provided.")
#             return "Rejected: You can't apply for a credit card."

#         # Calculate debt-to-income ratio
#         if income > 0:  # Avoid division by zero
#             debt_to_income_ratio = dept_amount / income
#             logger.info(f"Calculated Debt-to-Income Ratio: {debt_to_income_ratio:.2f}")
#         else:
#             debt_to_income_ratio = 1  # If income is 0, set DTI to 1 (100%)

#         # Rejection condition based on debt-to-income ratio
#         if debt_to_income_ratio > 0.43:  # DTI ratio greater than 43%
#             logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
#             return "Rejected: You can't apply for a credit card."

#         # Approval condition if car ownership or property ownership is "No" and DTI <= 43%
#         if car_ownership == "No" or property_ownership == "No":
#             if debt_to_income_ratio <= 0.43:
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."
#             else:
#                 logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
#                 return "Rejected: You can't apply for a credit card."

#         # Approval condition based on car and property ownership with income and debt checks
#         if car_ownership == "Yes" and property_ownership == "Yes":
#             if income > 500 and dept_amount < 400:
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."
#             else:
#                 logger.warning(f"Rejected: Income {income} is <= 500 or Debt {dept_amount} is >= 400.")
#                 return "Rejected: You can't apply for a credit card."

#         # Approval condition based on income and employment length
#         if income >= 50000 and employment_length >= 2:  # Example: high income and stable job
#             logger.info("Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # Additional checks for approval
#         if marital_status == "Married" and family_member_count > 1:  # Married with dependents
#             logger.info(f"Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # If none of the above conditions are met, it's a neutral case, but let's lean towards rejection
#         logger.info("Rejected: You can't apply for a credit card.")
#         return "Rejected: You can't apply for a credit card."
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         st.error(f"An unexpected error occurred: {str(e)}")
#         return "Error: Unable to process the request."
def make_prediction(age, income, debt, marital_status, family_member_count, 
                    dwelling_type, employment_status, employment_length, 
                    education_level, car_ownership, property_ownership, 
                    work_phone, phone, email, dept_amount):
    try:
        logger.info("Simulating model prediction based on manual logic")
        
        logger.info(f"Input Values: Age={age}, Income={income}, Debt={debt}, "
                     f"Marital Status={marital_status}, Family Count={family_member_count}, "
                     f"Dwelling Type={dwelling_type}, Employment Status={employment_status}, "
                     f"Employment Length={employment_length}, Education={education_level}, "
                     f"Car Ownership={car_ownership}, Property Ownership={property_ownership}, "
                     f"Work Phone={work_phone}, Phone={phone}, Email={email}, Debt Amount={dept_amount}") 

        # Check if all contact methods are "No"
        if work_phone == "No" and phone == "No" and email == "No":
            logger.warning("Rejected: No contact information provided.")
            return "Rejected: You can't apply for a credit card."

        # Calculate debt-to-income ratio
        if income > 0:  # Avoid division by zero
            debt_to_income_ratio = dept_amount / income
            logger.info(f"Calculated Debt-to-Income Ratio: {debt_to_income_ratio:.2f}")
        else:
            debt_to_income_ratio = 1  # If income is 0, set DTI to 1 (100%)

        # Rejection condition based on debt-to-income ratio
        if debt_to_income_ratio > 0.43:  # DTI ratio greater than 43%
            logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
            return "Rejected: You can't apply for a credit card."

        # Approval condition if income-to-debt ratio is valid and other factors are satisfied
        if income >= 300 and employment_length >= 2:
            if debt_to_income_ratio <= 0.43 and (car_ownership == "Yes" or property_ownership == "Yes"):
                logger.info("Approved: You can apply for a credit card.")
                return "Approved: You can apply for a credit card."

        # Approval condition if car ownership or property ownership is "No" and DTI <= 43%
        if car_ownership == "No" or property_ownership == "No":
            if debt_to_income_ratio <= 0.43:
                logger.info("Approved: You can apply for a credit card.")
                return "Approved: You can apply for a credit card."
            else:
                logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
                return "Rejected: You can't apply for a credit card."

        # Approval condition based on car and property ownership with income and debt checks
        if car_ownership == "Yes" and property_ownership == "Yes":
            if income >= 300 and dept_amount < 180:
                logger.info("Approved: You can apply for a credit card.")
                return "Approved: You can apply for a credit card."
            else:
                logger.warning(f"Rejected: Income {income} is <= 500 or Debt {dept_amount} is >= 400.")
                return "Rejected: You can't apply for a credit card."

        # Approval condition based on income and employment length
        if income >= 50000 and employment_length >= 2:  # Example: high income and stable job
            logger.info("Approved: You can apply for a credit card.")
            return "Approved: You can apply for a credit card."

        # If none of the above conditions are met, it's a neutral case, but let's lean towards rejection
        logger.info("Rejected: You can't apply for a credit card.")
        return "Rejected: You can't apply for a credit card."
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        return "Error: Unable to process the request."

st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .section-title {
        background-color: #ffcccb;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Existing UI components for user inputs
st.write("""
# Credit card approval prediction
This app predicts if an applicant will be approved for a credit card or not. Just fill in the following information and click on the Predict button.
""")

# Gender input
st.write("""
## Gender
""")
input_gender = st.radio("Select your gender", ["Male", "Female"], index=0)

# Age input slider with a unique key
st.write("""
## Age
""")
input_age = np.negative(
    st.slider("Select your age", value=42, min_value=18, max_value=70, step=1, key="age_slider")
) * 365.25  # Multiply after the slider to avoid syntax error

# Marital status input dropdown
st.write("""
## Marital status
""")
marital_status_values = ["Married", "Single", "Divorced", "Widowed"]
input_marital_status = st.selectbox("Select your marital status", marital_status_values)

# Family member count with a unique key
st.write("""
## Family member count
""")
fam_member_count = float(
    st.selectbox("Select your family member count", [1, 2, 3, 4, 5, 6], key="fam_member_count_selectbox")
)

# Dwelling type dropdown
st.write("""
## Dwelling type
""")
dwelling_type_values = ["House", "Rented", "Shared", "Others"]
input_dwelling_type = st.selectbox("Select the type of dwelling you reside in", dwelling_type_values)

# Income with a unique key
st.write("""
## Income
""")
input_income = int(st.text_input("Enter your income (in USD)", 0, key="income_text_input"))

# Employment status dropdown
st.write("""
## Employment status
""")
employment_status_values = ["Employed", "Unemployed", "Self-employed", "Retired"]
input_employment_status = st.selectbox("Select your employment status", employment_status_values)

# Employment length input slider
st.write("""
## Employment length
""")
input_employment_length = np.negative(
    st.slider("Select your employment length (in years)", value=6, min_value=0, max_value=30, step=1, key="employment_length_slider")
)

# Education level dropdown with a unique key
st.write("""
## Education level
""")
edu_level_values = ["High School", "Undergraduate", "Postgraduate", "Doctorate"]
input_edu_level = st.selectbox("Select your education status", edu_level_values, key="education_selectbox")

# Department Amount input section (added below education level)
st.write("""
## Debt Amount
""")
dept_amount = float(st.text_input("Enter your debt amount (in USD)", 0))  # Make sure it's a float

# Car ownership radio button with a unique key
st.write("""
## Car ownership
""")
input_car_ownship = st.radio("Do you own a car?", ["Yes", "No"], index=0, key="car_ownership_radio")

# Property ownership radio button with a unique key
st.write("""
## Property ownership
""")
input_prop_ownship = st.radio("Do you own a property?", ["Yes", "No"], index=0, key="property_ownership_radio")

# Work phone
st.write("""
## Work phone
""")
input_work_phone = st.radio("Do you have a work phone?", ["Yes", "No"], index=0, key="work_phone_radio")

# Phone
st.write("""
## Phone
""")
input_phone = st.radio("Do you have a phone?", ["Yes", "No"], index=0, key="phone_radio")

# Email
st.write("""
## Email
""")
input_email = st.radio("Do you have an email?", ["Yes", "No"], index=0, key="email_radio")

# Button to trigger the prediction
predict_bt = st.button("Predict", key="predict_button")

if predict_bt:
    # Call the prediction function with the user input values
    prediction = make_prediction(
        age=input_age,
        income=input_income,
        debt=0,  # Assuming debt is 0 for now; can add a slider for this
        marital_status=input_marital_status,
        family_member_count=fam_member_count,
        dwelling_type=input_dwelling_type,
        employment_status=input_employment_status,
        employment_length=input_employment_length,
        education_level=input_edu_level,
        car_ownership=input_car_ownship,
        property_ownership=input_prop_ownship,
        work_phone=input_work_phone,
        phone=input_phone,
        email=input_email,
        dept_amount=dept_amount  # Pass dept_amount here
    )
    import streamlit as st
    import time

    # Simulate the prediction result
    if "Approved" in prediction:
        # Success Case: Green Balloon for approval
        st.markdown(f"""
        <div style="font-size: 24px; color: green; font-weight: bold; text-align: center; margin-top: 50px; padding: 20px; border: 2px solid green; 
                    background-color: rgba(0, 255, 0, 0.1); border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 255, 0, 0.3);">
            {prediction} 🎈
        </div>
        """, unsafe_allow_html=True)
        st.balloons()  # Show balloons for success
        time.sleep(1)  # Delay for a brief moment to let the balloons appear

    elif "Rejected" in prediction:
        # Failure Case: Show error message for rejection
        st.markdown(f"""
        <div style="font-size: 28px; color: red; font-weight: bold; text-align: center; margin-top: 50px; padding: 20px; border: 2px solid red;
                    background-color: rgba(255, 0, 0, 0.1); border-radius: 10px; box-shadow: 0 4px 8px rgba(255, 0, 0, 0.3);">
            {prediction}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("An error occurred while processing your predicting")



# def make_prediction(age, income, debt, marital_status, family_member_count, 
#                     dwelling_type, employment_status, employment_length, 
#                     education_level, car_ownership, property_ownership, 
#                     work_phone, phone, email, dept_amount):
#     try:
#         logger.info("Simulating model prediction based on manual logic")
        
#         logger.info(f"Input Values: Age={age}, Income={income}, Debt={debt}, "
#                      f"Marital Status={marital_status}, Family Count={family_member_count}, "
#                      f"Dwelling Type={dwelling_type}, Employment Status={employment_status}, "
#                      f"Employment Length={employment_length}, Education={education_level}, "
#                      f"Car Ownership={car_ownership}, Property Ownership={property_ownership}, "
#                      f"Work Phone={work_phone}, Phone={phone}, Email={email}, Debt Amount={dept_amount}") 

#         # Check if all contact methods are "No"
#         if work_phone == "No" and phone == "No" and email == "No":
#             logger.warning("Rejected: No contact information provided.")
#             return "Rejected: You can't apply for a credit card."

#         # Calculate debt-to-income ratio
#         if income > 0:  # Avoid division by zero
#             debt_to_income_ratio = dept_amount / income
#             logger.info(f"Calculated Debt-to-Income Ratio: {debt_to_income_ratio:.2f}")
#         else:
#             debt_to_income_ratio = 1  # If income is 0, set DTI to 1 (100%)

#         # Rejection condition based on debt-to-income ratio
#         if debt_to_income_ratio > 0.43:  # DTI ratio greater than 43%
#             logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
#             return "Rejected: You can't apply for a credit card."

#         # Approval condition if income-to-debt ratio is valid and other factors are satisfied
#         if income >= 300 and employment_length >= 2:
#             if debt_to_income_ratio <= 0.43 and (car_ownership == "Yes" or property_ownership == "Yes"):
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."

#         # Approval condition if car ownership or property ownership is "No" and DTI <= 43%
#         if car_ownership == "No" or property_ownership == "No":
#             if debt_to_income_ratio <= 0.43:
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."
#             else:
#                 logger.warning(f"Rejected: Debt-to-Income ratio {debt_to_income_ratio:.2f} exceeds 43%.")
#                 return "Rejected: You can't apply for a credit card."

#         # Approval condition based on car and property ownership with income and debt checks
#         if car_ownership == "Yes" and property_ownership == "Yes":
#             if income > 500 and dept_amount < 400:
#                 logger.info("Approved: You can apply for a credit card.")
#                 return "Approved: You can apply for a credit card."
#             else:
#                 logger.warning(f"Rejected: Income {income} is <= 500 or Debt {dept_amount} is >= 400.")
#                 return "Rejected: You can't apply for a credit card."

#         # Approval condition based on income and employment length
#         if income >= 50000 and employment_length >= 2:  # Example: high income and stable job
#             logger.info("Approved: You can apply for a credit card.")
#             return "Approved: You can apply for a credit card."

#         # If none of the above conditions are met, it's a neutral case, but let's lean towards rejection
#         logger.info("Rejected: You can't apply for a credit card.")
#         return "Rejected: You can't apply for a credit card."
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         st.error(f"An unexpected error occurred: {str(e)}")
#         return "Error: Unable to process the request."

#     # if "Approved" in prediction:
    #     # Success Case: Green Balloon for approval
    #     st.success(prediction)
    #     st.balloons()
    #     time.sleep(1)
    # elif "Rejected" in prediction:
    #     # Failure Case: Show error message for rejection
    #     st.error(prediction)
    # else:
    #     st.error("An error occurred while processing your prediction.")

# def make_prediction(age, income, debt, marital_status, family_member_count, 
#                     dwelling_type, employment_status, employment_length, 
#                     education_level, car_ownership, property_ownership, 
#                     work_phone, phone, email, dept_amount):
#     try:
#         logger.info("Simulating model prediction based on manual logic")
        
#         logger.info(f"Input Values: Age={age}, Income={income}, Debt={debt}, "
#                      f"Marital Status={marital_status}, Family Count={family_member_count}, "
#                      f"Dwelling Type={dwelling_type}, Employment Status={employment_status}, "
#                      f"Employment Length={employment_length}, Education={education_level}, "
#                      f"Car Ownership={car_ownership}, Property Ownership={property_ownership}, "
#                      f"Work Phone={work_phone}, Phone={phone}, Email={email}, Debt Amount={dept_amount}") 

#         # Rejection condition based on debt-to-income ratio
#         if dept_amount > income * 0.5:  # Debt-to-Income ratio greater than 50%
#             logger.warning(f"Debt amount {dept_amount} is too high for the income {income}. Rejected.")
#             return "Rejected: Debt amount too high for income."

#         # Approval condition based on income and employment length
#         if income >= 50000 and employment_length >= 2:  # Example: high income and stable job
#             logger.info("Approved: High income and stable employment.")
#             return "Approved: High income and stable employment."

#         # Additional checks for approval
#         if marital_status == "Married" and family_member_count > 1:  # Married with dependents
#             logger.info(f"Approved: Married with dependents (family count {family_member_count}).")
#             return "Approved: Married with dependents."

#         if car_ownership == "Yes" and property_ownership == "Yes":  # Owns both car and property
#             logger.info("Approved: Owns both car and property.")
#             return "Approved: Owns both car and property."

#         # If none of the above conditions are met, it's a neutral case, but let's lean towards rejection
#         logger.info("Rejected: Does not meet approval criteria.")
#         return "Rejected: Does not meet approval criteria."
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         st.error(f"An unexpected error occurred: {str(e)}")
#         return "Error: Unable to process the request."



# # Existing UI components for user inputs
# st.write("""
# # Credit card approval prediction
# This app predicts if an applicant will be approved for a credit card or not. Just fill in the following information and click on the Predict button.
# """)

# # Gender input
# st.write("""
# ## Gender
# """)
# input_gender = st.radio("Select your gender", ["Male", "Female"], index=0)

# # Age input slider with a unique key
# st.write("""
# ## Age
# """)
# input_age = np.negative(
#     st.slider("Select your age", value=42, min_value=18, max_value=70, step=1, key="age_slider")
# ) * 365.25  # Multiply after the slider to avoid syntax error

# # Marital status input dropdown
# st.write("""
# ## Marital status
# """)
# marital_status_values = ["Married", "Single", "Divorced", "Widowed"]
# input_marital_status = st.selectbox("Select your marital status", marital_status_values)

# # Family member count with a unique key
# st.write("""
# ## Family member count
# """)
# fam_member_count = float(
#     st.selectbox("Select your family member count", [1, 2, 3, 4, 5, 6], key="fam_member_count_selectbox")
# )

# # Dwelling type dropdown
# st.write("""
# ## Dwelling type
# """)
# dwelling_type_values = ["House", "Rented", "Shared", "Others"]
# input_dwelling_type = st.selectbox("Select the type of dwelling you reside in", dwelling_type_values)

# # Income with a unique key
# st.write("""
# ## Income
# """)
# input_income = int(st.text_input("Enter your income (in USD)", 0, key="income_text_input"))

# # Employment status dropdown
# st.write("""
# ## Employment status
# """)
# employment_status_values = ["Employed", "Unemployed", "Self-employed", "Retired"]
# input_employment_status = st.selectbox("Select your employment status", employment_status_values)

# # Employment length input slider
# # Employment length input slider with a unique key
# st.write("""
# ## Employment length
# """)
# input_employment_length = np.negative(
#     st.slider("Select your employment length (in years)", value=6, min_value=0, max_value=30, step=1, key="employment_length_slider")
# )

# # Education level dropdown with a unique key
# st.write("""
# ## Education level
# """)
# edu_level_values = ["High School", "Undergraduate", "Postgraduate", "Doctorate"]
# input_edu_level = st.selectbox("Select your education status", edu_level_values, key="education_selectbox")

# # Department Amount input section (added below education level)
# st.write("""
# ## Debt Amount
# """)
# # Debt amount input field
# dept_amount = float(st.text_input("Enter your debt amount (in USD)", 0))  # Make sure it's a float

# st.write("""
# ## Car ownership
# """)
# # Car ownership radio button with a unique key
# input_car_ownship = st.radio("Do you own a car?", ["Yes", "No"], index=0, key="car_ownership_radio")

# st.write("""
# ## Property ownership
# """)
# # Property ownership radio button with a unique key
# input_prop_ownship = st.radio("Do you own a property?", ["Yes", "No"], index=0, key="property_ownership_radio")

# st.write("""
# ## Work phone
# """)

# # Work phone radio button with a unique key
# input_work_phone = st.radio("Do you have a work phone?", ["Yes", "No"], index=0, key="work_phone_radio")


# st.write("""
# ## Phone
# """)
# # Phone radio button with a unique key
# input_phone = st.radio("Do you have a phone?", ["Yes", "No"], index=0, key="phone_radio")

# st.write("""
# ## Email
# """)

# # Email radio button with a unique key
# input_email = st.radio("Do you have an email?", ["Yes", "No"], index=0, key="email_radio")
# st.markdown("##")
# st.markdown("##")

# # Button with a unique key to prevent DuplicateWidgetID error
# predict_bt = st.button("Predict", key="predict_button")

# if predict_bt:
#     # Call the prediction function with the user input values
#     prediction = make_prediction(
#     age=input_age,
#     income=input_income,
#     debt=0,  # Assuming debt is 0 for now; can add a slider for this
#     marital_status=input_marital_status,
#     family_member_count=fam_member_count,
#     dwelling_type=input_dwelling_type,
#     employment_status=input_employment_status,
#     employment_length=input_employment_length,
#     education_level=input_edu_level,
#     car_ownership=input_car_ownship,
#     property_ownership=input_prop_ownship,
#     work_phone=input_work_phone,
#     phone=input_phone,
#     email=input_email,
#     dept_amount=dept_amount  # Pass dept_amount here
# )
#     if prediction == 0:
#         # Green balloon effect on success
#         st.success("Congratulations! Your credit card application is approved.")
#         st.balloons()  # Show balloons for success
#         time.sleep(1)  # Delay for a brief moment to let the balloons appear
#     elif prediction == 1:
#         st.error("Sorry, your credit card application was rejected.")
#     else:
#         st.error("An error occurred while processing your prediction.")
