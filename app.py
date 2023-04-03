import streamlit as st
import pandas as pd
import os
import argparse
import shutil
# from predict.check_files import main
# import yaml
import sys

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline






# Necessary args to parse from predict.yaml
# args = argparse.ArgumentParser()
# args.add_argument('--config', default='predict.yaml')
# args.add_argument('--schema', default='schema.yaml')
# parsed_args = args.parse_args()
# config_path = parsed_args.config
# with open(config_path) as yaml_file:
#     config = yaml.safe_load(yaml_file)


# save_file_dir = config["predict"]["valid_and_preprocess"]["valid_and_preprocess_dir"]

# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
# os.makedirs(save_path, exist_ok=True)

# model_dir = config["predict"]["model_dir"]
# model_name = [ model for model in os.listdir(model_dir) if model[-4:] == ".sav" ]
# model_path = os.path.join(model_dir, model_name[0])

# pred_file_dir = config['predict']['valid_and_preprocess']['preprocessed_files']

# Setting Background for Streamlit
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# info customization
def display_info(txt, color="green"):
    st.markdown(f"""<p style='background-color:{color};
                                           color:white;
                                           font-size:18px;
                                           border-radius:3px;
                                           line-height:60px;
                                           padding-left:17px;
                                           opacity:0.6'>
                                           {txt}</style>
                                           <br></p>""" 
                ,unsafe_allow_html=True) 

# Contents on Page
st.title("Insurance Fraud Detection \n")

col1, col2 = st.columns(2)
choice = col1.radio (("Choose From Below Options: "), ["Show Available Files", "Upload Own File Here"])

save_file_path = os.path.join(os.getcwd(), "artifacts")
files_dir = os.path.join(os.getcwd(), "artifacts", "predict_files")

try:
    if choice == "Show Available Files":

        list_file_dir = os.listdir(files_dir)

        chosen_file = col2.radio("\n Choose From Below Options: ", list_file_dir)

        st.subheader(f"PREDICTION OF {chosen_file}")


        pred_file_path = os.path.join(files_dir, chosen_file)


        # @st.cache
        # def convert_df(df):
        #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
        #     return df.to_csv().encode('utf-8')
        #
        #
        # csv = convert_df(my_large_df)

        st.download_button(
            label=f"Download {chosen_file} as CSV",
            data=pred_file_path,
            file_name='large_df.csv',
            mime='text/csv',
        )

        pred = PredictPipeline()
        res = pred.initiate_predict_pipeline(pred_file_path)

        st.write(res)

    if choice == "Upload Own File Here":
        st.write("Coming Soon ...")


except Exception as e:
    raise CustomException(e, sys)
    
#     if choice == "Upload Own File Here":
#         col2.subheader("Drag and Drop your File ")
#         csv_file = col2.file_uploader("Upload csv file", type=["csv"])
#
#         if csv_file is not None:
#
#             # To See details
#             file_details = {"filename":csv_file.name, "filetype":csv_file.type,
#                             "filesize":csv_file.size}
#             col2.write(file_details)
#
#             if col2.button("Process"):
#                 df = pd.read_csv(csv_file)
#                 st.dataframe(df)
#                 df.to_csv(os.path.join(save_file_dir, csv_file.name), index=None)
#
#                 status, e = main()
#                 if status :
#                     display_info(e, "green")
#
#                     pred_file_path = os.path.join(pred_file_dir, "predict_file.csv")
#                     pred_file = pd.read_csv(pred_file_path)
#                     result = loaded_model.predict(pred_file)
#
#                     df = pd.DataFrame(result)
#                     df = df.replace(0, "Bad Wafer")
#                     df = df.replace(1, "Good Wafer")
#                     st.write(df)
#
#                 else:
#                     # st.error(e)
#                     display_info("FILE VALIDATION FAILED - CHANGE FILE", "red")
#                     display_info("Cross Check:- Must have 591 columns", "blue")
#
# except Exception as e:
#     raise e
#
#
