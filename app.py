import streamlit as st
import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline
from cfg.config import StreamlitConfig

class Streamlit:
    def __init__(self):
        self.streamlit_config = StreamlitConfig()

    # Setting Background for Streamlit
    def add_bg_from_url(self):
        try:
            st.markdown(
                 f"""
                 <style>
                 .stApp {{
                     background-image: url({self.streamlit_config.bg_url});
                     background-attachment: fixed;
                     background-size: cover
                 }}
                 </style>
                 """,
                 unsafe_allow_html=True
             )
            logging.info("Background added successfully")
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


    # info customization
    def display_info(self, txt, color="green"):
        try:
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
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


    # Contents on Page
    def content_on_page(self):
        try:
            st.title(f"{self.streamlit_config.title}  \n")

            col1, col2 = st.columns(2)
            choice = col1.radio(("Choose From Below Options: "), ["Show Available Files", "Upload Own File Here"])

            if choice == "Show Available Files":
                files_dir =  Path(self.streamlit_config.artifacts_dir) / self.streamlit_config.files_dir
                list_file_dir = os.listdir(files_dir)

                chosen_file = col2.radio("\n Choose From Below Options: ", list_file_dir)

                st.subheader(f"PREDICTED --- :orange[{chosen_file}]")

                pred_file_path = files_dir / Path(chosen_file)
                print(pred_file_path)
                pred = PredictPipeline()
                res = pred.initiate_predict_pipeline(pred_file_path)

                st.write(res)

                logging.info("Choice: Show Available Files run successfully")

            if choice == "Upload Own File Here":
                st.write("Coming Soon ...")
                logging.info("Choice: Upload Own File Here run successfully")

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Streamlit()
    obj.add_bg_from_url()
    obj.content_on_page()
