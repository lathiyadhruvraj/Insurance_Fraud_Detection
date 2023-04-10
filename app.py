import streamlit as st
import os
import sys

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

                list_file_dir = os.listdir(self.streamlit_config.files_dir)

                chosen_file = col2.radio("\n Choose From Below Options: ", list_file_dir)

                st.subheader(f"PREDICTED --- :orange[{chosen_file}]")

                pred_file_path = os.path.join(self.streamlit_config.files_dir, chosen_file)

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
