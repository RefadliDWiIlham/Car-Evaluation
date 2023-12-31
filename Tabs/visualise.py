import warnings
import matplotlib as plt
from sklearn import tree
import streamlit as st

from web_functions import train_model

def app(df, x, y):
    
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi")

    if st.checkbox("Plot Decision Tree") :
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(decision_tree=model,max_depth=4, out_file=None, filled=True, rounded=True, 
                    feature_names=x.columns,
                    class_names=['1','2','3','4'],
                    special_characters=True
)
        st.graphviz_chart(dot_data) 