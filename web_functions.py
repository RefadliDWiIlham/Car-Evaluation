#IMPORT MODUL
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

@st.cache_data()
def load_data():
  
  df = pd.read_csv('car-evaluation-data.csv')

  x = df[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]
  y = df[["class"]]

  return df, x, y

@st.cache_data()
def train_model(x,y) :
  model = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    random_state=42, splitter='best'
)
  model.fit(x,y)

  score = model.score(x,y)

  return model, score

def predict(x,y,feaures) :
  model, score = train_model(x,y)

  prediction = model.predict(np.array(feaures).reshape(1,-1))
  return prediction, score