import streamlit as st
import pandas as pd
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Malaria Outbreak Predictor',
    page_icon=':Mosquito', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_data():
    """ Get the NDVI AND NDWI VALUE INFORMATION AND INPUT IT INTO THE MODEL
    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'ndvi_ndwi_values_2000_2023_gen.csv'
    df = pd.read_csv(DATA_FILENAME)


    df = df.drop(columns= 'Unnamed: 0')

    return df

df = get_data()


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Malaria Outbreak Predictor
# :Mosquito
This model predicts the outbreak of malaria in Sudan

'''

''' 
below is a test to see if the data is uploading

'''

st.write(df.head())
# Add some spacing
''
''


#user_text1 = st.text_area('input protein sequence 1')
#st.write(user_text1)

#user_text2 = st.text_area('input protein sequence 2')
#st.write(user_text2)

#seqs = [user_text1, user_text2]

''
''



#merged = None

#if user_text1 and user_text2:

#    merged = seq_features(seqs)

#    st.write('the following are the features generated from the protein pair')

''
''
#st.write(merged)




#if merged:

#    ready_data = standard(usable_df, merged)

#    st.write('data is loading')


#def rfc(data):

#    rfc = RandomForestClassifier(criterion='entropy', n_estimators=500)

#    rfc = rfc.fit(data[0], data[1])

#    return rfc


#model = None

#if ready_data != None:

#    model = rfc(ready_data)

#    st.write('model is training')




#if model != None:

#    merged = np.array(merged)

#    merged = merged.reshape(1, -1)

#    res = model.predict(merged)

#    prob = model.predict_proba(merged)

#    st.write('model is predicting')

#    st.write(res)

#    st.write('percent chance is: ', prob)