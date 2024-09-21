import pandas as pd
import plotly.express as px
import streamlit as st

from dataloading import NewsDataModule
from main import NewsModel

st.set_page_config(page_title="Sports News Classifier", layout="wide")


@st.cache_data
def load_model(model_identifier):
    model = NewsModel.load_from_checkpoint(f'saved/{model_identifier}.ckpt')
    dm = NewsDataModule("data/bbcsport", model.hparams.batch_size, model.hparams.max_token_len)
    return model, dm

model_options = {
    "Baseline (60% of Original Data, No Synthetic)": "BASELINE",
    "Reduced Training Data (50% of Original, No Synthetic)": "REDUCED_TRAINING_DATA",
    "Synthetic Augmentation (50% of Original + Synthetic)": "SYNTHETIC_AUGMENTATION",
    "Synthetic Test Set (50% of Original data for Training)": "SYNTHETIC_TEST",
    "Combined Dataset (50% of Combined dataset for Training)": "COMBINED_DATASET"
}

st.markdown("""
<style>
/* Set a background color/image for the entire app */

/* Button customizations */
.stButton > button {
    width: 150px; /* Increase width */
    height: 50px; /* Increase height */
    font-size: 18px; /* Increase font size */
    margin: 10px auto; /* Center button with margin */
    display: block; /* Block display to fill space */
    background-color: #0d6efd; /* Bootstrap primary color for the button */
    color: white; /* White text color */
    border-radius: 5px; /* Rounded corners */
    border: none; /* Remove default border */
}

/* Button hover effect */
.stButton > button:hover {
    background-color: #0056b3; /* Darker shade on hover */
    cursor: pointer; /* Cursor changes to pointer to indicate clickable */
}

/* Footer styling */
footer {
    font-size: 0.875rem; /* Smaller font size for footer */
    text-align: center; /* Center align footer text */
    padding: 20px; /* Padding for spacing */
}
</style>
""", unsafe_allow_html=True)


st.title("Sports News Article Classifier")
st.markdown(
    """
     
    ##### Select the model configuration for classification
    """)
selected_model = st.selectbox("Choose Model Configuration", list(model_options.keys()))
model_identifier = model_options[selected_model]
model, dm = load_model(model_identifier)

st.markdown(
    """
     
    ##### Enter a sports news article to classify its category
    """)
user_input = st.text_area("Article", placeholder="Paste the sports news article here...", height=300)

if st.button('Classify'):
    if user_input:
        predicted_class = model.predict(user_input, dm.tokenizer, dm.class_labels)
        st.success(f"Predicted Category: {predicted_class}")
    else:
        st.error("Please enter an article for classification.")

st.markdown("""
---
&nbsp;
""")

with st.expander("Project Overview and Dataset Description"):
    st.markdown("""
    ## Project Overview
    This application is powered by a RoBERTa model fined-tuned to classify sports news articles into five categories: Athletics, Cricket, Football, Rugby, and Tennis.     

    ### Dataset
    The training data consists of two distinct parts:

    **BBCSport Dataset**:
    - Source: [BBC Sports website (2004-2005)](http://mlg.ucd.ie/datasets/bbc.html)
    - Total Documents: 737
    - Topics: Athletics, Cricket, Football, Rugby, Tennis

    **Synthetic Data**:
    - Generated with GPT-4
    - Total Articles: 136 (27% of our training data)
    - Class Distribution:
        - Athletics: 33
        - Cricket: 30
        - Tennis: 23
        - Rugby: 21
        - Football: 29

    ### Class Distribution Plot
    Below is the class distribution plot of the BBCSport dataset used for training our model. This visualization helps in understanding the balance of different sports categories in the dataset.
    """)
    st.image("saved/Distribution.png")

with st.expander("Model Configurations"):
    st.markdown("""
    ### Model Configurations
    **1. Baseline (60% of Original Data, No Synthetic):**
    - Train: 60% of original BBC data
    - Validation: 20% of original BBC data
    - Test: 20% of original BBC data
    - Purpose: To establish a baseline performance using a majority of the original dataset for training, without synthetic augmentation.

    **2. Reduced Training Data (50% of Original, No Synthetic):**
    - Train: 50% of original BBC data
    - Validation: 25% of original BBC data
    - Test: 25% of original BBC data
    - Purpose: To assess performance with a significantly reduced training dataset, without synthetic data.

    **3. Synthetic Augmentation (50% of Original + Synthetic):**
    - Train: 50% of original BBC data + All synthetic data (137 samples)
    - Validation: 25% of original BBC data
    - Test: 25% of original BBC data
    - Purpose: To evaluate the impact of augmenting training data with synthetic samples on model performance.

    **4. Synthetic Test Set (50% of Original Data for Training):**
    - Train: 50% of original BBC data
    - Validation: 25% of original BBC data
    - Test: All synthetic data (137 samples)
    - Purpose: To test the model's generalization on an entirely synthetic test set.

    **5. Combined Original and Synthetic Dataset:**
    - Train: 50% of the combined dataset
    - Validation: 25% of the combined dataset
    - Test: 25% of the combined dataset
    - Purpose: Evaluate the model's performance when the whole dataset is augmented with synthetic data.
    """)


# Experimental Results Table
df_metrics = pd.DataFrame({
    "Configurations": [
        "Baseline (60% of Original Data, No Synthetic)",
        "Reduced Training Data (50% of Original, No Synthetic)",
        "Synthetic Augmentation (50% of Original + Synthetic)",
        "Synthetic Test Set (50% of Original Data for Training)",
        "Combined Dataset (50% of Combined Dataset for Training)"
    ],
    "Acc (%)": [97.3, 93.5, 99.5, 99.3, 97.7],
    "F1": [97.3, 93.5, 99.5, 99.3, 97.7],
    "Precision": [97.5, 94.2, 99.5, 99.3, 97.9],
    "Recall": [97.3, 93.5, 99.5, 99.3, 97.7],
    "Test Loss": [0.4, 0.8, 0.3, 0.4, 0.5],
    "Val Loss": [0.4, 0.7, 0.2, 0.4, 0.5]
})

with st.expander("Model Performance"):
    st.table(df_metrics)

with st.expander("Interactive Performance Visualization"):
    fig = px.bar(df_metrics, x='Configurations', y='Acc (%)', title="Model Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)

footer = """
---
#### Developed by:
**Penjan Antonio Eng Lim, Young Bong Kim**  
*Department of Computer Science and Engineering, Chungnam National University*  
Email: [daehanlim@o.cnu.ac.kr](mailto:daehanlim@o.cnu.ac.kr), [kimsohong@naver.com](mailto:kimsohong@naver.com)
"""

st.markdown(footer)

