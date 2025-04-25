# import streamlit as st
# import joblib

# st.title("Reddit Comment Classification")
# st.markdown("### All you have to do to use this app is enter a comment and hit the Predict button.")

# reddit_comment = [st.text_area("Input your comment here:")]

# def load_artifacts():
#     model_pipeline = joblib.load("reddit_model_pipeline.joblib")
#     return model_pipeline

# model_pipeline = load_artifacts()

# def predict(reddit_comment):
#     X = reddit_comment
#     predictions = model_pipeline.predict_proba(X)
#     return {'Predictions': predictions}

# preds = predict(reddit_comment)
# st.write(preds)

# import streamlit as st
# import joblib
# import pandas as pd

# st.title("Reddit Comment Classification")
# st.markdown("### All you have to do to use this app is enter a comment and hit the Predict button.")

# # Get user input
# reddit_comment = st.text_area("Input your comment here:")

# def load_artifacts():
#     try:
#         model_pipeline = joblib.load("reddit_model_pipeline.joblib")
#         return model_pipeline
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Load model when the app starts
# model_pipeline = load_artifacts()

# # Make prediction when button is clicked
# if st.button("Predict"):
#     if reddit_comment:
#         try:
#             # The model was likely trained on a DataFrame or specific format
#             # Pass the text in the expected format - a list containing the comment
#             predictions = model_pipeline.predict_proba([reddit_comment])
            
#             # Format the prediction results for display
#             prob_negative = predictions[0][0]
#             prob_positive = predictions[0][1]
            
#             st.write("### Prediction Results:")
#             st.write(f"Negative class probability: {prob_negative:.4f}")
#             st.write(f"Positive class probability: {prob_positive:.4f}")
            
#             # Add a visual indicator for the prediction
#             if prob_positive > prob_negative:
#                 st.success(f"This comment is classified as POSITIVE with {prob_positive:.2%} confidence")
#             else:
#                 st.error(f"This comment is classified as NEGATIVE with {prob_negative:.2%} confidence")
                
#         except Exception as e:
#             st.error(f"Prediction error: {str(e)}")
#             st.info("Try retraining your model or check the model pipeline structure")
#     else:
#         st.warning("Please enter a comment to classify")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Reddit Comment Classification")
st.markdown("### All you have to do to use this app is enter a comment and hit the Predict button.")

# Load the model
def load_artifacts():
    try:
        model_pipeline = joblib.load("reddit_model_pipeline.joblib")
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_artifacts()

# Single prediction section
reddit_comment = st.text_area("Input your comment here:")

if st.button("Predict"):
    if reddit_comment:
        try:
            # For single predictions
            prediction = model_pipeline.predict([reddit_comment])[0]
            # Display the prediction using st.metric
            st.metric("Should this comment be removed (0: No; 1: Yes)", int(prediction))
            
            # Optional: display probability
            prob = model_pipeline.predict_proba([reddit_comment])[0]
            st.write(f"Confidence scores - Keep: {prob[0]:.2f}, Remove: {prob[1]:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.warning("Please enter a comment to predict")

# Batch prediction section
st.header("Get a Batch of Predictions")
batches = st.file_uploader("Upload File", type='csv')

if batches is not None:
    try:
        # Read the uploaded CSV
        dataframe = pd.read_csv(batches)
        
        # Check if the file has column names
        if len(dataframe.columns) == 1:
            # If only one column, use it as comments
            comments = dataframe.iloc[:, 0].tolist()
        else:
            # If multiple columns, assume first column contains comments
            comments = dataframe.iloc[:, 0].tolist()
            st.info("Using first column for predictions")
        
        # Generate predictions
        predictions = model_pipeline.predict_proba(comments)
        
        # Create results dataframe
        batch_predictions = pd.DataFrame({
            "Comment": comments,
            "Keep": [p[0] for p in predictions],
            "Remove": [p[1] for p in predictions],
            "Prediction": model_pipeline.predict(comments)
        })
        
        # Display results
        st.write(batch_predictions)
        
        # Download button
        st.download_button(
            'Download Predictions', 
            data=batch_predictions.to_csv(index=False).encode('utf-8'), 
            file_name='predictions.csv', 
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")
        st.info("Make sure your CSV file contains comments in the first column")