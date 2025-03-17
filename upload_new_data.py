import pandas as pd
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# 1. Load the New Resume Data from CSV
# -------------------------------
# If your CSV file doesn't have headers, set header=None
new_data = pd.read_csv("RES_BAN.csv", header=None)
# If the file has a header but it's not what you expect, you can still rename it:
new_data.columns = ['Resume_str']
print("Columns after renaming:", new_data.columns)
print("New data loaded. Sample:")
print(new_data.head())

# -------------------------------
# 2. Preprocess the Resume Text
# -------------------------------
def preprocess_text(text):
    """
    Converts text to lowercase, removes digits, punctuation,
    and extra whitespace.
    """
    if not isinstance(text, str):  # Check if text is not a string
        text = str(text) if text is not None else ""  # Convert to string or empty string
    
    text = text.lower()                         # Convert to lowercase
    text = re.sub(r'\d+', '', text)           # Remove digits
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
    return text

# Apply preprocessing on the "Resume_str" column
new_data["cleaned_resume"] = new_data["Resume_str"].apply(preprocess_text)
print("\nPreprocessed text sample:")
print(new_data["cleaned_resume"].head())

# -------------------------------
# 3. Convert Text to Sequences and Pad
# -------------------------------
# Load the tokenizer saved during training
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully!")

# Convert the cleaned text to sequences
sequences = tokenizer.texts_to_sequences(new_data["cleaned_resume"])

# Pad the sequences to the same length as used in training
max_sequence_length = 200  # Ensure this matches your training settings
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
print("\nPadded sequences shape:", padded_sequences.shape)

# -------------------------------
# 4. Load the Trained LSTM Model and Predict
# -------------------------------
model = load_model("lstm_resume_classifier.h5")
print("Trained LSTM model loaded successfully.")

# Get prediction probabilities from the model
predictions = model.predict(padded_sequences)
print("\nPrediction probabilities (sample):")
print(predictions[:5])  # Display sample predictions

# Convert predictions to class indices
predicted_classes = predictions.argmax(axis=-1)

# Check predicted classes Values
print(predicted_classes)

# -------------------------------
# 5. Map Predictions to Labels and Display Results
# -------------------------------
# Define the label mapping (adjust based on your training labels)
labels = ['ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
       'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
       'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
       'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
       'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER']
new_data["Predicted_Category"] = [labels[i] for i in predicted_classes]



# Display a sample of the predictions
print("\nSample predictions:")
print(new_data[["Resume_str", "Predicted_Category"]].head())

# -------------------------------
# 6. Visualize the Distribution of Predicted Categories
# -------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(x="Predicted_Category", data=new_data, palette="viridis")
plt.xlabel("Predicted Category")
plt.ylabel("Count")
plt.title("Distribution of Predicted Resume Categories")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
