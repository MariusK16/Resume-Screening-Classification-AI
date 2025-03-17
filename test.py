import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the trained LSTM model
model = load_model("lstm_resume_classifier.h5")
print("Model loaded successfully!")


# Load and inspect new resume dataset
new_resumes = pd.read_csv("res1 (1).csv")  # Adjust path if needed
print("New data loaded:", new_resumes.shape)

# Preview the first few resumes
print(new_resumes.head())


# Text Preprocessing 
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()


# Apply text cleaning
new_resumes["cleaned_resume"] = new_resumes["Resume_str"].apply(preprocess_text)

print("Text preprocessing completed!")


# Tokenization & Padding (should match training settings)
tokenizer = Tokenizer()  # Load your actual tokenizer if saved
tokenizer.fit_on_texts(new_resumes["cleaned_resume"])

# Convert to sequences
new_sequences = tokenizer.texts_to_sequences(new_resumes["cleaned_resume"])

# Pad sequences (should match max_length from training)
max_length = 500
new_X = pad_sequences(new_sequences, maxlen=max_length)

print("Tokenization & padding done!")


# Make Predictions
predictions = model.predict(new_X)

# Convert probabilities to class labels
new_resumes["Predicted_Category"] = (predictions > 0.5).astype(int)  # Adjust threshold if needed

print("Predictions made!")
print(new_resumes[["Resume_str", "Predicted_Category"]].head())  # Show some results


# Visualization of Predictions
plt.figure(figsize=(8, 6))
sns.countplot(x=new_resumes["Predicted_Category"], palette="viridis")
plt.xlabel("Predicted Category")
plt.ylabel("Count")
plt.title("Distribution of Predicted Resume Categories")
plt.xticks([0, 1], ["Category 0", "Category 1"])  # Adjust based on actual categories
plt.show()

print("Results visualized with a bar chart!")
