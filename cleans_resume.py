import pandas as pd
import re

# Define a basic set of English stopwords
stopwords = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 
    'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 
    'yourself', 'yourselves'
}

def simple_preprocess(text):
    """Clean and preprocess resume text using regex and simple tokenization."""
    if not isinstance(text, str):
        return ""
    # 1. Lowercase the text
    text = text.lower()
    # 2. Remove HTML tags (if any)
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Remove special characters and numbers (retain alphabets and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Tokenize using split (since nltk.word_tokenize isn't working)
    tokens = text.split()
    # 6. Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # (Optional) Further steps like rudimentary lemmatization could be added here
    return " ".join(tokens)

# Load your dataset (adjust the file path as needed)
df = pd.read_csv("resume_dataset.csv")
df.columns = df.columns.str.strip()  # Remove any accidental leading/trailing spaces

# Ensure the 'Resume_str' column is a string and drop rows with missing data
df['Resume_str'] = df['Resume_str'].astype(str)
df = df.dropna(subset=['Resume_str'])

# Apply the preprocessing function to the 'Resume_str' column
df['cleaned_resume'] = df['Resume_str'].apply(simple_preprocess)

# Save the cleaned data to a new CSV file
df.to_csv("cleaned_resume_processed.csv", index=False)

print("Preprocessing complete!")
