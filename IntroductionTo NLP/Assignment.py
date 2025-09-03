"""

Perform Text Preprocessing on SMS_SpamCollection Dataset.  

"""

# Import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
# The dataset usually has two columns: 'label' (spam/ham) and 'message'
df = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])

print("First 5 rows of dataset:")
print(df.head())

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for preprocessing text
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove numbers and special characters
    text = re.sub(r'[^a-z]', ' ', text)
    # 3. Tokenization (split into words)
    words = text.split()
    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]
    # 5. Stemming
    words = [ps.stem(word) for word in words]
    # 6. Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    # 7. Join back into sentence
    return " ".join(words)

# Apply preprocessing
df["cleaned_message"] = df["message"].apply(preprocess_text)

# Display processed dataset
print("\nAfter Preprocessing:")
print(df.head())

# Save processed data
df.to_csv("Preprocessed_SMS.csv", index=False)
print("\nPreprocessed dataset saved as Preprocessed_SMS.csv")
