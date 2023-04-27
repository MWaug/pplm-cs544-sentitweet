import pandas as pd
import re
import contractions
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define a function to clean and preprocess text
def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove HTML and URLs
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+', '', text)
    
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    
    # Lemmatize the tokens
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word, pos='v') for word in tokens]
    
    # Join the cleaned tokens back into a single string
    clean_text = ' '.join(tokens)
    
    # Remove excess whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


def main():
    # Import punkt resource 
    # nltk.download('punkt')

    # Load the csv file into a pandas dataframe
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])

    # Replace 0 with 3 and 4 with 2 in the target column
    df['target'].replace({0: 3, 4: 2}, inplace=True)

    # Drop the columns that are not needed
    df.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)

    # Clean and preprocess the text column
    df['text'] = df['text'].apply(preprocess_text)

    # Print the first few rows of the resulting dataframe
    print(df.head())

    # Save the preprocessed dataframe to a CSV file
    df.to_csv('dataset_preprocessed.tsv', header=False, index=False, sep="\t")
    

if __name__ == '__main__':
    main()