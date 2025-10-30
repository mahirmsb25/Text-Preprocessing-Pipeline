# Text Preprocessing Pipeline (TripAdvisor Reviews)

## Overview
This project implements a complete text preprocessing pipeline using Python’s NLTK and Pandas libraries.  
The pipeline transforms raw TripAdvisor hotel reviews into a clean, structured format suitable for Natural Language Processing (NLP) tasks such as sentiment analysis or topic modeling.

## Objectives
- Clean and normalize raw text data  
- Remove punctuation, symbols, and stopwords  
- Tokenize reviews into individual words  
- Apply stemming and lemmatization  
- Generate and analyze n-grams to identify language patterns

## Technologies Used
- Python 3  
- NLTK  
- Pandas  
- Regular Expressions (re)

## Installation
Clone the repository:
```
git clone https://github.com/<your-username>/text-preprocessing-pipeline.git
cd text-preprocessing-pipeline
```

Install dependencies:
```
pip install -r requirements.txt
```

Download NLTK resources (run once):
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Dataset
The project uses the TripAdvisor Hotel Reviews dataset available on [Kaggle]().  
Each entry contains a hotel review and its corresponding rating.

## How to Run
1. Place the dataset file `tripadvisor_hotel_reviews.csv` in the project directory.  
2. Run the script:
```
python preprocessing.py
```

The script will:
- Convert text to lowercase  
- Remove stopwords and punctuation  
- Tokenize, stem, and lemmatize the reviews  
- Generate unigrams, bigrams, and four-grams  
- Display sample outputs in the console

## Processing Workflow

| Step | Transformation | Example |
|------|----------------|----------|
| 1 | Lowercasing | "Clean Room" → "clean room" |
| 2 | Stopword Removal | "the hotel was not clean" → "hotel not clean" |
| 3 | Punctuation Removal | "hotel!" → "hotel" |
| 4 | Tokenization | "hotel not clean" → ['hotel', 'not', 'clean'] |
| 5 | Stemming | "cleaned" → "clean" |
| 6 | Lemmatization | "better" → "good" |
| 7 | n-Grams | "friendly staff" → ('friendly', 'staff') |

## Example Output

**Top 5 Unigrams:**
```
hotel        1500  
room         1423  
staff        1298  
good         1130  
clean        1027
```

**Top 5 Bigrams:**
```
(friendly, staff)     85  
(clean, room)         72  
(great, location)     68
```

## Author
**Mahir**
B.Tech in Artificial Intelligence, Shoolini University  