# 🏷️ Text-to-Tag Generator

A machine learning-based web application that automatically generates relevant tags from input text using Natural Language Processing (NLP) techniques.

This project uses **TF-IDF vectorization** and **cosine similarity** to recommend the most relevant tags for a given text.

---

## 🚀 Features

- 🔍 Generate tags from any input text
- 🧠 NLP preprocessing (tokenization, stopword removal, lemmatization)
- ⚡ TF-IDF + cosine similarity-based tag prediction
- 🌐 Interactive web interface using Flask
- 🔄 Dynamic model rebuild using `tags.csv`
- 📊 Supports evaluation using dataset (Precision@K)

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn
- **NLP:** NLTK
- **Frontend:** HTML, CSS
- **Libraries:** NumPy, Pandas

---

## 📂 Project Structure
text-to-tag-generator/
│
├── flask_app.py # Flask web application
├── text_to_tag.py # NLP + ML model logic
├── tags.csv # Tag definitions (tag, description)
├── dataset.csv # Optional dataset for evaluation
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/
└── styles.css # Styling

▶️ Run the Application
python flask_app.py

Then open in browser:LOCAL HOST
http://127.0.0.1:5000/

🧪 How It Works
1.Preprocessing
-Lowercasing
-Removing punctuation, URLs
-Stopword removal
-Lemmatization
2.Vectorization
-TF-IDF applied on tag descriptions
3.Prediction
-Cosine similarity between input text and tag vectors
-Top-K relevant tags returned

💡 Usage
Web UI
-Enter text
-Select:
Top K tags
-Minimum score
Click Predict

API (JSON)
POST /predict

Example:
{
  "text": "Machine learning and AI are transforming industries",
  "top_k": 5
}

Response:
{
  "tags": [
    {"tag": "machine learning", "score": 0.85},
    {"tag": "data science", "score": 0.72}
  ]
}

📊 Evaluation (Optional)
Supports dataset evaluation using:
dataset.csv
Metric: Precision@K

📌 Future Improvements
🚀 Use BERT / Transformers for better accuracy
🌐 Deploy on cloud (Render / AWS)
📱 Improve UI/UX
🔌 Add REST API documentation
📈 Add more evaluation metrics
