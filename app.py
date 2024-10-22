import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


def split_into_sentences(text):
    """
    Split the text into sentences using regular expressions.
    """
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences


def check_plagiarism(text):
    """
    Check each sentence for plagiarism and return the highlighted text.
    """
    sentences = split_into_sentences(text)
    highlighted_text = ""

    for sentence in sentences:
        vectorized_sentence = tfidf_vectorizer.transform([sentence])
        result = model.predict(vectorized_sentence)
        is_plagiarized = (result[0] == 1)

        if is_plagiarized:
            # Highlight plagiarized sentences
            highlighted_text += f"<span style='background-color: yellow;'>{sentence}</span> "
        else:
            highlighted_text += sentence + " "

    return highlighted_text


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/detect", methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']

    # Get highlighted text
    highlighted_text = check_plagiarism(input_text)

    # Display result based on the presence of highlighted text
    result = "Plagiarism Detected" if 'background-color: yellow;' in highlighted_text else "No Plagiarism"

    return render_template('index.html', result=result, text=highlighted_text)


if __name__ == "__main__":
    app.run(debug=True)
