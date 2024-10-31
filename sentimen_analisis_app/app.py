from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = "" 
    message = ""
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        if prediction[0] == 2:
            result = 'Positif'
        elif prediction[0] == 1:
            result = 'Netral'
        else:
            result = 'Negatif'
    return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
