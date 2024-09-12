from flask import Flask,request,render_template_string
from sklearn.neighbors import NearestNeighbors
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow

with open('model_tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('next_word_model_2.keras')

def predict_next_word(sentence,model,tokenize,max_len):
  token = tokenize.texts_to_sequences([sentence])[0]
  if len(token) > max_len:
    token = token[-(max_len-1):]
  padded_token = pad_sequences([token],maxlen=max_len-1)
  logits = model.predict(padded_token)
  pred = tensorflow.nn.softmax(logits).numpy()
  pred = pred.argmax(axis=1)
  for word,index in tokenize.word_index.items():
    if index == pred:
      return word
  return 'no word found'

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():
    next_word = ''
    if request.method == 'POST':
        input_text = request.form['input_text']
        max_length = model.input_shape[1] + 1
        next_word = predict_next_word(input_text, model, tokenizer, max_length)
    
    return render_template_string('''
        <!doctype html>
        <title>Next Word Prediction</title>
        <h1>Next Word Prediction</h1>
        <form method=post>
        <label for="input_text">Enter the sentence:</label>
        <input type=text name=input_text value="{{ request.form['input_text'] if request.form else 'And I am sicke at' }}">
        <input type=submit value=Predict>
        </form>
        <p>Next Word: {{ next_word }}</p>
    ''', next_word=next_word)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)