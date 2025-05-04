import os
from flask import Flask, request, render_template, flash, url_for
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/assets', exist_ok=True)

# Load the models
model_dnn = load_model('model_dnn1.h5')
model_cnn1 = load_model('model_cnn1.h5')
model_cnn2 = load_model('model_cnn2.h5')

# Genre mapping (adjust according to your model's output)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_mfcc(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=22050)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    
    # Transpose to match the model's expected input shape
    mfccs = mfccs.T
    
    # Pad or truncate to match the expected input shape (130, 13)
    if mfccs.shape[0] < 130:
        mfccs = np.pad(mfccs, ((0, 130 - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:130]
    
    return mfccs

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return render_template('index.html')
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Extract features
                mfccs = extract_mfcc(filepath)
                mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
                
                # Make predictions
                pred_dnn = model_dnn.predict(mfccs)
                pred_cnn1 = model_cnn1.predict(mfccs)
                pred_cnn2 = model_cnn2.predict(mfccs)
                
                # Get genre predictions
                genre_dnn = GENRES[np.argmax(pred_dnn[0])]
                genre_cnn1 = GENRES[np.argmax(pred_cnn1[0])]
                genre_cnn2 = GENRES[np.argmax(pred_cnn2[0])]
                
                # Get confidence scores
                conf_dnn = float(np.max(pred_dnn[0]) * 100)
                conf_cnn1 = float(np.max(pred_cnn1[0]) * 100)
                conf_cnn2 = float(np.max(pred_cnn2[0]) * 100)
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                return render_template('index.html', 
                                     genre_dnn=genre_dnn,
                                     genre_cnn1=genre_cnn1,
                                     genre_cnn2=genre_cnn2,
                                     conf_dnn=conf_dnn,
                                     conf_cnn1=conf_cnn1,
                                     conf_cnn2=conf_cnn2)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 