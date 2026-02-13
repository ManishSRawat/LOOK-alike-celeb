# Celebrity Look-Alike Finder 🎭

A Streamlit web application that finds your Bollywood celebrity look-alike using deep learning and facial recognition.

## 🌟 Features

- **Face Detection**: Uses MTCNN for accurate face detection
- **Deep Learning**: VGGFace ResNet50 model for facial feature extraction
- **100 Celebrities**: Compare against 100 Bollywood celebrities
- **Real-time Results**: Instant celebrity matching with similarity scores
- **User-Friendly**: Simple drag-and-drop interface

## 🚀 Demo

Upload a photo with a clear face, and the app will find which Bollywood celebrity you look like!

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/celebrity-lookalike.git
cd celebrity-lookalike
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Celebrity Embeddings (First Time Only)

This step processes all celebrity images and creates the facial feature database. It takes approximately 30-45 minutes.

```bash
python feature_extractor.py
```

This will create two files:
- `embedding.pkl` - Celebrity facial feature embeddings (~68 MB)
- `filenames.pkl` - Paths to celebrity images

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the App

1. Click "Choose an image" to upload a photo
2. Make sure the photo has a clear, front-facing face
3. Wait for processing (a few seconds)
4. See your celebrity match!

## 📁 Project Structure

```
celebrity-lookalike/
├── app.py                    # Main Streamlit application
├── feature_extractor.py      # Script to generate embeddings
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── data/                     # Celebrity images (100 folders)
│   ├── Aamir_Khan/
│   ├── Shah_Rukh_Khan/
│   └── ...
├── uploads/                  # Temporary user uploads
├── embedding.pkl             # Pre-computed embeddings (generated)
└── filenames.pkl             # Image paths (generated)
```

## 🛠️ Technical Details

### Architecture

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Feature Extraction**: VGGFace ResNet50 (pre-trained on celebrity faces)
- **Similarity Metric**: Cosine similarity
- **Frontend**: Streamlit
- **Backend**: TensorFlow/Keras

### How It Works

1. **Face Detection**: MTCNN detects and crops the face from the uploaded image
2. **Feature Extraction**: VGGFace ResNet50 extracts a 2048-dimensional feature vector
3. **Similarity Matching**: Compares the feature vector against 8,664 celebrity embeddings using cosine similarity
4. **Result Display**: Shows the best matching celebrity with their image

## 🐛 Troubleshooting

### "No face detected in the uploaded image"
- Ensure the image has a clear, front-facing face
- Try a photo with good lighting
- Avoid images with multiple faces or obscured faces

### Application won't start
- Verify virtual environment is activated
- Check that `embedding.pkl` and `filenames.pkl` exist
- Run `pip install -r requirements.txt` again

### Slow processing
- This is normal - each prediction takes ~150-200ms
- First run may be slower as models are loaded

## 📊 Dataset

The application uses a dataset of 100 Bollywood celebrities with approximately 86 images per celebrity (8,664 total images).

## 🔮 Future Enhancements

- [ ] Support for multiple faces in one image
- [ ] Confidence score display
- [ ] Top 5 celebrity matches
- [ ] Custom celebrity database support
- [ ] Batch processing mode
- [ ] Mobile app version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VGGFace model by [rcmalli](https://github.com/rcmalli/keras-vggface)
- MTCNN implementation
- Streamlit framework
- TensorFlow/Keras

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/YOUR_USERNAME/celebrity-lookalike](https://github.com/YOUR_USERNAME/celebrity-lookalike)

---

⭐ If you found this project helpful, please give it a star!
