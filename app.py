import sys

# CRITICAL: Monkey-patch Keras BEFORE any imports to fix layer names with '/' for Keras 3.x
# This must be done first before tensorflow/keras imports anything
def patch_keras_operation():
    """Intercept Keras Operation class to automatically fix layer names"""
    try:
        # Import and patch Operation class
        from keras.src.ops.operation import Operation
        
        # Store the original __init__
        _original_init = Operation.__init__
        
        # Create wrapper that fixes names
        def _patched_init(self, name=None, **kwargs):
            # Fix name if it contains '/'
            if name and isinstance(name, str) and '/' in name:
                name = name.replace('/', '_')
            # Call original with fixed name
            _original_init(self, name=name, **kwargs)
        
        # Apply the patch
        Operation.__init__ = _patched_init
        return True
    except Exception as e:
        print(f"Warning: Could not patch Keras Operation: {e}")
        return False

# Apply patch immediately
patch_keras_operation()

from tensorflow.keras import utils as keras_utils
import types

# Patch for keras-vggface compatibility with TF 2.x
# Patch 'keras.utils.data_utils' which is missing in newer TF
if not hasattr(keras_utils, 'data_utils'):
    keras_utils.data_utils = types.ModuleType("keras.utils.data_utils")
    sys.modules['keras.utils.data_utils'] = keras_utils.data_utils

if not hasattr(keras_utils.data_utils, 'get_file'):
    from tensorflow.keras.utils import get_file
    keras_utils.data_utils.get_file = get_file

# Patch 'keras.utils.layer_utils'
if not hasattr(keras_utils, 'layer_utils'):
    keras_utils.layer_utils = types.ModuleType("keras.utils.layer_utils")
    sys.modules['keras.utils.layer_utils'] = keras_utils.layer_utils

if not hasattr(keras_utils.layer_utils, 'get_source_inputs'):
    # Try to find get_source_inputs
    if hasattr(keras_utils, 'get_source_inputs'):
        keras_utils.layer_utils.get_source_inputs = keras_utils.get_source_inputs
    else:
        try:
            from tensorflow.keras.utils import get_source_inputs
            keras_utils.layer_utils.get_source_inputs = get_source_inputs
        except ImportError:
            pass

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

if not os.path.exists('embedding.pkl') or not os.path.exists('filenames.pkl'):
    st.error("Error: 'embedding.pkl' or 'filenames.pkl' not found. Please run 'feature_extractor.py' first.")
    st.stop()

feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if not results:
        # Fallback if no face detected in the immediate crop, or return None
        # For now, let's just return None or handle it gracefully
        return None

    x, y, width, height = results[0]['box']
    
    # Fix negative coordinates or out of bounds
    x = max(0, x)
    y = max(0, y)

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        
        if features is not None:
            # recommend
            index_pos = recommend(feature_list,features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            # display
            col1,col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image)
            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos],width=300)
        else:
            st.error("No face detected in the uploaded image. Please try another one.")

