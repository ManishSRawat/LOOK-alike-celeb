import sys
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

# Patch keras-vggface for Keras 3.x compatibility (layer names cannot contain '/')
def patch_keras_vggface():
    """Patch keras-vggface models.py to replace '/' with '_' in layer names for Keras 3.x compatibility"""
    try:
        import keras_vggface
        import os
        import re
        
        # Find the models.py file
        vggface_path = os.path.dirname(keras_vggface.__file__)
        models_file = os.path.join(vggface_path, 'models.py')
        
        if not os.path.exists(models_file):
            return False
            
        # Read the file
        with open(models_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if 'conv1_7x7_s2' in content and 'conv1/7x7_s2' not in content:
            return True  # Already patched
        
        # Replace all '/' with '_' in layer names
        # Pattern: name='something/something' or name="something/something"
        pattern = r"name=(['\"])([^'\"]*)/([^'\"]*)\1"
        
        # Keep replacing until no more matches
        while re.search(pattern, content):
            content = re.sub(pattern, lambda m: f"name={m.group(1)}{m.group(2)}_{m.group(3)}{m.group(1)}", content)
        
        # Also handle concatenated strings like: + "/bn"
        content = re.sub(r'\+ \"(/[^\"]*)\"', lambda m: f'+ "{m.group(1).replace("/", "_")}"', content)
        
        # Write back
        with open(models_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Warning: Could not patch keras-vggface: {e}")
        return False

# Apply the patch before importing VGGFace
patch_keras_vggface()

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

