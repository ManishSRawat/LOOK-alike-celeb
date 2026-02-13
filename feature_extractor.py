import os
import pickle
import sys
import types
from tensorflow.keras import utils as keras_utils

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
            # Fallback for very new TF where it looks different or resides elsewhere
            pass

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm

# Ensure data directory exists or handle it (assuming data/ exists as per original code)
if not os.path.exists('data'):
    print("Error: 'data' directory not found. Please create a 'data' directory and add subdirectories with celebrity images.")
    exit()

actors = os.listdir('data')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))

pickle.dump(filenames,open('filenames.pkl','wb'))

filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
print(model.summary())

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))
