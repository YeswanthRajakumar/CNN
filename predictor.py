# load and evaluate a saved model

import numpy as np
from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
 
# load model
classifier = load_model('my_classifier_model.h5')
# summarize model.
classifier.summary()
# load dataset



# Testing Neural Network

test_image = image.load_img('dog_random.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = classifier.predict(test_image)
print(result)

if result[0][0] >=0.5:
	prediction ='dog'
else:
	prediction ='cat'

print(prediction)

