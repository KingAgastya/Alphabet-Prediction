import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import PIL.ImageOps
from PIL import Image

X = np.load('image.npz')['arr_0']
y = pd.read_csv('data.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 7500, test_size = 2500, random_state = 42)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial')
classifier = classifier.fit(x_train_scaled, y_train)

def get_prediction(image):

    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    generic_long_variable_name = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    generic_long_variable_name = np.asarray(generic_long_variable_name)/max_pixel
    test_sample = np.array(generic_long_variable_name).reshape(1, 784)
    test_pred = classifier.predict(test_sample)
    return(test_pred[0])
