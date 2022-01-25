"""Juli 2021 for FID BAUdigital at TU Darmstadt"""

'@author: Paul Steggemann (github@ Paulinos739)'

'This program tests a trained Classification CNN model on a list of floor plan images. '
'Images have to be located in the same folder'
""

# Import Dependencies
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd


# load trained classifier, hdf5-file in this case
def MultiClassifier():
    classifier = load_model(
        'your_trained_model.hdf5',
        compile=True)
    return classifier


def main():
    def floor_plan_prediction_single(predict=True):
        if predict:
            img_path = "validation\your_drawing.jpg"
            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = MultiClassifier().predict(img_array)
            print(prediction)

    def floor_plan_prediction_multiple(predict=True):
        if predict:
            # image folder
            folder_path = 'validation'
            # dimensions of images, given by the CNN input dim
            img_width, img_height = 64, 64

            # create a list and...
            images = []

            # ...stack up images to pass them for prediction
            for img in os.listdir(folder_path):
                img = os.path.join(folder_path, img)
                img = image.load_img(img, target_size=(img_width, img_height))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images.append(img)

            images = np.vstack(images)

            # call model and run inference
            predictions = MultiClassifier().predict(images)
            print(predictions)

            # calculating the prediction confidence in %
            from scipy.special import softmax
            confidence = softmax(predictions)
            confidence = np.around(confidence, decimals=2)

            # Then prepare the export as csv, json and xlsx
            filenames_list = os.listdir(folder_path)
            # lists of all prediction data
            lst1 = filenames_list
            lst2 = predictions
            lst3 = confidence

            # Call DataFrame constructor after zipping, with columns specified
            df = pd.DataFrame(list(zip(lst1, lst2, lst3)),
                              columns=['sample', 'label', 'confidence'])
            print(df)

            # Finally export to files
            df.to_csv("metadata\predictions.csv", index=False)
            df.to_json("metadata\predictions.json", orient='records')
            df.to_excel("metadata\predictions.xlsx")

    # Call functions
    floor_plan_prediction_single()
    floor_plan_prediction_multiple()


if __name__ == '__main__':
    main()


