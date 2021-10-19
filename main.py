import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


from utils import data_preprocessing as dp, modeling as md, plotting, evaluation as ev


if __name__ == '__main__':
    metadata_labels = pd.read_csv('csv_files/metadata_labels.csv')

    # variables
    images_paths = ["data/HAM10000_images_part_1/"]
    image_size = (224, 224)
    limit = 2000
    starting_row_part_two = 5000

    # choose a model you want to train and save
    model_list = [('v1_resnet', 'v1_history'), ('v2_jacques', 'v2_history')]
    choice = 1

    # prepare path and names, based on model choice
    model_choice = model_list[choice]
    model_name = model_choice[0] + '.h5'
    history_name = model_choice[1]
    model_path = os.path.join('model', model_name)
    history_path = os.path.join('model', history_name)

    # get preprocessed images
    images_array_lst = []
    for path in images_paths:
        if choice == 0:
            images_array = dp.get_preprocessed_images_transfer_learning(path, image_size, limit)
            images_array_lst.append(images_array)
        elif choice == 1:
            images_array = dp.get_preprocessed_images_no_tl(path, limit)    # image_size already defined in function
            images_array_lst.append(images_array)
    skin_images = np.vstack(images_array_lst)

    # data augmentation


    # divide X and y, one hot encode
    X = skin_images
    target = metadata_labels.dx[:X.shape[0]]
    target_one_hot = tf.keras.utils.to_categorical(target, num_classes=7, dtype="int")

    X_train, X_val, y_train, y_val, X_test, y_test = dp.split_in_train_val_test(X, target_one_hot)

    # train model if model does not exist
    try:
        model = load_model(model_path)
        history_hist = pickle.load(open(history_path, 'rb'))
    except:
        history_hist, model = md.instantiate_model(choice, X_train, X_val, y_train, y_val)

    plotting.plot_history(history_hist)

    # evaluate
    # invert one hot encoding
    y_test_one_column = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    ev.print_accuracy_classificationreport_confusionmmatrix(y_test_one_column,
                                                y_pred, model, X_test, y_test)

    cm = confusion_matrix(y_test_one_column, y_pred)
    classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
    plotting.plot_confusion_matrix(choice, cm, classes, normalize=True)
