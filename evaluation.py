import numpy as np
import tensorflow
from sklearn.metrics import classification_report


def print_accuracy_classificationreport_confusionmmatrix(y_test_one_column, y_pred,
                                                         model, X_test, y_test):
    results = model.evaluate(X_test, y_test, batch_size=20)
    print(f"Accuracy on test set is {results[1] * 100:.2f}%")
    print(classification_report(y_test_one_column, y_pred))
    print(tensorflow.math.confusion_matrix(y_test_one_column, y_pred))
