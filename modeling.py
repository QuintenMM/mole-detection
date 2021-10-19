import pickle
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPool2D


def instantiate_model(model_choice, X_train, X_val, y_train, y_val):
    if model_choice == 0:
        history_hist, model = instantiate_model_v1_resnet(X_train, X_val, y_train, y_val)
        return history_hist, model

    elif model_choice == 1:
        history_hist, model = instantiate_model_v2_jacques(X_train, X_val, y_train, y_val)
        return history_hist, model


# def instantiate_model(X_train, X_val, y_train, y_val):
def instantiate_model_v1_resnet(X_train, X_val, y_train, y_val):
    model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(model)
    new_model.add(BatchNormalization())
    new_model.add(Flatten())
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(7, activation='softmax'))

    new_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = new_model.fit(X_train, y_train,
                            epochs=3,
                            batch_size=16,
                            validation_data=(X_val, y_val)
                            )

    new_model.save('model/v1_resnet.h5')
    with open('model/history', 'wb') as filepath:
        pickle.dump(history.history, filepath)

    return history.history, new_model


def instantiate_model_v2_jacques(X_train, X_val, y_train, y_val):
    early_stopping_monitor = EarlyStopping(patience=3)

    base_model = Sequential()
    base_model.add(Conv2D(7, kernel_size=4, activation='relu', input_shape=(224, 224, 3)))
    base_model.add(MaxPool2D(2))
    base_model.add(BatchNormalization())
    base_model.add(Flatten())
    base_model.add(Dense(64, activation='relu'))
    base_model.add(Dropout(0.25))
    base_model.add(Dense(35, activation='relu'))
    base_model.add(Dense(7, activation='softmax'))

    base_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    history = base_model.fit(X_train, y_train,
                             epochs=3,
                             batch_size=16,
                             validation_data=(X_val, y_val),
                             callbacks=[early_stopping_monitor])

    base_model.save('model/v2_jacques.h5')
    with open('model/v2_history', 'wb') as filepath:
        pickle.dump(history.history, filepath)

    return history.history, base_model
