# -*- coding: utf-8 -*-
"""Created on Tue, 2021.06.29 FID BAUdigital | This Multi-class classification model with 3 categories is trained
with data to distinguish the following architectural drawing types: floor plan, elevation and section. """

# Import the library
from tensorflow import keras


# Initializing Convolutional Neural Network
def create_MultiClassifier():
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense

    # start via the add() method
    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    # choose the input-image's format to be 64x64 with 3 channels
    classifier.add(Conv2D(6, (3, 3), input_shape=(64, 64, 3), activation="relu"))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(4, 4)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(12, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten after Convolution
    classifier.add(Flatten())

    # Step 4 - Full connection, at least 3 units in last layer
    classifier.add(Dense(activation="softmax", units=3))

    # Compiling the CNN
    opt = keras.optimizers.Adam(learning_rate=0.01)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print a list of Keras Layers
    classifier.summary()

    return classifier


def main():
    from keras.preprocessing.image import ImageDataGenerator
    import datetime

    # Set training parameters
    initial_learning_rate = 0.01
    epochs = 10

    # perform data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      zoom_range=0.2)

    # load training data in batches
    training_data = train_datagen.flow_from_directory('data\Train',
        classes=['elevation', 'floor-plan', 'section'],
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    # load test data in batches
    test_data = test_datagen.flow_from_directory('data\Test',
        classes=['elevation', 'floor-plan', 'section'],
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    print(training_data.class_indices)
    print(test_data.class_indices)

    # Then start training the model
    classifier = create_MultiClassifier()

    # Callback Learning Rate scheduler
    def lr_time_based_scheduler(epoch, lr):
        decay = initial_learning_rate / epochs
        if epoch < 20:
            return lr
        else:
            return lr * 1 / (1 + decay * epoch)

    classifier_callbacks = [
        keras.callbacks.LearningRateScheduler(lr_time_based_scheduler),
        keras.callbacks.ProgbarLogger(count_mode="steps", stateful_metrics=None),
        keras.callbacks.TensorBoard(log_dir="./logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                    histogram_freq=1),  # Run with:tensorboard --logdir currentfolder (with logs)
        keras.callbacks.CSVLogger(f'training_log{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
                                  separator=",", append=False),
        keras.callbacks.ModelCheckpoint(filepath='fitted_classifier.hdf5',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        monitor='accuracy',
                                        mode='auto')
    ]
    # fit model to data
    history = classifier.fit(training_data,
                             steps_per_epoch=None,
                             epochs=epochs,
                             validation_data=test_data,
                             validation_steps=None,
                             callbacks=classifier_callbacks
                             )

    # Save the models final state
    def save_classifier():
        classifier.save(
            "MultiClassifier.hdf5",
            overwrite=True)

        print("Trained model successfully saved to disk!")

    # Visualize performance after training
    def model_performance_plotting():
        print(history.history.keys())

        from matplotlib import pyplot as plt
        # summarize history for accuracy
        plt.style.use("ggplot")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.title('Algorithm accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.savefig('model_accuracy.png')

        # summarize history for loss
        plt.style.use("ggplot")
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['val_loss'])
        plt.title('Algorithm loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.savefig('model_loss.png')

    # call the functions
    save_classifier()
    model_performance_plotting()


if __name__ == '__main__':
    main()
