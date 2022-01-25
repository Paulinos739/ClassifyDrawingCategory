# -*- coding: utf-8 -*-
"""
Created on Tue, 2021.06.29
FID BAUdigital | This Multi-class classification model with 3 categories, is trained with data to distinguish the
following architectural drawing types: floor plan, elevation and section.
"""


# Initializing the Neural Network
def load_MultiClassifier():
    from tensorflow import keras
    from keras.models import load_model

    # load pretrained model
    classifier = load_model(
        "trained_model", compile=False)

    # Compiling the CNN
    opt = keras.optimizers.Adam(learning_rate=0.01)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


def main():
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator
    import datetime

    # Set some stuff
    initial_learning_rate = 0.01
    epochs = 400

    # perform data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      zoom_range=0.2)

    # load training data in batches
    training_data = train_datagen.flow_from_directory(
        'data\Train',
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    # load test data in batches
    test_data = test_datagen.flow_from_directory(
        'data\Test',
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    # start computation and train the model
    classifier = load_MultiClassifier()

    # Define Learning Rate Scheduler
    def lr_time_based_scheduler(epoch, lr):
        decay = initial_learning_rate / epochs
        if epoch < 10:
            return lr
        else:
            return lr * 1 / (1 + decay * epoch)

    # Initialize callbacks
    classifier_callbacks = [
        keras.callbacks.LearningRateScheduler(lr_time_based_scheduler),
        keras.callbacks.CSVLogger(f'training_log{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv', separator=",", append=False),
        # Run Command: tensorboard --logdir logs currentfolder
        keras.callbacks.TensorBoard(log_dir="./logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1),
        keras.callbacks.ModelCheckpoint(filepath='fitted_classifier\multi_class.hdf5',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        monitor='accuracy',
                                        mode='auto')
    ]
    # Continue to fit the model to data
    history = classifier.fit(training_data,
                             steps_per_epoch=None,
                             epochs=epochs,
                             validation_data=test_data,
                             validation_steps=None,
                             callbacks=classifier_callbacks)

    # Save trained CNN to a HDF file
    def save_classifier(save_h5=True):
        if save_h5:
            classifier.save(
                "MultiClassifier.hdf5",
                overwrite=True
            )
            print("Trained model successfully saved to disk!")
        else:
            print("model was not saved to disk!")

    # Visualize the performance after training
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
