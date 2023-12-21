import os
import sys
import tensorflow as tf
from src.utils import save_model
from src.exception import CustomException
from src.logger import logging
from keras.preprocessing.image import ImageDataGenerator

class ModelTrainerConfig:
    gender_trained_model_file_path = os.path.join("artifacts", "gender_model")
    age_trained_model_file_path = os.path.join("artifacts", "age_model")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, data_path):
        try:
            logging.info("Splitting data into gender and age model data.")
            Gender_data = data_path['gender_data']
            X_gender_train, X_gender_test, y_gender_train, y_gender_test = Gender_data[0], Gender_data[1], Gender_data[2], Gender_data[3]
            Age_data = data_path['age_data']
            X_age_train, X_age_test, y_age_train, y_age_test = Age_data[0], Age_data[1], Age_data[2], Age_data[3]
            gender_train, gender_test = self.get_augmented_data(X_gender_train, X_gender_test, y_gender_train, y_gender_test)
            gender_model = self.build_gender_model()
            logging.info("Gender model built.")
            gender_history = gender_model.fit(
                gender_train,
                validation_data=gender_test,
                epochs=25,
                shuffle=True,
                verbose=1
            )
            logging.info("Training data fitted for Gender model.")
            save_model(
                ModelTrainerConfig.gender_trained_model_file_path,
                gender_model
            )
            logging.info("Saved gender model.")
            age_train, age_test = self.get_augmented_data(X_age_train, X_age_test, y_age_train, y_age_test)
            age_model = self.build_age_model()
            logging.info("Age model built.")
            age_history = age_model.fit(
                age_train,
                validation_data=age_test,
                epochs = 40,
                shuffle=True,
                verbose=1
            )
            logging.info("Training data fitted for Age model.")
            save_model(
                ModelTrainerConfig.age_trained_model_file_path,
                age_model
            )
            logging.info("Saved age model.")

        except Exception as e:
            logging.info("Custom Exception raised with error:\t.".format(e))
            raise CustomException(e, sys)

    def build_age_model(self):
        agemodel = tf.keras.models.Sequential()
        agemodel.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
        agemodel.add(tf.keras.layers.MaxPooling2D((2,2)))
        agemodel.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        agemodel.add(tf.keras.layers.MaxPooling2D((2,2)))
        agemodel.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
        agemodel.add(tf.keras.layers.MaxPooling2D((2,2)))
        agemodel.add(tf.keras.layers.Flatten())
        agemodel.add(tf.keras.layers.Dense(64, activation='relu'))
        agemodel.add(tf.keras.layers.Dropout(0.5))
        agemodel.add(tf.keras.layers.Dense(1, activation='relu'))
        agemodel.compile(loss='mean_squared_error',
                    optimizer='adam')
        return agemodel
    
    def build_gender_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model
    
    def get_augmented_data(self, x_train, x_test, y_train, y_test ):
        datagen = ImageDataGenerator(
            rescale=1./255., width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True
        )
        train = datagen.flow(x_train, y_train, batch_size=32)
        test_datagen = ImageDataGenerator(rescale=1./255)
        test = test_datagen.flow(x_test, y_test, batch_size=32)
        return (
            train,
            test
        )