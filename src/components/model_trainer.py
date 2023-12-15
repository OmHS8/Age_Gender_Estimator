import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from src.utils import load_model, save_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    gender_trained_model_file_path = os.path.join("artifacts", "gender_model.pkl")
    age_trained_model_file_path = os.path.join("artifacts", "age_model.pkl")

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
            gender_model = self.build_models(1, activation='sigmoid', loss='binary_crossentropy')
            logging.info("Gender model built.")
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
            gender_history = gender_model.fit(
                X_gender_train,
                y_gender_train,
                validation_split=0.2,
                batch_size=64,
                epochs=25,
                callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), callback],
                verbose=2
            )
            logging.info("Training data fitted for Gender model.")
            gender_acc = gender_model.evaluate(X_gender_test, y_gender_test)[1]
            print("The accuracy for gender model is:\t {}".format(gender_acc))
            logging.info("Gender model evaluated and accuracy is\t.".format(gender_acc))
            save_model(
                ModelTrainerConfig.gender_trained_model_file_path,
                gender_model
            )
            logging.info("Saved gender model.")

            age_model = self.build_models(7, activation='softmax', loss='sparse_categorical_crossentropy')
            logging.info("Age model built.")
            age_history = age_model.fit(
                X_age_train,
                y_age_train,
                validation_split=0.2,
                batch_size=64,
                epochs=50,
                callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), callback],
                verbose=2
            )
            logging.info("Training data fitted for Age model.")
            age_acc = age_model.evaluate(X_age_test, y_age_test)[1]
            print("The accuracy for gender model is:\t {}".format(age_acc))
            logging.info("Age model evlauted and accuracy is\t.".format(age_acc))
            save_model(
                ModelTrainerConfig.age_trained_model_file_path,
                age_model
            )
            logging.info("Saved age model.")

        except Exception as e:
            logging.info("Custom Exception raised with error:\t.".format(e))
            raise CustomException(e, sys)

    def build_models(self, num_classes, activation='softmax', loss='sparse_categorical_crossentropy'):
        img_height = 48
        img_width = 48
        inputs = tf.keras.Input(shape=(img_height, img_width, 1))
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = tf.keras.layers.Conv2D(16, 3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer = 'adam',
            loss = loss,
            metrics = ['accuracy']
        )
        return model