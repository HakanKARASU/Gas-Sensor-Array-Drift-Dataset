import pandas as pd
import glob
import os
import logging
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GasSensorModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.gas_types = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']
        self.gas_label_mapping = {gas: i for i, gas in enumerate(self.gas_types, 1)}
        self.df_all = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.history = None
        self.nn_classifier = None

    def process_data(self, file_path, batch_id):
        """Process individual data files to extract features and gas labels."""
        data_list = []
        with open(file_path, 'r') as file:
            for line in file:
                gas_label, rest = line.strip().split(';')
                gas_label_name = self.gas_types[int(gas_label) - 1]
                concentration, features = rest.split(' ', 1)
                entry_dict = {'Batch ID': batch_id, 'gas_label': gas_label_name, 'concentration': float(concentration)}
                for feature in features.split():
                    index, value = feature.split(':')
                    entry_dict[f'feature_{index}'] = float(value)
                data_list.append(entry_dict)
        return pd.DataFrame(data_list)

    def load_and_process_data(self, num_files=10):
        """Load and process the gas sensor data."""
        all_files = glob.glob(os.path.join(self.data_path, "*.dat"))[
                    :num_files]  # Process only the first num_files files
        df_all = pd.DataFrame()

        for i, file in enumerate(all_files, start=1):
            df_batch = self.process_data(file, batch_id=i)
            df_all = pd.concat([df_all, df_batch], ignore_index=True)

        # Move 'Batch ID' column to the first position
        df_all = df_all[['Batch ID'] + [col for col in df_all.columns if col != 'Batch ID']]

        # Sort by gas_label and reindex
        df_all.sort_values(by=['gas_label'], inplace=True)
        df_all.reset_index(drop=True, inplace=True)

        # Map the gas labels to numerical values
        df_all['gas_label_num'] = df_all['gas_label'].map(self.gas_label_mapping)

        self.df_all = df_all
        logger.info(f"Processed data loaded successfully. Shape: {self.df_all.shape}")

    def save_data_to_csv(self, csv_path):
        """Save the processed DataFrame to a CSV file."""
        if self.df_all is not None:
            self.df_all.to_csv(csv_path, index=False)
            logger.info(f"Data saved to {csv_path}")
        else:
            logger.error("No processed data to save. Load and process data first.")

    def prepare_data_for_ml(self):
        """Prepare data for machine learning: handle missing values, standardize features, split into train/val/test sets."""
        if self.df_all is not None:
            # Define X and y
            X = self.df_all[[col for col in self.df_all.columns if col.startswith('feature_')]]
            y = self.df_all['gas_label']  # Use gas_label instead of gas_label_num

            # Use LabelEncoder to encode gas labels to integers
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Shuffle the data
            X, y = shuffle(X, y, random_state=0)

            # Handle missing values and standardize features
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            # Split the data into training, validation, and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)

            # Standardize the features
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_val = sc_X.transform(X_val)
            X_test = sc_X.transform(X_test)

            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            # Logging sizes of datasets
            logger.info(f"Training set size: {self.X_train.shape[0]} samples")
            logger.info(f"Validation set size: {self.X_val.shape[0]} samples")
            logger.info(f"Test set size: {self.X_test.shape[0]} samples")

            logger.info("Data prepared successfully for machine learning.")
        else:
            logger.error("No processed data available. Load and process data first.")

    def train_knn_classifier_with_tuning(self):
        """Train KNN classifier with hyperparameter tuning."""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        knn_classifier = KNeighborsClassifier()
        grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logger.info(f"Best Parameters for KNN: {best_params}")

        knn_classifier = grid_search.best_estimator_
        knn_classifier.fit(self.X_train, self.y_train)

        y_pred_knn = knn_classifier.predict(self.X_test)
        knn_cm = confusion_matrix(self.y_test, y_pred_knn)
        knn_accuracy = accuracy_score(self.y_test, y_pred_knn)
        knn_misclassification = 1 - knn_accuracy  # Calculate misclassification error
        logger.info(f"KNN Confusion Matrix:\n{knn_cm}")
        logger.info(f"KNN Accuracy: {knn_accuracy:.4f}")
        logger.info(f"KNN Misclassification Error: {knn_misclassification:.4f}")

        return knn_classifier, knn_accuracy, knn_misclassification

    def train_svm_classifier_with_tuning(self):
        """Train SVM classifier with hyperparameter tuning."""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
        svm_classifier = SVC()
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logger.info(f"Best Parameters for SVM: {best_params}")

        svm_classifier = grid_search.best_estimator_
        svm_classifier.fit(self.X_train, self.y_train)

        y_pred_svm = svm_classifier.predict(self.X_test)
        svm_cm = confusion_matrix(self.y_test, y_pred_svm)
        svm_accuracy = accuracy_score(self.y_test, y_pred_svm)
        svm_misclassification = 1 - svm_accuracy  # Calculate misclassification error
        logger.info(f"SVM Confusion Matrix:\n{svm_cm}")
        logger.info(f"SVM Accuracy: {svm_accuracy:.4f}")
        logger.info(f"SVM Misclassification Error: {svm_misclassification:.4f}")

        return svm_classifier, svm_accuracy, svm_misclassification

    def train_random_forest_classifier_with_tuning(self):
        """Train Random Forest classifier with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_classifier = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logger.info(f"Best Parameters for RandomForest: {best_params}")

        rf_classifier = grid_search.best_estimator_
        rf_classifier.fit(self.X_train, self.y_train)

        y_pred_rf = rf_classifier.predict(self.X_test)
        rf_cm = confusion_matrix(self.y_test, y_pred_rf)
        rf_accuracy = accuracy_score(self.y_test, y_pred_rf)
        rf_misclassification = 1 - rf_accuracy  # Calculate misclassification error
        logger.info(f"RandomForest Confusion Matrix:\n{rf_cm}")
        logger.info(f"RandomForest Accuracy: {rf_accuracy:.4f}")
        logger.info(f"RandomForest Misclassification Error: {rf_misclassification:.4f}")

        return rf_classifier, rf_accuracy, rf_misclassification

    def train_neural_network_classifier(self, batch_size=10, epochs=100):
        """Train Neural Network classifier."""
        nn_classifier = Sequential()
        nn_classifier.add(Dense(units=65, kernel_initializer='uniform', activation='relu', kernel_regularizer=l2(0.001),
                                input_dim=128))
        nn_classifier.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        nn_classifier.add(
            Dense(units=65, kernel_initializer='uniform', activation='relu', kernel_regularizer=l2(0.001)))
        nn_classifier.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        nn_classifier.add(Dense(units=6, kernel_initializer='uniform', activation='softmax'))

        nn_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = nn_classifier.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs,
                                    validation_data=(self.X_val, self.y_val), callbacks=[early_stopping])

        self.history = history
        self.nn_classifier = nn_classifier
        logger.info("Neural Network model trained and evaluated.")

    def evaluate_neural_network_classifier(self):
        """Evaluate Neural Network classifier on test data and calculate confusion matrix."""
        if self.nn_classifier:
            y_pred = self.nn_classifier.predict_classes(self.X_test)
            confusion = confusion_matrix(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            misclassification = 1 - accuracy  # Calculate misclassification error
            logger.info(f"Neural Network Test Accuracy: {accuracy:.4f}")
            logger.info(f"Neural Network Confusion Matrix:\n{confusion}")
            logger.info(f"Neural Network Misclassification Error: {misclassification:.4f}")
            return accuracy, confusion, misclassification
        else:
            logger.error("Neural Network model not trained. Train the model first.")

    def plot_training_history(self):
        """Plot training history of Neural Network model."""
        if self.history is not None:
            plt.figure(figsize=(12, 6))

            # Plotting accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Neural Network Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Plotting loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Neural Network Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            logger.error("No training history available. Train a model first.")


if __name__ == "__main__":
    data_path = r"C:/Users/hakan/Desktop/lsv/gas+sensor+array+drift+dataset+at+different+concentrations"
    model = GasSensorModel(data_path)
    model.load_and_process_data(num_files=10)
    model.save_data_to_csv("C:/Users/hakan/Desktop/lsv/processed_gas_sensor_data.csv")
    model.prepare_data_for_ml()

    # Train KNN classifier with tuning
    knn_classifier, knn_accuracy, knn_misclassification = model.train_knn_classifier_with_tuning()

    # Train SVM classifier with tuning
    svm_classifier, svm_accuracy, svm_misclassification = model.train_svm_classifier_with_tuning()

    # Train Random Forest classifier with tuning
    rf_classifier, rf_accuracy, rf_misclassification = model.train_random_forest_classifier_with_tuning()

    # Train Neural Network classifier
    model.train_neural_network_classifier(batch_size=10, epochs=100)

    # Evaluate Neural Network classifier
    nn_accuracy, nn_confusion, nn_misclassification = model.evaluate_neural_network_classifier()

    # Plot training history of Neural Network model
    model.plot_training_history()

    # Calculate and log misclassification errors
    logger.info(f"KNN Misclassification Error: {knn_misclassification:.4f}")
    logger.info(f"SVM Misclassification Error: {svm_misclassification:.4f}")
    logger.info(f"Random Forest Misclassification Error: {rf_misclassification:.4f}")
    logger.info(f"Neural Network Misclassification Error: {nn_misclassification:.4f}")
