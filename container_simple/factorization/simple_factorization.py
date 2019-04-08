import keras
import keras.backend as K
from keras.callbacks import EarlyStopping
import json
import os


class SimpleFactorization:
    """
    A simple matrix factorization model for user-item ratings implemented in Keras.
    """

    def __init__(self, n_users, n_items, n_latent_factors):
        """
        Arguments:
            n_users {int} -- total number of users
            n_item {int} -- total number of items
            n_latent_factors {int} -- rank of factor matrices
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_latent_factors = n_latent_factors

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the keras model.

        Returns:
            keras.Model -- The untrained keras model.
        """

        movie_input = keras.layers.Input(shape=[1], name='Item')
        movie_embedding = keras.layers.Embedding(self.n_items + 1,
                                                 self.n_latent_factors,
                                                 name='Movie-Embedding')(movie_input)

        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

        user_input = keras.layers.Input(shape=[1],name='User')
        user_vec = keras.layers.Flatten(name='FlattenUsers')(
            keras.layers.Embedding(self.n_users + 1, self.n_latent_factors,name='User-Embedding')(user_input)
        )

        prod = keras.layers.Lambda(lambda x: K.sum(x[0] * x[1], axis=-1,keepdims=True))([movie_vec, user_vec])

        model = keras.Model([user_input, movie_input], prod)
        model.compile('adam', 'mean_squared_error')

        return model

    @staticmethod
    def build_model_from_file(model_path, model_name):
        """
        Build the keras model according to specifications safed in a model file.
        
        Arguments:
            model_path {string} -- Path to the model files
            model_name {string} -- Name of the model to load
        
        Returns:
            keras.model -- Keras model loaded from file.
        """
        
        with open(os.path.join(model_path, '{model_name}.json'.format(model_name=model_name)), 'r') as fp:
            model_params = json.load(fp)

        model = SimpleFactorization(**model_params)

        model._load_weights(model_path, model_name)

        return model

    def save_model(self, model_path, model_name):
        """
        Save model to file.
       
        Arguments:
            model_path {string} -- Path to the model files
            model_name {string} -- Name of the model to save
        """

        model_params = {
            'n_users': int(self.n_users),
            'n_items': int(self.n_items),
            'n_latent_factors': int(self.n_latent_factors)
        }

        with open(os.path.join(model_path, '{model_name}.json'.format(model_name=model_name)), 'w') as fp:
            json.dump(model_params, fp)

        self._save_weights(model_path, model_name)

    def _save_weights(self, model_path, model_name):
        """
        Save the weigths of the model to a file.
      
        Arguments:
            model_path {string} -- Path to folder to save the model weights
            model_name {string} -- Name of the model to save weights for
        """

        self.model.save_weights(os.path.join(model_path, '{model_name}.h5'.format(model_name=model_name)))

    def _load_weights(self, model_path, model_name):
        """
        Load the weights of the model from a file.
       
        Arguments:
            model_path {string} -- Path to folder containing the model weights
            model_name {string} -- Name of the model to load weights for
        """

        self.model.load_weights(os.path.join(model_path, '{model_name}.h5'.format(model_name=model_name)))
        self.model._make_predict_function()

    def train(self, train_x, train_y, **keras_params):
        """
        Train the keras model

        Arguments:
            train_x {list[list[int]]} -- List of pairs of user and item ids
            train_y {list[int]} -- List of user ratings

        Returns:
            keras.callbacks.History -- History of the training
        """
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

        history = self.model.fit(train_x, train_y, validation_split=0.1, callbacks=[es], **keras_params)

        return history

    def predict(self, x_pred):
        """
        Predict rating for pairs of users and items.

        Arguments:
            x_pred {list[list[int]]} -- list of user and item ids

        Returns:
            numpy.array -- Numpy array with predictions.
        """

        return self.model.predict(x_pred)

    def evaluate(self, x_val, y_val):
        """
        Evaluate mean squared error of the model.

        Arguments:
            x_val {list[list[int]]} -- list of user and item ids
            y_val {list[int]} -- list of user ratings
               
        Returns:
            float -- Mean squared error
        """

        return self.model.evaluate(x_val, y_val)
