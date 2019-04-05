import keras
from keras.layers import Dense, Concatenate
import json
import os


class DeepFactorization:
    """
    A neural-network matrix factorization model for user-item ratings implemented in Keras.
    """

    def __init__(self, n_users, n_items, n_latent_factors, n_hidden):
        """
        Arguments:
            n_users {int} -- total number of users
            n_items {int} -- total number of items
            n_latent_factors {int} -- rank of factor matrices
            n_hidden {int} -- size of hidden neural network layer
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_latent_factors = n_latent_factors
        self.n_hidden = n_hidden

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the keras model.

        Returns:
            keras.Model -- The untrained keras model.
        """

        movie_input = keras.layers.Input(shape=[1], name='Item')

        item_emb_layer = keras.layers.Embedding(self.n_items + 1, self.n_latent_factors, name='Movie-Embedding')
        movie_embedding = item_emb_layer(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

        user_input = keras.layers.Input(shape=[1], name='User')
        user_emb_layer = keras.layers.Embedding(self.n_users + 1, self.n_latent_factors, name='User-Embedding')

        user_vec = keras.layers.Flatten(name='FlattenUsers')(
            user_emb_layer(user_input)
        )

        input_vecs = Concatenate()([user_vec, movie_vec])

        x = Dense(self.n_hidden, activation='relu')(input_vecs)

        y = Dense(1)(x)

        model = keras.Model([user_input, movie_input], y)
        model.compile('adam', 'mean_squared_error')

        return model


    @staticmethod
    def build_model_from_file(model_path, model_name):
        """
        Build the keras model according to specifications 
        safed in a model file.

        Arguments:
            model_path {string} -- Path to the model files
            model_name {string} -- Name of the model to load  
        Returns:
            keras.model -- Keras model loaded from file.
        """

        with open(os.path.join(model_path, '{model_name}.json'.format(model_name=model_name), 'r')) as fp:
            model_params = json.load(fp)

        model = DeepFactorization(**model_params)

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
            'n_latent_factors': int(self.n_latent_factors),
            'n_hidden': int(self.n_hidden)
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
        file_name = '{model_name}.h5'.format(model_name=model_name)

        self.model.save_weights(os.path.join(model_path, file_name))

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

        history = self.model.fit(train_x, train_y, **keras_params)

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
