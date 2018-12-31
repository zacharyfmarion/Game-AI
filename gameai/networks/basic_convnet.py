from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

from gameai.core import Network


class BasicConvNet(Network):
    '''
    Basic convolutional network

     Attributes:
        input_shape (tuple): Shape of the input
        policy_dims (int): Shape of the policy output
        num_layers (int): Number of convolutional layers in the network
        model (tf.keras.Model): Keras model

    Examples:
        >>> BasicConvNet(input_shape=(3,3,1), policy_dims=(9,), num_layers=1)
    '''

    def __init__(self, input_shape=None, policy_dims=None, num_layers=3):
        self.input_shape = input_shape
        self.policy_dims = policy_dims
        self.num_layers = num_layers

        input_layer = Input(shape=input_shape)
        layer = input_layer
        for _ in range(num_layers):
            layer = Conv2D(10, (1, 1), padding='same',
                           activation='relu')(layer)
        flatten = Flatten()(layer)
        dense1 = Dense(10)(flatten)
        dense2 = Dense(10)(dense1)
        policy = Dense(policy_dims, activation='softmax')(dense2)
        reward = Dense(1, activation="sigmoid")(dense2)

        self.model = Model(inputs=input_layer, outputs=[policy, reward])
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, examples, **kwargs):
        '''
        Train the network

        Args:
            examples (np.ndarray): TODO
        '''
        self.model.fit()

    def predict(self, inputs, **kwargs):
        return self.model.predict(inputs)

    def predict_single(self, x, **kwargs):
        pass
