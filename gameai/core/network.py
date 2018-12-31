class Network:
    '''
    TODO: Document this
    '''

    def train(self, examples, **kwargs):
        '''
        Train the network
        '''
        raise NotImplementedError

    def predict(self, inputs, **kwargs):
        '''
        Given an array of inputs predict the outputs

        Args:
            inputs (np.ndarray): Array of inputs. Note that to predict a single
                example you need to add an extra dimension before calling this

        Returns:
            np.ndarray: The output of the network
        '''
        raise NotImplementedError

    def predict_single(self, x, **kwargs):
        '''
        Given a single input predict the output

        Args:
            x (np.array): Input you want to predict

        Returns:
            any: The output of the network
        '''
        raise NotImplementedError
