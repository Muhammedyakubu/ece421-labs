import numpy as np


class Activation:
    def __init__(self, name):
        # Based on the name used for creating an Activation object,
        # we set the self.optimize to be the desiarable method.
        if name == "linear":
            self.forward = self.forward_linear
            self.backward = self.backward_linear
        elif name == "sigmoid":
            self.forward = self.forward_sigmoid
            self.backward = self.backward_sigmoid
        elif name == "tanh":
            self.forward = self.forward_tanh
            self.backward = self.backward_tanh
        elif name == "arctan":
            self.forward = self.forward_arctan
            self.backward = self.backward_arctan
        elif name == "relu":
            self.forward = self.forward_relu
            self.backward = self.backward_relu
        elif name == "softmax":
            self.forward = self.forward_softmax
            self.backward = self.backward_softmax
        else:
            raise NotImplementedError("{} activation is not implemented".format(name))
        
    def forward_linear(self, Z):
        """
        Forward pass for f(z) = z. 

        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.
        
        Returns
        -------
        f(z): as described above applied elementwise to `Z`
        """

        return Z

    def backward_linear(self, Z, dY):
        """
        Backward pass for f(z) = z.

        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        return dY

    def forward_tanh(self, Z):
        """
        Forward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.
        
        Returns
        -------
        f(z): as described above applied elementwise to `Z`
        """

        return 2 / (1 + np.exp(-2 * Z)) - 1

    def backward_tanh(self, Z, dY):
        """
        Backward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        fn = self.forward(Z)
        return dY * (1 - fn ** 2)
    
    def forward_arctan(self, Z):
        """
        Forward pass for f(z) = arctan(z).

        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.
        
        Returns
        -------
        f(z): as described above applied elementwise to `Z`
        """
        return np.arctan(Z)

    def backward_arctan(self, Z, dY):
        """
        Backward pass for f(z) = arctan(z).
        
        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        return dY * 1 / (Z ** 2 + 1)
    
    def forward_relu(self, Z):
        """
        Forward pass for relu activation: f(z) = z if z >= 0, and 0 otherwise
        
        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.

        Returns
        -------
        f(z): as described above applied elementwise to `Z`
        """
        fz = np.maximum(0, Z)
        # print(fz)
        return fz

    def backward_relu(self, Z, dY):
        """
        Backward pass for relu activation.
        
        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        dLdZ = dY * (Z > 0)
        return dLdZ

    def forward_softmax(self, Z):
        """
        Forward pass for softmax activation.
        Note that the naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.

        Returns
        -------
        f(z) as described above. It has the same shape as `Z`
        """
        # Extract number of training points
        n = Z.shape[0]
        fz = np.zeros_like(Z)
        for value in range(n):

            # Calculates softmax for each training point
            m = np.max(Z[value])
            expZ = np.exp(Z[value]-m)
            fz[value] = expZ/np.sum(expZ)
        return fz

    def backward_softmax(self, Z, dY):
        """
        Backward pass for softmax activation.
        
        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        # Extract number of classes
        k = Z.shape[1]

        # Compute the softmax output 
        softmax = self.forward(Z)

        # Compute the Jacobian matrix (includes both cases where i != j and i = j)
        jacobian = softmax[:, :, np.newaxis] * (np.eye(k) - softmax[:, np.newaxis, :])

        # Compute the gradient for each sample (dL/dZ = dY/dZ * dL/dY)
        dZ = np.einsum('ijk,ij->ik', jacobian, dY)

        return dZ
        

    def forward_sigmoid(self, Z):
        """
        Forward pass for sigmoid function f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z:  The input to the activation function (i.e. pre-activation). 
            It is an np.ndarray and can have any shape.

        Returns
        -------
        f(z): as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        pass

    def backward_sigmoid(self, Z, dY):
        """
        Backward pass for sigmoid.
        
        Parameters
        ----------
        Z:  Input to `forward` method.
        dY: Gradient of loss w.r.t. the output of this layer.
            It is an np.ndarray with the same shape as `Z`.

        Returns
        -------
        gradient of loss w.r.t. input of the activation function, i.e., 'Z'.
        It is an np.ndarray with the same shape as `Z`.
        """
        ### YOUR CODE HERE ###
        pass
