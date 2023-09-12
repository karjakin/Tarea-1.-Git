# %load network.py


#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
# toda la red se crea como una clase y esta clase tiene diferentes métodos, estos son las deferentes partes de la red
class Network(object):

    def __init__(self, sizes):
        """
        Inicializa la red neuronal con un número aleatorio de sesgos y pesos.
        sizes: lista de enteros que representa el número de neuronas en cada capa.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Realiza una propagación hacia adelante a través de la red.
        a: la entrada de la red.
        return: la salida de la red.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Entrena la red neuronal usando el algoritmo de descenso de gradiente estocástico.
        training_data: lista de tuplas que representan los datos de entrenamiento y sus etiquetas.
        epochs: número de épocas para entrenar.
        mini_batch_size: tamaño del mini-lote.
        eta: tasa de aprendizaje.
        test_data: datos de prueba para evaluar el rendimiento de la red (opcional).
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualiza los pesos y sesgos de la red aplicando una iteración del descenso de gradiente.
        mini_batch: lista de tuplas representando un mini-lote.
        eta: tasa de aprendizaje.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Realiza una retropropagación para calcular los gradientes respecto a los sesgos y los pesos.
        x: vector de entrada.
        y: salida deseada (etiqueta).
        return: tupla (nabla_b, nabla_w) representando el gradiente para los sesgos y los pesos.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evalúa el rendimiento de la red neuronal en un conjunto de datos de prueba.
        test_data: conjunto de datos de prueba.
        return: número de entradas de prueba para las cuales la red neuronal produce el resultado correcto.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Devuelve la derivada del costo respecto a la activación de salida.
        output_activations: vector de activaciones de salida.
        y: vector de salida deseada (etiqueta).
        return: vector representando la derivada del costo.
        """
        return (output_activations-y)
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
