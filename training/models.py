from typing import Tuple
import numpy as np

model_map = {}
class DetectorModel:

    # Inputs is a length-n Numpy array, where n is the number of datapoints.
    # Output is a bool, with malicious being True.
    # Returns whether or not the guess was correct.
    # The model should also train itself based on its performance.
    def train(self, inputs, output) -> bool:
        print("unimplemented!")
        raise Exception()

    # The same as the train function, except the state should not be modified.
    def test(self, inputs, output) -> bool:
        print("unimplemented!")
        raise Exception()

class BasicPerceptron(DetectorModel):
    def __init__(self, n_datapoints):
        self.weights = np.zeros(n_datapoints)
        self.bias = 0

    def train(self, inputs: np.ndarray, output: bool):
        # The vector of inputs (each a value between 0 and 1) is elementwise rounded to the nearest integer. 0's are converted to -1.
        rounded_inputs = np.where(inputs >= 0.5, 1, -1)
        # The perceptron predictor's score is computed using a dot product of the rounded inputs and weights.
        score = np.sum(self.weights * rounded_inputs) # + self.bias
        if output:
            if score > 0:
                return True # If the output is True (malicious) and the score is positive, then a true positive occurs.
            # Otherwise, a false negative occurs and the weights are adjusted to make the false negative less likely to occur in the future.
            self.weights += rounded_inputs
            self.bias += 1
        else:
            if score <= 0:
                return True # If the output is False (benign) and the score is not positive, then a true negative occurs.
            # Otherwise, a false positive occurs and the weights are adjusted to make the false positive less likely to occur in the future.
            self.weights -= rounded_inputs
            self.bias -= 1
        return False

    def test(self, inputs: np.ndarray, output: bool):
        # The score computation is handled the same way as in the training function.
        rounded_inputs = np.where(inputs >= 0.5, 1, -1)
        score = np.sum(self.weights * rounded_inputs) # + self.bias
        if output:
            return score > 0
        else:
            return score <= 0

# New entries to the model map are registered here.
model_map["basic_perceptron"] = BasicPerceptron
        