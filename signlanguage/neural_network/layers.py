from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class AbstractLayer(ABC):
  def __init__(self):
    self.input_shape = None
    self.output_shape = None
    self.weights = None
    self.weights_gradient = None
    self.bias = None
    self.bias_gradient = None
    self.prev_input = None
  
  @abstractproperty
  def has_weights(self) -> bool:
    pass
  
  @property
  def weights_shape(self) -> tuple:
    return ()
  
  @property
  def weights_size(self) -> int:
    return np.prod(self.weights_shape) if self.weights_shape else 0
  
  def update_weights(self, weights: np.array):
    self.weights = weights
  
  @abstractproperty
  def has_bias(self) -> bool:
    pass

  @property
  def bias_size(self) -> int:
    return 0
  
  def update_bias(self, bias: np.array):
    self.bias = bias
  
  @property
  def input_size(self) -> int:
    return np.prod(self.input_shape)
  
  @property
  def output_size(self) -> int:
    return np.prod(self.output_shape)

  def initialize(self):
    pass

  @abstractmethod
  def forward(self, x: np.array) -> np.array:
    raise NotImplementedError

  @abstractmethod
  def backward(self, dout: np.array) -> np.array:
    raise NotImplementedError


class InputLayer(AbstractLayer):
  def __init__(self, shape: tuple):
    super().__init__()
    self.input_shape = shape
    self.output_shape = shape
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  def forward(self, x: np.array) -> np.array:
    return x
  
  def backward(self, dout: np.array) -> np.array:
    return dout


class DenseLayer(AbstractLayer):
  def __init__(self, units: int):
    super().__init__()
    self.units = units
    self.output_shape = (units, )
  
  @property
  def has_weights(self):
    return True

  @property
  def has_bias(self):
    return True
  
  @property
  def weights_shape(self) -> tuple:
    return (self.input_size, self.output_size)
  
  @property
  def bias_size(self) -> int:
    return self.units

  def initialize(self):
    if self.weights is None:
      self.weights = np.random.randn(*self.weights_shape) * 0.1
    if self.bias is None:
      self.bias = np.zeros(self.bias_size)

  def forward(self, x: np.array) -> np.array:
    self.prev_input = x.copy()
    return (x @ self.weights) + self.bias

  def backward(self, dout: np.array) -> np.array:
    self.weights_gradient = self.prev_input.T @ dout
    self.bias_gradient = dout.sum(axis=0)
    return dout @ self.weights.T


class ActivationLayer(AbstractLayer):
  def __init__(self, function):
    super().__init__()
    self.function = function
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  @property
  def output_shape(self) -> tuple:
    return self.input_shape
  
  @output_shape.setter
  def output_shape(self, value):
    pass
  
  def forward(self, x: np.array) -> np.array:
    self.prev_input = x.copy()
    return self.function(x)
  
  def backward(self, dout: np.array) -> np.array:
    return dout * self.function(self.prev_input, derivative=True)


class DropoutLayer(AbstractLayer):
  def __init__(self, drop_probability: float):
    super().__init__()
    self.keep_probability = 1 - drop_probability
    self.scale = 1 / self.keep_probability if self.keep_probability > 0 else 0
    self.mask = None
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  @property
  def output_shape(self) -> tuple:
    return self.input_shape
  
  @output_shape.setter
  def output_shape(self, value):
    pass

  def forward(self, x: np.array) -> np.array:
    self.mask = np.random.binomial(1, self.keep_probability, size=self.output_shape)
    return x * self.mask * self.scale

  def backward(self, dout: np.array) -> np.array:
    return dout * self.mask * self.scale


class FlattenLayer(AbstractLayer):
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  @property
  def output_shape(self) -> tuple:
    return self.input_shape
  
  @output_shape.setter
  def output_shape(self, value):
    pass
  
  def forward(self, x: np.array) -> np.array:
    return x.copy().reshape((-1, self.input_size))
  
  def backward(self, dout: np.array) -> np.array:
    return dout.copy().reshape((-1, *self.input_shape))


class Conv2DLayer(AbstractLayer):
  def __init__(self, num_of_kernels: int, kernel_shape: tuple, num_of_channels: int, stride: tuple=(1, 1)):
    super().__init__()
    self.num_of_kernels = num_of_kernels
    self.kernel_shape = self.kernel_height, self.kernel_width = kernel_shape
    self.kernel_channels = num_of_channels
    self.stride = self.vertical_stride, self.horizontal_stride = stride
  
  @property
  def has_weights(self):
    return True

  @property
  def has_bias(self):
    return True

  @property
  def weights_shape(self) -> tuple:
    return (self.num_of_kernels, self.kernel_channels, *self.kernel_shape)
  
  @property
  def bias_size(self) -> int:
    return self.num_of_kernels

  def initialize(self):
    if self.weights is None:
      self.weights = np.random.randn(*self.weights_shape) * 0.1
    if self.bias is None:
      self.bias = np.zeros(self.bias_size)
  
  @property
  def output_height(self) -> int:
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_height - self.kernel_height) // self.vertical_stride

  @property
  def output_width(self) -> int:
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_width - self.kernel_width) // self.horizontal_stride

  @property
  def output_shape(self) -> tuple:
    return (self.num_of_kernels, self.output_height, self.output_width)
  
  @output_shape.setter
  def output_shape(self, value):
    pass

  def forward(self, x: np.array) -> np.array:
        N = x.shape[0]  # Number of samples in the input
        output = np.zeros((N, self.num_of_kernels, self.output_height, self.output_width))

        for j in range(self.output_height):
            for i in range(self.output_width):
                x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
                x_max, y_max = x_min + self.kernel_width, y_min + self.kernel_height

                input_slice = x[:, :, y_min:y_max, x_min:x_max]

                # Ensure the shapes of input_slice and weights are compatible for element-wise multiplication
                # Assuming weights shape: (num_kernels, input_channels, kernel_height, kernel_width)
                convolved = np.zeros((N, self.num_of_kernels, input_slice.shape[2], input_slice.shape[3]))

                for k in range(self.num_of_kernels):
                    # Perform element-wise multiplication for each kernel
                    convolved[:, k] = np.sum(input_slice * self.weights[k][:, np.newaxis, :, :], axis=1)

                # Sum across the kernel dimension and add bias
                output[:, :, j, i] = np.sum(convolved, axis=1) + self.bias

        self.prev_input = x.copy()
        return output



  

def backward(self, dout: np.array) -> np.array:
    input_samples, input_channels, input_height, input_width = self.prev_input.shape

    input_gradient = np.zeros_like(self.prev_input)
    self.weights_gradient = np.zeros_like(self.weights)
    
    self.bias_gradient = np.sum(dout, axis=(0, 2, 3))

    for kernel in range(self.num_of_kernels):
        vertical_steps = input_height - self.kernel_height + 1
        for j in range(0, vertical_steps, self.vertical_stride):
            horizontal_steps = input_width - self.kernel_width + 1
            for i in range(0, horizontal_steps, self.horizontal_stride):
                input_slice = self.prev_input[:, :, j:j+self.kernel_height, i:i+self.kernel_width]
                output_x_index, output_y_index = i // self.horizontal_stride, j // self.vertical_stride
                next_layer_gradient_slice = dout[:, kernel, output_y_index, output_x_index]
                
                # Accumulate gradients
                self.weights_gradient[kernel] += np.sum(
                    input_slice * next_layer_gradient_slice[:, np.newaxis, np.newaxis, np.newaxis],
                    axis=0
                )
                
                input_gradient[:, :, j:j+self.kernel_height, i:i+self.kernel_width] += \
                    next_layer_gradient_slice[:, np.newaxis, np.newaxis, np.newaxis] * self.weights[kernel]
    
    return input_gradient


class MaxPooling2DLayer(AbstractLayer):
  def __init__(self, pool_shape: tuple, stride: tuple=(1, 1)):
    super().__init__()
    self.pool_shape = self.pool_height, self.pool_width = pool_shape
    self.stride = self.vertical_stride, self.horizontal_stride = stride
    self.prev_output = None
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  @property
  def output_height(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_height - self.pool_height) // self.vertical_stride
  
  @property
  def output_width(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_width - self.pool_width) // self.horizontal_stride

  @property
  def output_shape(self):
    input_channels, input_height, input_width = self.input_shape
    return (input_channels, self.output_height, self.output_width)
  
  @output_shape.setter
  def output_shape(self, value):
    pass
 
  def forward(self, x: np.array) -> np.array:
    N = len(x)
    output = np.zeros((N, *self.output_shape))
    for j in range(self.output_height):
      for i in range(self.output_width):
        x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
        x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
        input_slice = x[:, :, y_min:y_max, x_min:x_max]
        output[:, :, j, i] = np.max(input_slice, axis=(2, 3))
    self.prev_input = x.copy()
    self.prev_output = output.copy()
    return output

  def backward(self, dout: np.array) -> np.array:
    input_samples, input_channels, input_height, input_width = self.prev_input.shape
    input_gradient = np.zeros_like(self.prev_input)
    for j in range(self.output_height):
      for i in range(self.output_width):
        x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
        x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
        input_slice = self.prev_input[:, :, y_min:y_max, x_min:x_max]
        mask = (input_slice.reshape((input_samples, input_channels, self.pool_height*self.pool_width)) == self.prev_output[:, :, j, i, np.newaxis])
        gradient_slice = input_gradient[:, :, y_min:y_max, x_min:x_max]
        gradient_slice += (mask * dout[:, :, j, i, np.newaxis]).reshape(gradient_slice.shape)
    return input_gradient