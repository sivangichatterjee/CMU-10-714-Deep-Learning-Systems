"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from numpy._core.multiarray import broadcast

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs

        grad_a=out_grad*b*power(a,b-1)
        grad_b=out_grad*node*log(a) #a^b*log(a)(a^b =node)

        return grad_a, grad_b

        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,=node.inputs

        grad_a=out_grad*self.scalar* power_scalar(a, self.scalar-1)

        return grad_a
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs

        grad_a=out_grad/b
        grad_b=-out_grad*a/power_scalar(b,2)

        return grad_a,grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
          return array_api.swapaxes(a,-1,-2)
        else:
          return array_api.swapaxes(a, self.axes[0],self.axes[1])

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,=node.inputs
        return reshape(out_grad,a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

##reshape/transpose= 1 i/p value->1 o/p value
#backward-=rearrange gradient back
##broadcast=one input value -> many output values
#backward = sum all copied gradients back
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
       
        a,=node.inputs
        input_shape=a.shape
        output_shape=self.shape

        # Broadcasting aligns shapes from the RIGHT.
        # Example:
        # input_shape  = (3,)
        # output_shape = (2, 3)
        #
        # Treat input_shape as:
        # padded_input_shape = (1, 3)
        #
        # This lets us compare dimensions axis-by-axis.

        padded_input_shape=(1,) * (len(output_shape)-len(input_shape)) +input_shape
        axes=[]

         # axes will store the dimensions where broadcasting happened.
        # Broadcasting happens when an input dimension was 1
        # and got expanded to a larger output dimension.
        #
        # Example:
        # padded_input_shape = (1, 3, 1)
        # output_shape       = (2, 3, 4)
        #
        # axis 0: 1 -> 2  broadcasted
        # axis 1: 3 -> 3  not broadcasted
        # axis 2: 1 -> 4  broadcasted
        #
        # So axes = [0, 2]

        for i,(in_dim, out_dim) in enumerate(zip(padded_input_shape,output_shape)):
          if in_dim==1 and out_dim!=1:
            axes.append(i)

         # Forward broadcast copies one input value into many output positions.
        # During backward, all gradients from those copied positions
        # must be added together.
        #
        # Therefore, we sum out_grad over every broadcasted axis.
        grad=summation(out_grad, tuple(axes))

         # summation removes dimensions.
        # Example:
        # summation over axes=(0, 2) may give shape (3,)
        #
        # But the gradient w.r.t. input must have the same shape
        # as the original input.
        #
        # So reshape it back to input_shape.
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,=node.inputs

        input_shape=a.shape

        # If axes=None, summation reduced all dimensions.
        # Example: (3, 4) -> scalar
        if self.axes is None:
          axes=tuple(range(len(input_shape)))
        elif isinstance(self.axes,int):
          axes=(self.axes,)
        else:
          axes=self.axes

        # Handle negative axes.
        # Example: axis -1 means last axis.
        axes=tuple(ax if ax>=0 else ax+len(input_shape) for ax in axes)


        # Summation removes the axes that were summed over.
    # In backward, we need to "put back" those removed axes as size 1.
    #
    # Example:
    # input_shape = (3, 4)
    # axes = (1,)
    # forward: (3, 4) -> (3,)
    # out_grad shape is (3,)
    #
    # To broadcast back, first reshape out_grad to (3, 1).
        reshape_shape=list(input_shape)
        for ax in axes:
          reshape_shape[ax]=1

        return broadcast_to(reshape(out_grad, tuple(reshape_shape)),input_shape)

        
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        grad_a=matmul(out_grad, transpose(b))
        grad_b=matmul(transpose(a),out_grad)

         # If broadcasting created extra leading dimensions,
        # sum them away so gradient shape matches original input shape.
        if len(grad_a.shape) > len(a.shape):
            axes = tuple(range(len(grad_a.shape) - len(a.shape)))
            grad_a = summation(grad_a, axes)

        if len(grad_b.shape) > len(b.shape):
            axes = tuple(range(len(grad_b.shape) - len(b.shape)))
            grad_b = summation(grad_b, axes)

        return grad_a,grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad

        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,=node.inputs
        return out_grad/a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask=Tensor(node.realize_cached_data()>0)
        return out_grad*mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

