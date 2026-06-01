#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i=0;i<m;i+=batch)
    {
      size_t bs=std::min(batch, m-i);

      float *logits= new float[bs * k]();  //Z
      float *probs=new float[bs * k]();    //softmax(Z)
      float *grad= new float[n * k]();    //gradient 

      //1. Compute logits
      //How does THIS example use all features to produce class scores?
      // X -> m*n , theta->n*k, logits -> bs*k, 
      //logits[b][j] = Σ_d X[i+b][d] * theta[d][j]
      //(i+b)*n+d= row*no of columns+col
      for (size_t b=0;b<bs;b++) //fix example
      {
        for(size_t j=0;j<k;j++)//fix class
        {
          float sum=0.0;
          for(size_t d=0;d<n;d++)//sum over features
          {
            sum+=X[(i+b)*n+d]*theta[d*k+j];
          }
          logits[b*k+j]=sum;
        }
      }

      //2.Softmax
      for(size_t b=0;b<bs;b++)
      {
        float sum_exp=0.0;
        for(size_t j=0;j<k;j++)
        {
          probs[b*k+j]=exp(logits[b*k+j]);
          sum_exp+=probs[b*k+j];
        }
         for(size_t j=0;j<k;j++)
        {
          probs[b*k+j]/=sum_exp;
        }
      }

      //3. compute P-Y
      //y[i+b]   = correct class index
      for(size_t b=0;b<bs;b++)
      {
        probs[b*k+y[i+b]]-=1.0;
      }

      //4. Compute gradient
      //How does THIS feature across ALL examples affect class j?
      for(size_t d=0;d<n;d++) //fix feature
      {
        for(size_t j=0;j<k;j++)//fix class
        {
          float sum=0.0;
          for(size_t b=0;b<bs;b++)//sum over examples
          {
            sum+=X[(i+b)*n+d] * probs[b*k+j];
          }
          grad[d*k+j]=sum/bs;
        }
      }

      //5. Update theta
      for(size_t d=0;d<n;d++)
      {
        for(size_t j=0;j<k;j++)
        {
          theta[d*k+j]-=lr*grad[d*k+j];
        }
      }

      delete[] logits;
      delete[] probs;
      delete[] grad; 


      


    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
