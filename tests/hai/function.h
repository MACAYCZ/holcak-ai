#ifndef HAI_FUNCTION_H_
#define HAI_FUNCTION_H_
#include <stdlib.h>

typedef struct {
	double(*forward)(double);
	double(*backward)(double);
} HAI_function_t;

#define HAI_FUNCTION_NULL (HAI_function_t){NULL, NULL}

#define HAI_SIGMOID (HAI_function_t){HAI_sigmoid_forward, HAI_sigmoid_backward}
double HAI_sigmoid_forward(double x);
double HAI_sigmoid_backward(double x);

#define HAI_TANH (HAI_function_t){HAI_tanh_forward, HAI_tanh_backward}
double HAI_tanh_forward(double x);
double HAI_tanh_backward(double x);

#define HAI_RELU (HAI_function_t){HAI_relu_forward, HAI_relu_backward}
double HAI_relu_forward(double x);
double HAI_relu_backward(double x);

#endif//HAI_FUNCTION_H_
