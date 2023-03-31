#ifndef HAI_ACTIVATION_H_
#define HAI_ACTIVATION_H_

typedef double(*HAI_activation_t)(double);

#define HAI_SIGMOID HAI_sigmoid_activate, HAI_sigmoid_derivative
double HAI_sigmoid_activate(double x);
double HAI_sigmoid_derivative(double x);

#define HAI_TANH HAI_tanh_activate, HAI_tanh_derivative
double HAI_tanh_activate(double x);
double HAI_tanh_derivative(double x);

#endif//HAI_ACTIVATION_H_
