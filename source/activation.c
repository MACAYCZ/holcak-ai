#include "activation.h"
#include <math.h>

double HAI_sigmoid_activate(double x) {
	return 1 / (1 + exp(-x));
}

double HAI_sigmoid_derivative(double x) {
	double a = HAI_sigmoid_activate(x);
	return a * (1 - a);
}

double HAI_tanh_activate(double x) {
	double a = exp(x), b = exp(-x);
	return (a - b) / (a + b);
}

double HAI_tanh_derivative(double x) {
	double a = HAI_tanh_activate(x);
	return 1 - a * a;
}

double HAI_leaky_relu_activate(double x) {
	return fmax(0.05f * x, x);
}

double HAI_leaky_relu_derivative(double x) {
	return x > 0.0f ? 1.0f : 0.05f;
}
