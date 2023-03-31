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
