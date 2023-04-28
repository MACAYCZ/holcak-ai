#include "function.h"
#include <math.h>

double HAI_sigmoid_forward(double x) {
	return 1.0f / (1.0f + exp(-x));
}

double HAI_sigmoid_backward(double x) {
	double y = HAI_sigmoid_forward(x);
	return y * (1.0f - y);
}

double HAI_tanh_forward(double x) {
	double y = exp(2.0f * x);
	return (y - 1.0f) / (y + 1.0f);
//	double y = exp(x), z = exp(-x);
//	return (y - z) / (y + z);
}

double HAI_tanh_backward(double x) {
	double y = HAI_tanh_forward(x);
	return 1.0f - y * y;
}

double HAI_relu_forward(double x) {
	return x > 0.0f ? x : 0.0f;
}

double HAI_relu_backward(double x) {
	return x > 0.0f ? 1.0f : 0.0f;
}
