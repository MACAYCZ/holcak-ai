#include "neuron.h"
#include <stdlib.h>

#define RANDOM()((double)(rand()) / (double)(RAND_MAX) * 2.0f - 1.0f)

HAI_neuron_t HAI_neuron_init(uint32_t inputs_size, HAI_activation_t activate, HAI_activation_t derivative) {
#if 1
	HAI_neuron_t s = {
		.bias = RANDOM(),
		.weights = malloc(inputs_size * sizeof(double)),
		.activate = activate,
		.derivative = derivative
	};
	while (--inputs_size != UINT32_MAX) {
		s.weights[inputs_size] = RANDOM();
	}
	return s;
#else
	HAI_neuron_t s = {
		.bias = 0.0f,
		.weights = malloc(inputs_size * sizeof(double)),
		.activate = activate,
		.derivative = derivative
	};
	while (--inputs_size != UINT32_MAX) {
		s.weights[inputs_size] = 1.0f;
	}
	return s;
#endif
}

void HAI_neuron_free(HAI_neuron_t *s) {
	free(s->weights);
}

double HAI_neuron_weighted(HAI_neuron_t *s, double *inputs, uint32_t inputs_size) {
	double output = s->bias;
	while (--inputs_size != UINT32_MAX) {
		output += inputs[inputs_size] * s->weights[inputs_size];
	}
	return output;
}

double HAI_neuron_forward(HAI_neuron_t *s, double *inputs, uint32_t inputs_size) {
	return s->activate(HAI_neuron_weighted(s, inputs, inputs_size));
}
