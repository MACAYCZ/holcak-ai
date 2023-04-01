#include "neuron.h"
#include <stdlib.h>
#include <math.h>

HAI_neuron_t HAI_neuron_init(uint32_t inputs_size, HAI_activation_t activate, HAI_activation_t derivative) {
	HAI_neuron_t s = {
		.bias = (double)(rand()) / (double)(RAND_MAX) * 2.0f - 1.0f,
		.weights = malloc(inputs_size * sizeof(double)),
		.activate = activate,
		.derivative = derivative
	};
	while (--inputs_size != UINT32_MAX) {
		s.weights[inputs_size] = (double)(rand()) / (double)(RAND_MAX) * 2.0f - 1.0f;
	}
	return s;
}

void HAI_neuron_free(HAI_neuron_t *s) {
	free(s->weights);
}

double HAI_neuron_forward(HAI_neuron_t *s, double *inputs, uint32_t inputs_size) {
	double output = s->bias;
	while (--inputs_size != UINT32_MAX) {
		output += inputs[inputs_size] * s->weights[inputs_size];
	}
	return s->activate(output);
}

double *HAI_neuron_forward_info(HAI_neuron_t *s, double *inputs, uint32_t inputs_size) {
	double *outputs = malloc(2 * sizeof(double));
	outputs[0] = s->bias;
	while (--inputs_size != UINT32_MAX) {
		outputs[0] += inputs[inputs_size] * s->weights[inputs_size];
	}
	outputs[1] = s->activate(outputs[0]);
	return outputs;
}

void HAI_neuron_update(HAI_neuron_t *s, double slope, double *activated_inputs, double learning_rate, uint32_t inputs_size) {
	for (uint32_t i = 0; i < inputs_size; i++) {
		s->weights[i] -= activated_inputs[i] * slope * learning_rate;
		s->weights[i] = fmax(-1.0f, fmin(1.0f, s->weights[i]));
	}
	s->bias -= slope * learning_rate;
	s->bias = fmax(-1.0f, fmin(1.0f, s->bias));
}
