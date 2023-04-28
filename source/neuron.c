#include "neuron.h"

void HAI_neuron_init(HAI_neuron_t *s, uint32_t inputs_size) {
	s->bias = HAI_RANDOM();
	s->weights = malloc(inputs_size * sizeof(double));
	for (uint32_t i = 0; i < inputs_size; i++) {
		s->weights[i] = HAI_RANDOM();
	}
}

void HAI_neuron_free(HAI_neuron_t *s) {
	free(s->weights);
}

double HAI_neuron_forward(HAI_neuron_t *s, const double *inputs, double inputs_size) {
	double output = s->bias;
	for (uint32_t i = 0; i < inputs_size; i++) {
		output += s->weights[i] * inputs[i];
	}
	return output;
}

void HAI_neuron_update(HAI_neuron_t *s, const double *inputs, uint32_t inputs_size, double slope, double learning_rate) {
	double learning_slope = slope * learning_rate;
	for (uint32_t i = 0; i < inputs_size; i++) {
		s->weights[i] -= inputs[i] * learning_slope;
	}
	s->bias -= learning_slope;
}
