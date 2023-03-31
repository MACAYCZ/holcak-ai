#include "layer.h"
#include <stdlib.h>

HAI_layer_t HAI_layer_init(uint32_t neurons_size, uint32_t inputs_size, HAI_activation_t activate, HAI_activation_t derivative) {
	HAI_layer_t s = {
		.neurons = malloc(neurons_size * sizeof(HAI_neuron_t)),
		.neurons_size = neurons_size,
	};
	while (--neurons_size != UINT32_MAX) {
		s.neurons[neurons_size] = HAI_neuron_init(inputs_size, activate, derivative);
	}
	return s;
}

void HAI_layer_free(HAI_layer_t *s, uint32_t inputs_size) {
	while (--inputs_size != UINT32_MAX) {
		HAI_neuron_free(&s->neurons[inputs_size]);
	}
	free(s->neurons);
}

double *HAI_layer_forward(HAI_layer_t *s, double *inputs, uint32_t inputs_size) {
	double *outputs = malloc(s->neurons_size * sizeof(double));
	for (uint32_t i = 0; i < s->neurons_size; i++) {
		outputs[i] = HAI_neuron_forward(&s->neurons[i], inputs, inputs_size);
	}
	return outputs;
}

double *HAI_layer_backward(HAI_layer_t *s, double *inputs, uint32_t inputs_size, HAI_layer_t *next_layer, double *delta, double learning_rate, bool output_delta) {
#define MAX(A, B)((A)>(B)?(A):(B))
#define MIN(A, B)((A)<(B)?(A):(B))
	for (uint32_t outi = 0; outi < s->neurons_size; outi++) {
		for (uint32_t inpi = 0; inpi < inputs_size; inpi++) {
			s->neurons[outi].weights[inpi] -= inputs[inpi] * delta[outi] * learning_rate;
			s->neurons[outi].weights[inpi] = MAX(MIN(s->neurons[outi].weights[inpi], 1.0f), -1.0f);
		}
		s->neurons[outi].bias -= delta[outi] * learning_rate;
		s->neurons[outi].bias = MAX(MIN(s->neurons[outi].bias, 1.0f), -1.0f);
	}
	if (output_delta == false) {
		return NULL;
	}
	double *outputs = malloc(inputs_size * sizeof(double));
	for (uint32_t i = 0; i < inputs_size; i++) {
		for (uint32_t j = 0; j < s->neurons_size; j++) {			
			outputs[i] = s->neurons[i].weights[j] * delta[j];
		}
	}
	return outputs;
}
