#include "network.h"
#include <stdbool.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

HAI_network_t HAI_network_init(uint32_t layers_size, uint32_t inputs_size, ...) {
	HAI_network_t s = {
		.layers = malloc(layers_size * sizeof(HAI_layer_t)),
		.layers_size = layers_size,
		.inputs_size = inputs_size
	};
	va_list layers_info;
	va_start(layers_info, inputs_size);
	struct timespec current_time;
	clock_gettime(CLOCK_MONOTONIC, &current_time);
	srand(current_time.tv_nsec);
	while (--layers_size != UINT32_MAX) {
		uint32_t neurons_size = va_arg(layers_info, uint32_t);
		HAI_activation_t activate = va_arg(layers_info, HAI_activation_t);
		HAI_activation_t derivative = va_arg(layers_info, HAI_activation_t);
		s.layers[layers_size] = HAI_layer_init(neurons_size, inputs_size, activate, derivative);
		inputs_size = neurons_size;
	}
	va_end(layers_info);
	return s;
}

void HAI_network_free(HAI_network_t *s) {
	uint32_t inputs_size = s->inputs_size;
	while (--s->layers_size != UINT32_MAX) {
		HAI_layer_free(&s->layers[s->layers_size], inputs_size);
		inputs_size = s->layers[s->layers_size].neurons_size;
	}
	free(s->layers);
}

double *HAI_network_forward(HAI_network_t *s, double *inputs) {
	inputs = HAI_layer_forward(&s->layers[0], inputs, s->inputs_size);
	for (uint32_t i = 1; i < s->layers_size; i++) {
		double *old_inputs = inputs;
		inputs = HAI_layer_forward(&s->layers[i], inputs, s->layers[i-1].neurons_size);
		free(old_inputs);
	}
	return inputs;
}

void HAI_network_backward(HAI_network_t *s, double *inputs, double *expected, double learning_rate) {
	double **weighted = malloc(s->layers_size * sizeof(double*));
	double **activated = malloc((s->layers_size + 1) * sizeof(double*));
	uint32_t inputs_size = s->inputs_size;
	activated[0] = inputs;

	for (uint32_t i = 0; i < s->layers_size; i++) {
		weighted[i] = malloc(s->layers[i].neurons_size * sizeof(double));
		activated[i+1] = malloc(s->layers[i].neurons_size * sizeof(double));
		for (uint32_t j = 0; j < s->layers[i].neurons_size; j++) {
			weighted[i][j] = HAI_neuron_weighted(&s->layers[i].neurons[j], activated[i], inputs_size);
			activated[i+1][j] = s->layers[i].neurons[j].activate(weighted[i][j]);
		}
		inputs_size = s->layers[i].neurons_size;
	}

	// NOTE: delta = nodeValues
	double *delta = malloc(s->layers[s->layers_size-1].neurons_size * sizeof(double));
	for (uint32_t i = 0; i < s->layers[s->layers_size-1].neurons_size; i++) {
		double cost_delta = 2 * (activated[s->layers_size][i] - expected[i]);
		double weighted_delta = s->layers[s->layers_size-1].neurons[i].derivative(weighted[s->layers_size-1][i]);
		delta[i] = cost_delta * weighted_delta;
	}

	for (uint32_t i = s->layers_size-1; i > 0; i--) {
		double *old_delta = delta;
		delta = HAI_layer_backward(&s->layers[i], activated[i], s->layers[i-1].neurons_size, &s->layers[i+1], delta, learning_rate, true);
		free(old_delta);
	}
	HAI_layer_backward(&s->layers[0], activated[0], s->inputs_size, NULL, delta, learning_rate, false);

	for (uint32_t i = 0; i < s->layers_size; i++) {
		free(activated[i+1]);
		free(weighted[i]);
	}
	free(activated);
	free(weighted);
}

double HAI_network_cost(HAI_network_t *s, double *inputs, double *expected) {
	double cost = 0.0f;
	double *outputs = HAI_network_forward(s, inputs);
	for (uint32_t i = 0; i < s->layers[s->layers_size-1].neurons_size; i++) {
		double output_cost = outputs[i] - expected[i];
		cost += output_cost * output_cost;
	}
	free(outputs);
	return cost / 2;
}
