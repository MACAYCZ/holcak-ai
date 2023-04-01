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
	while (--s->layers_size != UINT32_MAX) {
		HAI_layer_free(&s->layers[s->layers_size]);
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
	double **activated = malloc(s->layers_size * sizeof(double*));
	for (uint32_t i = 0; i < s->layers_size; i++) {
		double **outputs = HAI_layer_forward_info(&s->layers[i],
				i == 0 ? inputs : activated[i-1],
				i == 0 ? s->inputs_size : s->layers[i-1].neurons_size);
		weighted[i] = outputs[0];
		activated[i] = outputs[1];
		free(outputs);
	}
	double *slope = HAI_layer_output_slope(&s->layers[s->layers_size-1],
			weighted[s->layers_size-1], activated[s->layers_size-1], expected);
	for (uint32_t i = s->layers_size - 1; i != UINT32_MAX; i--) {
		double *old_slope = slope;
		slope = i > 0 ? HAI_layer_hidden_slope(&s->layers[i], slope, weighted[i-1], activated[i-1], &s->layers[i-1], learning_rate, s->layers[i-1].neurons_size) :
			HAI_layer_hidden_slope(&s->layers[i], slope, NULL, inputs, NULL, learning_rate, s->inputs_size);
		free(old_slope);
	}
	for (uint32_t i = 0; i < s->layers_size; i++) {
		free(weighted[i]);
		free(activated[i]);
	}
	free(weighted);
	free(activated);
}

double HAI_network_cost(HAI_network_t *s, double *inputs, double *expected) {
	double cost = 0.0f;
	double *outputs = HAI_network_forward(s, inputs);
	for (uint32_t i = 0; i < s->layers[s->layers_size-1].neurons_size; i++) {
		double output_cost = outputs[i] - expected[i];
		cost += output_cost * output_cost;
	}
	free(outputs);
	return cost;
}
