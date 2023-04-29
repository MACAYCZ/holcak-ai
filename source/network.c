#include "network.h"
#include <stdlib.h>
#include <stdarg.h>

void HAI_network_init_(HAI_network_t *s, uint32_t inputs_size, ...) {
	s->inputs_size = inputs_size;
	s->layers = NULL;
	s->size = 0;

	va_list args;
	va_start(args, inputs_size);
	uint32_t biggest_layer = 0;
	while (1) {
		HAI_function_t function = va_arg(args, HAI_function_t);
		if (function.forward == NULL) {
			break;
		}

		uint32_t size = va_arg(args, uint32_t);
		if (size > biggest_layer) {
			biggest_layer = size;
		}

		s->layers = realloc(s->layers, ++s->size * sizeof(HAI_layer_t));
		HAI_layer_init(&s->layers[s->size-1], inputs_size, function, size);
		inputs_size = size;
	}

	s->slope1 = malloc(biggest_layer * sizeof(double));
	s->slope2 = malloc(biggest_layer * sizeof(double));
	va_end(args);
}

void HAI_network_free(HAI_network_t *s) {
	for (uint32_t i = 0; i < s->size; i++) {
		HAI_layer_free(&s->layers[i]);
	}
	free(s->layers);
	free(s->slope1);
	free(s->slope2);
}

const double *HAI_network_forward(HAI_network_t *s, const double *inputs) {
	uint32_t inputs_size = s->inputs_size;
	for (uint32_t i = 0; i < s->size; i++) {
		inputs = HAI_layer_forward(&s->layers[i], inputs, inputs_size);
		inputs_size = s->layers[i].size;
	}
	return inputs;
}

void HAI_network_backward(HAI_network_t *s, const double *inputs, const double *expected, double learning_rate) {
	HAI_network_forward(s, inputs);
	for (uint32_t i = 0; i < s->layers[s->size-1].size; i++) {
		s->slope1[i] = 2.0f * (s->layers[s->size-1].activated[i] - expected[i]);
		s->slope1[i] *= s->layers[s->size-1].function.backward(s->layers[s->size-1].weighted[i]);
	}
	for (uint32_t i = s->size-1; i > 0; i--) {
		HAI_layer_backward(&s->layers[i], &s->layers[i-1], s->slope1, s->slope2, learning_rate);
		double *temp_slope = s->slope1;
		s->slope1 = s->slope2;
		s->slope2 = temp_slope;
	}
#pragma GCC diagnostic ignored "-Wdiscarded-qualifiers"
	HAI_layer_t input_layer = {
		.activated = inputs,
		.size = s->inputs_size,
	};
#pragma GCC diagnostic pop
	HAI_layer_backward(&s->layers[0], &input_layer, s->slope1, NULL, learning_rate);
}

double HAI_network_cost(HAI_network_t *s, const double *expected) {
	double cost = 0.0f;
	for (uint32_t i = 0; i < s->layers[s->size-1].size; i++) {
		double error = s->layers[s->size-1].activated[i] - expected[i];
		cost += error * error;
	}
	return cost;
}

uint32_t HAI_network_predict(HAI_network_t *s) {
	uint32_t predict = 0;
	double predict_value = 0.0f;
	for (uint32_t i = 0; i < s->layers[s->size-1].size; i++) {
		if (s->layers[s->size-1].activated[i] > predict_value) {
			predict_value = s->layers[s->size-1].activated[i];
			predict = i;
		}
	}
	return predict;
}
