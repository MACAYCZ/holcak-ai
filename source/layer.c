#include "layer.h"
#include <stdlib.h>

void HAI_layer_init(HAI_layer_t *s, uint32_t inputs_size, HAI_function_t function, uint32_t size) {
	s->function = function;
	s->weights = malloc(size * sizeof(double*));
	s->bias = malloc(size * sizeof(double*));
	for (uint32_t i = 0; i < size; i++) {
		s->weights[i] = malloc(inputs_size * sizeof(double));
		for (uint32_t j = 0; j < inputs_size; j++) {
			s->weights[i][j] = HAI_RANDOM();
		}
		s->bias[i] = HAI_RANDOM();
	}
	s->size = size;
	s->weighted = malloc(size * sizeof(double));
	s->activated = malloc(size * sizeof(double));
}

void HAI_layer_free(HAI_layer_t *s) {
	for (uint32_t i = 0; i < s->size; i++) {
		free(s->weights[i]);
	}
	free(s->weights);
	free(s->bias);
	free(s->weighted);
	free(s->activated);
}

const double *HAI_layer_forward(HAI_layer_t *s, const double *inputs, uint32_t inputs_size) {
	for (uint32_t i = 0; i < s->size; i++) {
		s->weighted[i] = s->bias[i];
		for (uint32_t j = 0; j < inputs_size; j++) {
			s->weighted[i] += s->weights[i][j] * inputs[j];
		}
		s->activated[i] = s->function.forward(s->weighted[i]);
	}
	return s->activated;
}

void HAI_layer_backward(HAI_layer_t *s, HAI_layer_t *p, const double *slope, double *slope_out, double learning_rate) {
	if (slope_out != NULL) {
		for (uint32_t i = 0; i < p->size; i++) {
			slope_out[i] = 0.0f;
			for (uint32_t j = 0; j < s->size; j++) {
				slope_out[i] += s->weights[j][i] * slope[j];
			}
			slope_out[i] *= p->function.backward(p->weighted[i]);
		}
	}
	for (uint32_t i = 0; i < s->size; i++) {
		double learning_slope = slope[i] * learning_rate;
		for (uint32_t j = 0; j < p->size; j++) {
			s->weights[i][j] -= p->activated[j] * learning_slope;
		}
		s->bias[i] -= learning_slope;
	}
}
