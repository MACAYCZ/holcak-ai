#include "layer.h"
#include <stdlib.h>

void HAI_layer_init(HAI_layer_t *s, uint32_t inputs_size, HAI_function_t function, uint32_t size) {
	s->function = function;
	s->neurons = malloc(size * sizeof(HAI_neuron_t));
	for (uint32_t i = 0; i < size; i++) {
		HAI_neuron_init(&s->neurons[i], inputs_size);
	}
	s->size = size;
	s->weighted = malloc(size * sizeof(double));
	s->activated = malloc(size * sizeof(double));
}

void HAI_layer_free(HAI_layer_t *s) {
	for (uint32_t i = 0; i < s->size; i++) {
		HAI_neuron_free(&s->neurons[i]);
	}
	free(s->neurons);
	free(s->weighted);
	free(s->activated);
}

const double *HAI_layer_forward(HAI_layer_t *s, const double *inputs, uint32_t inputs_size) {
	for (uint32_t i = 0; i < s->size; i++) {
		s->weighted[i] = HAI_neuron_forward(&s->neurons[i], inputs, inputs_size);
		s->activated[i] = s->function.forward(s->weighted[i]);
	}
	return s->activated;
}

void HAI_layer_backward(HAI_layer_t *s, HAI_layer_t *p, const double *slope, double *slope_out, double learning_rate) {
	if (p->weighted != NULL) {
		for (uint32_t i = 0; i < p->size; i++) {
			slope_out[i] = 0.0f;
			for (uint32_t j = 0; j < s->size; j++) {
				slope_out[i] += s->neurons[j].weights[i] * slope[j];
			}
			slope_out[i] *= p->function.backward(p->weighted[i]);
		}
	}
	for (uint32_t i = 0; i < s->size; i++) {
		HAI_neuron_update(&s->neurons[i], p->activated, p->size, slope[i], learning_rate);
	}
}
