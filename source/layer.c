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

void HAI_layer_free(HAI_layer_t *s) {
	while (--s->neurons_size != UINT32_MAX) {
		HAI_neuron_free(&s->neurons[s->neurons_size]);
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

double **HAI_layer_forward_info(HAI_layer_t *s, double *inputs, uint32_t inputs_size) {
	double **outputs = malloc(2 * sizeof(double*));
	for (uint32_t i = 0; i < 2; i++) {
		outputs[i] = malloc(s->neurons_size * sizeof(double));
	}
	for (uint32_t i = 0; i < s->neurons_size; i++) {
		double *output = HAI_neuron_forward_info(&s->neurons[i], inputs, inputs_size);
		outputs[0][i] = output[0];
		outputs[1][i] = output[1];
		free(output);
	}
	return outputs;
}

double *HAI_layer_output_slope(HAI_layer_t *s, double *weighted_outputs, double *activated_outputs, double *expected_outputs) {
	double *slope = malloc(s->neurons_size * sizeof(double));
	for (uint32_t i = 0; i < s->neurons_size; i++) {
		slope[i] = s->neurons[i].derivative(weighted_outputs[i]) * 2.0f * (activated_outputs[i] - expected_outputs[i]);
	}
	return slope;
}

double *HAI_layer_hidden_slope(HAI_layer_t *s, double *slope, double *weighted_inputs, double *activated_inputs, HAI_layer_t *prev_layer, double learning_rate, uint32_t inputs_size) {
	for (uint32_t i = 0; i < s->neurons_size; i++) {
		HAI_neuron_update(&s->neurons[i], slope[i], activated_inputs, learning_rate, inputs_size);
	}
	if (weighted_inputs == NULL) {
		return NULL;
	}
	double *new_slope = malloc(inputs_size * sizeof(double));
	for (uint32_t i = 0; i < inputs_size; i++) {
		new_slope[i] = 0.0f;
		for (uint32_t j = 0; j < s->neurons_size; j++) {
			new_slope[i] += s->neurons[j].weights[i] * slope[i];
		}
		new_slope[i] *= prev_layer->neurons[i].derivative(weighted_inputs[i]);
	}
	return new_slope;
}
