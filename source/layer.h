#ifndef HAI_LAYER_H_
#define HAI_LAYER_H_
#include <stdbool.h>
#include <stdint.h>
#include "neuron.h"

typedef struct {
	HAI_neuron_t *neurons;
	uint32_t neurons_size;
} HAI_layer_t;

HAI_layer_t HAI_layer_init(uint32_t neurons_size, uint32_t inputs_size, HAI_activation_t activate, HAI_activation_t derivative);
void HAI_layer_free(HAI_layer_t *s, uint32_t inputs_size);
double *HAI_layer_forward(HAI_layer_t *s, double *inputs, uint32_t inputs_size);
double **HAI_layer_forward_info(HAI_layer_t *s, double *inputs, uint32_t inputs_size);
double *HAI_layer_output_slope(HAI_layer_t *s, double *weighted_outputs, double *activated_outputs, double *expected_outputs);
double *HAI_layer_hidden_slope(HAI_layer_t *s, double *slope, double *weighted_inputs, double *activated_inputs, HAI_layer_t *prev_layer, double learning_rate, uint32_t inputs_size);

#endif//HAI_LAYER_H_
