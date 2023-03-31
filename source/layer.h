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
double *HAI_layer_backward(HAI_layer_t *s, double *inputs, uint32_t inputs_size, HAI_layer_t *next_layer, double *delta, double learning_rate, bool output_delta);

#endif//HAI_LAYER_H_
