#ifndef HAI_LAYER_H_
#define HAI_LAYER_H_
#include <stdint.h>
#include "function.h"
#include "neuron.h"

typedef struct {
	HAI_function_t function;
	HAI_neuron_t *neurons;
	uint32_t size;

	double *weighted;
	double *activated;
} HAI_layer_t;

void HAI_layer_init(HAI_layer_t *s, uint32_t inputs_size, HAI_function_t function, uint32_t size);
void HAI_layer_free(HAI_layer_t *s);
const double *HAI_layer_forward(HAI_layer_t *s, const double *inputs, uint32_t inputs_size);
void HAI_layer_backward(HAI_layer_t *s, HAI_layer_t *p, const double *slope, double *slope_out, double learning_rate);

#endif//HAI_LAYER_H_
