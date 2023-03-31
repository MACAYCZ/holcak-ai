#ifndef HAI_NETWORK_H_
#define HAI_NETWORK_H_
#include <stdint.h>
#include "layer.h"

typedef struct {
	HAI_layer_t *layers;
	uint32_t layers_size;
	uint32_t inputs_size;
} HAI_network_t;

HAI_network_t HAI_network_init(uint32_t layers_size, uint32_t inputs_size, ...);
void HAI_network_free(HAI_network_t *s);
double *HAI_network_forward(HAI_network_t *s, double *inputs);
void HAI_network_backward(HAI_network_t *s, double *inputs, double *expected, double learning_rate);
double HAI_network_cost(HAI_network_t *s, double *inputs, double *expected);

#endif//HAI_NETWORK_H_
