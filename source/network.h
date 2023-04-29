#ifndef HAI_NETWORK_H_
#define HAI_NETWORK_H_
#include <stdint.h>
#include "layer.h"

typedef struct {
	uint32_t inputs_size;
	HAI_layer_t *layers;
	uint32_t size;

	double *slope1;
	double *slope2;
} HAI_network_t;

void HAI_network_init_(HAI_network_t *s, uint32_t inputs_size, ...);
#define HAI_network_init(S, InputsSize, ...)(HAI_network_init_(S, InputsSize, __VA_ARGS__, HAI_FUNCTION_NULL))
void HAI_network_free(HAI_network_t *s);
const double *HAI_network_forward(HAI_network_t *s, const double *inputs);
void HAI_network_backward(HAI_network_t *s, const double *inputs, const double *expected, double learning_rate);
double HAI_network_cost(HAI_network_t *s, const double *expected);
uint32_t HAI_network_predict(HAI_network_t *s);

#endif//HAI_NETWORK_H_
