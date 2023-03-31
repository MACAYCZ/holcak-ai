#ifndef HAI_NEURON_H_
#define HAI_NEURON_H_
#include <stdint.h>
#include "activation.h"

typedef struct {
	double bias, *weights;
	HAI_activation_t activate;
	HAI_activation_t derivative;
} HAI_neuron_t;

HAI_neuron_t HAI_neuron_init(uint32_t inputs_size, HAI_activation_t activate, HAI_activation_t derivative);
void HAI_neuron_free(HAI_neuron_t *s);
double HAI_neuron_weighted(HAI_neuron_t *s, double *inputs, uint32_t inputs_size);
double HAI_neuron_forward(HAI_neuron_t *s, double *inputs, uint32_t inputs_size);

#endif//HAI_NEURON_H_
