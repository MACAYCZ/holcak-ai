#ifndef HAI_NEURON_H_
#define HAI_NEURON_H_
#include <stdint.h>
#include <stdlib.h>

#define HAI_RANDOM()((double)rand() / (double)RAND_MAX * 2.0f - 1.0f)

typedef struct {
	double bias;
	double *weights;
} HAI_neuron_t;

void HAI_neuron_init(HAI_neuron_t *s, uint32_t inputs_size);
void HAI_neuron_free(HAI_neuron_t *s);
double HAI_neuron_forward(HAI_neuron_t *s, const double *inputs, double inputs_size);
void HAI_neuron_update(HAI_neuron_t *s, const double *inputs, uint32_t inputs_size, double slope, double learning_rate);

#endif//HAI_NEURON_H_
