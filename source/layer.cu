#include "layer.cuh"

// TODO: Implement the He-et-al initialization!
// https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e

// TODO: Move the random to a singleton class!
#include <cstdlib>
// TODO: Try to generate only positive numbers!
#define HAI_RANDOM() ((double)std::rand() / (double)RAND_MAX * 2.0 - 1.0)

namespace HAI
{
	Layer::Layer(std::size_t inputs, std::size_t outputs)
		: inputs(inputs)
		, outputs(outputs)
	{
		cudaMallocManaged(&this->biases, outputs * sizeof(double));
		cudaMallocManaged(&this->weights, outputs * inputs * sizeof(double));
		for (std::size_t i = 0; i < outputs; i++) {
			this->biases[i] = HAI_RANDOM();
			for (std::size_t j = 0; j < inputs; j++) {
				this->weights[i * inputs + j] = HAI_RANDOM();
			}
		}
	}

	Layer::~Layer()
	{
		cudaFree(this->biases);
		cudaFree(this->weights);
	}
}