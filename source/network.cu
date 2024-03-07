#include <cstdlib>
#include "network.cuh"

#define HAI_RANDOM() ((double)std::rand() / (double)RAND_MAX * 2.0 - 1.0)

namespace HAI
{
	Layer::Layer(std::size_t inputs, std::size_t outputs)
		: inputs(inputs)
		, outputs(outputs)
	{
		cudaMalloc(&this->weights, outputs * inputs * sizeof(double));
		cudaMalloc(&this->biases, outputs * sizeof(double));
		for (std::size_t i = 0; i < outputs; i++) {
			// TODO: Random values based on previous layer size!
			this->biases[i] = HAI_RANDOM();
			for (std::size_t j = 0; j < inputs; j++) {
				this->weights[i * inputs + j] = HAI_RANDOM();
			}
		}
	}

	Layer::~Layer()
	{
		cudaFree(this->weights);
		cudaFree(this->biases);
	}

	__device__ void Layer::Forward(const double* inputs, double* weighted, double* activated)
	{
		for (std::size_t i = 0; i < this->outputs; i++) {
			weighted[i] = 0.0;
			for (std::size_t j = 0; j < this->inputs; j++) {
				weighted[i] += inputs[j] * this->weights[j];
			}
			// TODO: Hardcoded Tanh function!
			activated[i] = tanh(weighted[i]);
		}
	}

	Network::~Network()
	{
		cudaFree(this->layers);
	}
}
