#include "trainer.cuh"

namespace HAI
{
	Trainer::Trainer(Network network, std::size_t batch, std::size_t threads)
		: network(network)
		, batch(batch)
		, threads(threads)
	{
		cudaMallocManaged(&this->layers, network.size * sizeof(Layer));
		for (std::size_t i = 0; i < network.size; i++) {
			cudaMallocManaged(&this->layers[i].weighted, batch * network.layers[i].outputs * sizeof(double));
			cudaMallocManaged(&this->layers[i].activated, batch * network.layers[i].outputs * sizeof(double));
		}
	}

	Trainer::~Trainer()
	{
		cudaFree(&this->layers);
	}

	__global__ void Calculate()
	{
		// TODO!
	}

	__global__ void Update()
	{
		// TODO!
	}

	void Trainer::Step(const double* inputs, const double* outputs)
	{
		// TODO!
	}
}
