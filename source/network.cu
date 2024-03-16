#include <memory>
#include "network.cuh"

namespace HAI
{
	Network::~Network()
	{
		std::destroy_n(this->layers, this->size);
		cudaFree(this->layers);
	}
}
