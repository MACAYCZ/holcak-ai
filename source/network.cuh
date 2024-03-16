#ifndef HAI_NETWORK_CUH_
#define HAI_NETWORK_CUH_
#include <cstddef>
#include "layer.cuh"

namespace HAI
{
	class Trainer;

	class Network
	{
		friend Trainer;

	public:
		template <std::size_t N>
		Network(const std::size_t (&size)[N]);
		~Network();

	private:
		Layer* layers;
		std::size_t size;
	};

	template <std::size_t N>
	Network::Network(const std::size_t (&size)[N])
		: size(N-1)
	{
		// TODO: The initialization is really slow due to managed memory!
		// The solution for that is to make kernel that will initialize the memory!
		cudaMallocManaged(&this->layers, this->size * sizeof(Layer));
		for (std::size_t i = 0; i < this->size; i++) {
			this->layers[i] = Layer(size[i], size[i+1]);
		}
	}
}

#endif // HAI_NETWORK_CUH_
