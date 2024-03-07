#ifndef HAI_NETWORK_CUH_
#define HAI_NETWORK_CUH_
#include <cstddef>

namespace HAI
{
	// TODO: Move layer to it's own file!
	class Layer
	{
	public:
		Layer(std::size_t inputs, std::size_t outputs);
		~Layer();

		__device__ void Forward(const double* inputs, double* weighted, double* activated);
		__device__ void Backward();
		__device__ void Update();

	public:
		// TODO: Make the variables private!
		std::size_t inputs;
		std::size_t outputs;
		double* weights;
		double* biases;
	};

	class Network
	{
	public:
		template <std::size_t N>
		Network(const std::size_t (&size)[N]) requires(N > 1);
		~Network();

	private:
		Layer* layers;
		std::size_t size;
	};

	template <std::size_t N>
	Network::Network(const std::size_t (&size)[N]) requires(N > 1)
	{
		cudaMallocManaged(&this->layers, N * sizeof(Layer));
		for (std::size_t i = 1; i < N; i++) {
			this->layers[i] = Layer(size[i-1], size[i]);
		}
		this->size = N;
	}
}

#endif // HAI_NETWORK_CUH_
