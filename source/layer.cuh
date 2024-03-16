#ifndef HAI_LAYER_CUH_
#define HAI_LAYER_CUH_
#include <cstddef>

namespace HAI
{
	class Trainer;

	class Layer
	{
		friend Trainer;

	public:
		Layer(std::size_t inputs, std::size_t outputs);
		~Layer();

	private:
		std::size_t inputs;
		std::size_t outputs;
		double* biases;
		double* weights;
	};
}

#endif // HAI_LAYER_CUH_
