#ifndef HAI_TRAINER_CUH_
#define HAI_TRAINER_CUH_
#include <cstddef>
#include "network.cuh"

// The network architecture cannot be modified durning the training process!
// TODO: Trainer could be a virtual class to split different training algorithms!

namespace HAI
{
	class Trainer
	{
	public:
		Trainer(Network network, std::size_t batch, std::size_t threads);
		~Trainer();

		void Step(const double* inputs, const double* outputs);

	private:
		struct Layer
		{
			double* weighted;
			double* activated;
		};

	private:
		Network network;
		std::size_t threads;
		std::size_t batch;
		Layer* layers;
	};
}

#endif // HAI_TRAINER_CUH_
