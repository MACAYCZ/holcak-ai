#include <iostream>
#include <chrono>
#include "source/trainer.cuh"

// TODO: Implement a timer class to the framework!
[[maybe_unused]] static std::chrono::time_point<std::chrono::high_resolution_clock> __timer;
#define TIMER_RESET() \
	do { \
		using namespace std::chrono; \
		__timer = high_resolution_clock::now(); \
	} while (0)
#define TIMER_PRINT() \
	do { \
		using namespace std::chrono; \
		std::cout << "Time elapsed: " << duration_cast<milliseconds>(high_resolution_clock::now() - __timer).count() << "ms" << std::endl; \
	} while (0)

int main()
{
	TIMER_RESET();
	HAI::Network network({2, 2, 1});
	HAI::Trainer trainer(network, 4, 4);
	TIMER_PRINT();
	return 0;
}
