#include <iostream>
#include "source/tensor.cuh"

/*
 * Building:
 * 	https://stackoverflow.com/questions/6319274/how-do-i-run-msbuild-from-the-command-line-using-windows-sdk-7-1
 * 	https://stackoverflow.com/questions/498106/how-do-i-compile-a-visual-studio-project-from-the-command-line
 *	https://cgold.readthedocs.io/en/latest/tutorials/libraries/static-shared.html
 *  https://stackoverflow.com/questions/17511496/how-to-create-a-shared-library-with-cmake
 *  https://stackoverflow.com/questions/50028570/is-it-possible-to-build-cmake-projects-directly-using-msbuildtools
 */

int main(void)
{
	HAI::Tensor<2> tensor({10, 10});
	for (std::size_t i = 0; i < tensor.Size(0); i++) {
		for (std::size_t j = 0; j < tensor.Size(1); j++) {
			(double&)tensor[i][j] = i * tensor.Size(0) + j;
		}
	}
	for (std::size_t i = 0; i < tensor.Surface(); i++) {
		std::cout << tensor.Data()[i] << std::endl;
	}
	return 0;
}
