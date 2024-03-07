#include <cstdio>
#include "source/network.cuh"

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
	HAI::Network network({
		2,
		2,
	});
	/*	
	for (std::size_t i = 0; i < 10; i++) {
		// TODO: This isn't possible because kernels cannot be class members!
		network.Gradient<<<1, 1024>>>();
		network.Update<<<1, 1024>>>();
	}
	*/
	return 0;
}
