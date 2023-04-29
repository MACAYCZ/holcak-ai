#include "hai/network.h"
#include <stdio.h>
#include <time.h>

void debug_print(HAI_network_t *s) {
	uint32_t inputs_size = s->inputs_size;
	for (uint32_t i = 0; i < s->size; i++) {
		const HAI_layer_t *l = &s->layers[i];
		printf("Layer %u:\n", i);
		for (uint32_t j = 0; j < l->size; j++) {
			printf("  Neuron %u:\n", j);
			printf("    Bias: %.10f\n", l->bias[j]);
			printf("    Weights:\n");
			for (uint32_t k = 0; k < inputs_size; k++) {
				printf("      Weight %u: %.10f\n", k, l->weights[j][k]);
			}
		}
		inputs_size = l->size;
	}
}

void load_inputs(const char *path, double **inputs, uint32_t size) {
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "Error: Opening `%s` failed!\n", path);
		exit(EXIT_FAILURE);
	}
	fseek(file, 16, SEEK_SET);
	for (uint32_t i = 0; i < size; i++) {
		inputs[i] = malloc(28*28 * sizeof(double));
		for (uint32_t j = 0; j < 28*28; j++) {
			inputs[i][j] = (double)fgetc(file) / (double)UINT8_MAX;
		}
	}
	fclose(file);
}

void load_labels(const char *path, double **labels, uint32_t size) {
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "Error: Opening `%s` failed!\n", path);
		exit(EXIT_FAILURE);
	}
	fseek(file, 8, SEEK_SET);
	for (uint32_t i = 0; i < size; i++) {
		labels[i] = malloc(10 * sizeof(double));
		uint32_t v = fgetc(file);
		for (uint32_t j = 0; j < 10; j++) {
			labels[i][j] = (double)(v == j);
		}
	}
	fclose(file);
}

int main(void) {
	srand(time(NULL));

	HAI_network_t network;
	HAI_network_init(&network, 28*28, HAI_SIGMOID, 100, HAI_SIGMOID, 10);

	double **inputs = malloc(70000 * sizeof(double*));
	double **labels = malloc(70000 * sizeof(double*));
	load_inputs("datasets/mnist/train-images.idx3-ubyte", inputs, 60000);
	load_inputs("datasets/mnist/t10k-images.idx3-ubyte", inputs + 60000, 10000);
	load_labels("datasets/mnist/train-labels.idx1-ubyte", labels, 60000);
	load_labels("datasets/mnist/t10k-labels.idx1-ubyte", labels + 60000, 10000);
	printf("Training data successfully loaded!\n");

	const uint32_t iterations = 10000;
	const double learning_rate = 0.001f;

	for (uint32_t i = 0; i < iterations; i++) {
		double training_cost = 0.0f;
		uint32_t training_correct = 0;
		for (uint32_t j = 0; j < 60000; j++) {
			HAI_network_backward(&network, inputs[j], labels[j], learning_rate);
			training_cost += HAI_network_cost(&network, labels[j]);
			training_correct += labels[j][HAI_network_predict(&network)];
		}
		double testing_cost = 0.0f;
		uint32_t testing_correct = 0;
		for (uint32_t j = 0; j < 10000; j++) {
			HAI_network_forward(&network, inputs[j + 60000]);
			testing_cost += HAI_network_cost(&network, labels[j + 60000]);
			testing_correct += labels[j + 60000][HAI_network_predict(&network)];
		}
		training_cost /= 60000.0f;
		testing_cost /= 10000.0f;
		printf("Training %3u: %5u / 60000: %f\n", i, training_correct, training_cost);
		printf("Testing  %3u: %5u / 10000: %f\n\n", i, testing_correct, testing_cost);
	}

	debug_print(&network);
	printf("Freeing allocated memory!\n");
	for (uint32_t i = 0; i < 70000; i++) {
		free(inputs[i]);
		free(labels[i]);
	}
	free(inputs);
	free(labels);
	HAI_network_free(&network);
	return 0;
}
