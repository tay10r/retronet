// This file is just for testing the code written to read the MINST dataset.
// There is nothing about neural networks in this test program.

#include "minst.h"

#include <iostream>
#include <sstream>
#include <vector>

#include <cstdlib>

#include "stb_image_write.h"

int main() {
	MINSTDataset train_dataset;

	if (!train_dataset.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte")) {
		std::cout << "Failed to load training dataset." << std::endl;
		return EXIT_FAILURE;
	}

	const int num_samples = 4;

	for (int i = 0; i < num_samples; i++) {
		const retronet::Sample s = train_dataset.get_item(i);
		std::vector<uint8_t> buffer(28 * 28);
		for (int i = 0; i < buffer.size(); i++) {
			buffer[i] = static_cast<uint8_t>(s.input.data[i] * 255);
		}
		std::ostringstream path_stream;
		path_stream << "sample_" << i << ".png";
		stbi_write_png(path_stream.str().c_str(), 28, 28, 1, buffer.data(), 28);
	}

	return EXIT_SUCCESS;
}
