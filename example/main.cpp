#include <retronet.h>

#include "minst.h"

#include <iostream>
#include <memory>
#include <random>

#include <stdint.h>
#include <stdlib.h>

namespace {

class TrainingLogger final : public retronet::TrainingObserver {
public:
	void on_training_progress(const float completion, const float avg_loss) override {
		std::cout << "%" << static_cast<int>(completion * 100) << ": " << avg_loss << std::endl;
	}
};

// Optionally defined in order to improve training
class RNG final : public retronet::RNG {
	std::mt19937 rng_;

public:
	explicit RNG(int seed) :
			rng_(seed) {}

	int randint(int min_v, int max_v) override {
		std::uniform_int_distribution<int> dist(min_v, max_v);
		return dist(rng_);
	}

	float uniform(const float min_v, const float max_v) override {
		std::uniform_real_distribution<float> dist(min_v, max_v);
		return dist(rng_);
	}
};

} // namespace

int main() {
	RNG rng(/*seed=*/0);
	std::unique_ptr<retronet::Builder> builder(retronet::Builder::make());

	builder->reset()
			.linear(28 * 28, 256)
			.relu()
			.linear(256, 128)
			.relu()
			.linear(128, 10);

	std::unique_ptr<retronet::Network> net(builder->build());

	net->init_xavier(rng);

	MINSTDataset train_dataset;

	if (!train_dataset.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte")) {
		std::cout << "Failed to load training dataset." << std::endl;
		return EXIT_FAILURE;
	}

	const float learning_rate{ 0.001F };

	const int num_epochs{ 16 };

	TrainingLogger logger;

	for (int i = 0; i < num_epochs; i++) {
		const float train_loss = retronet::train(*net, train_dataset, learning_rate, retronet::LossKind::CrossEntropy, &rng, &logger);
		std::cout << "train loss: " << train_loss << std::endl;
	}

	return EXIT_SUCCESS;
}
