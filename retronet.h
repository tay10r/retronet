#ifndef RETRONET_H
#define RETRONET_H

namespace retronet {

/// @brief This class can be used to track memory usage or customizing memory allocation in the library.
class MemManager {
public:
	/// @brief If you do not plan on tracking memory or performing custom memory allocation, a pass-through memory manager
	///        returned by this function is used instead.
	static MemManager *get_default();

	virtual ~MemManager() = default;

	/// @brief Allocates a chunk of memory.
	///
	/// @param size The number of bytes to allocate.
	///
	/// @return A pointer the the allocated memory or a null pointer if the memory allocation failed.
	[[nodiscard]] virtual void *alloc(int size) = 0;

	/// @brief Releases a previously allocated memory block.
	///
	/// @param addr The address of the memory chunk to free.
	///
	/// @note The memory address may be a null pointer, in which case the memory manager should do nothing.
	virtual void release(void *addr) = 0;
};

/// @brief An interface for a random number generator.
class RNG {
public:
	virtual ~RNG() = default;

	/// @brief Produces a random integer within a given range.
	///
	/// @param min_v The minimum of the range to produce the value in.
	///
	/// @param max_v The maximum of the range to produce the value in.
	///
	/// @return A randomly chosen integer within the given range.
	virtual int randint(int min_v, int max_v) = 0;

	/// @brief Produces a random value uniformly within a given range.
	virtual float uniform(float min_v, float max_v) = 0;
};

/// @brief This enumerates the various types of loss functions that are supported.
enum class LossKind {
	MSE,
	CrossEntropy
};

/// @brief Represents a neural network.
/// @details To instantiate this class, see @ref Builder.
class Network {
public:
	virtual ~Network() = default;

	/// @brief Initializes the parameters in the network based on a uniform distribution.
	///
	/// @note This function is here for completeness, but it's not recommended over Xavier initialization. Consider using @ref Network::init_xavier instead.
	virtual void init_uniform(RNG &rng, float min_v, float max_v) = 0;

	/// @brief Initializes the parameters in the network based on Xavier initialization.
	///
	/// @note This is usually a good choice. If you're not sure how to initialize your network, just pick this approach.
	virtual void init_xavier(RNG &rng) = 0;

	/// @brief Runs the forward pass of the network.
	///
	/// @param input A pointer to the input vector.
	///
	/// @param input_rows The number of rows in the input vector.
	///
	/// @param output_rows The number of rows in the output vector.
	///
	/// @return A pointer to the output vector.
	[[nodiscard]] virtual const float *forward(const float *input, int input_rows, int *output_rows) = 0;

	/// @brief Computes the loss of the network and computes the gradients for each weight.
	///
	/// @param target The expected output vector of the network.
	///
	/// @param target_rows The number of rows in the expected output vector.
	///
	/// @param loss_kind The type of loss to compute.
	///
	/// @return The loss for the given target vector.
	virtual float backward(const float *target, int target_rows, LossKind loss_kind) = 0;

	/// @brief Scales the gradients in the network and subtracts their scaled values from the weights.
	///
	/// @param learning_rate How much to scale the gradients before subtracting them from the weights.
	virtual void gradient_descent(float learning_rate) = 0;

	/// @brief Zeros the gradients in the network.
	virtual void zero_grad() = 0;

	/// @brief Gets a pointer to the trained network parameters.
	[[nodiscard]] virtual float *parameters() = 0;

	/// @brief Gets a pointer to the trained network parameters.
	[[nodiscard]] virtual const float *parameters() const = 0;

	/// @brief Indicates the number of parameters in the network, in terms of the number of scalar values.
	[[nodiscard]] virtual int num_parameters() const = 0;
};

/// @brief Used for constructing a neural network.
class Builder {
public:
	/// @brief Instantiates a new network builder.
	///
	/// @param max_layers The maximum number of layers that the builder may construct. This value is used to indicate how
	///                   much memory should be allocated for the builder.
	///
	/// @param mem_manager The memory manager, used for allocating the builder data and the neural network data.
	[[nodiscard]] static Builder *make(int max_layers = 32, MemManager *mem_manager = MemManager::get_default());

	virtual ~Builder() = default;

	/// Removes all added layers, leaving the builder as if it was just instantiated.
	///
	/// @note This does not free any memory.
	virtual Builder &reset() = 0;

	Builder &linear(int in_features, int out_features);

	virtual Builder &matmul(int in_features, int out_features) = 0;

	virtual Builder &bias(int features) = 0;

	virtual Builder &relu() = 0;

	virtual Builder &sigmoid() = 0;

	[[nodiscard]] virtual Network *build() = 0;
};

struct VecView final {
	const float *data;
	int rows;

	bool empty() const;
};

struct Sample final {
	VecView input;

	VecView target;

	bool empty() const;
};

/// @brief This is an interface for a dataset, meant to be implemented by the caller.
///
/// @details The dataset class is used to train a network. The implementation of a dataset should produce network inputs
///          and expected network outputs (called targets) in pairs.
class Dataset {
public:
	virtual ~Dataset() = default;

	/// @brief Gets a sample (input and target pair) from the dataset at a given index.
	///
	/// @param index The index of the training sample to get.
	///
	/// @return The sample at the given index. If the index provided is out of bounds for the dataset, the dataset should
	///         return a sample whose input and target vectors have zero rows.
	[[nodiscard]] virtual Sample get_item(int index) const = 0;

	/// @brief Indicates to the caller the number of samples in the dataset.
	///
	/// @return The number of samples in the dataset.
	[[nodiscard]] virtual int len() const = 0;
};

class TrainingObserver {
public:
	virtual ~TrainingObserver() = default;

	virtual void on_training_progress(float completion, float avg_loss) = 0;
};

/// @brief Trains a module on a given dataset.
///
/// @param net The network to train.
///
/// @param d The dataset to train the network on.
///
/// @param learning_rate How much to scale the gradients by when performing gradient descent.
///
/// @param loss_kind The type of loss used for optimizing the network.
///
/// @param rng An optional random number generator to shuffle the samples around.
///
/// @return The average loss for the dataset and network.
float train(Network &net, Dataset &d, float learning_rate, LossKind loss_kind, RNG *rng = nullptr, TrainingObserver *observer = nullptr);

} // namespace retronet

#endif /* RETRONET_H */
