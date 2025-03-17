#include "retronet.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace retronet {

namespace {

class MemManagerImpl final : public MemManager {
public:
	void *alloc(int size) override {
		return malloc(static_cast<size_t>(size));
	}

	void release(void *addr) override {
		free(addr);
	}
};

template <typename T>
T *alloc(MemManager *mem, const int num_elements) {
	return static_cast<T *>(mem->alloc(sizeof(T) * num_elements));
}

} // namespace

MemManager *MemManager::get_default() {
	static MemManagerImpl def;
	return &def;
}

namespace {

void matvec(const float *a, const int a_rows, const int a_cols,
		const float *vec, float *out) {
	// Initialize out to zero
	for (int i = 0; i < a_rows; i++) {
		out[i] = 0.0f;
	}
	// out[i] += a[col_index + i] * vec[j]
	for (int j = 0; j < a_cols; j++) {
		float scalar = vec[j];
		int col_index = j * a_rows;
		for (int i = 0; i < a_rows; i++) {
			out[i] += a[col_index + i] * scalar;
		}
	}
}

enum class LayerKind {
	matmul,
	bias,
	relu,
	sigmoid
};

struct MatMulLayer final {
	int in_features;
	int out_features;
};

struct BiasLayer final {
	int features;
};

struct AnyLayer final {
	LayerKind kind;
	union {
		MatMulLayer matmul;
		BiasLayer bias;
	};
};

// Returns how many weight parameters a particular layer has
int num_params(const AnyLayer &l) {
	switch (l.kind) {
		case LayerKind::matmul:
			return l.matmul.in_features * l.matmul.out_features;
		case LayerKind::bias:
			return l.bias.features;
		case LayerKind::relu:
		case LayerKind::sigmoid:
			// activation layers have no trainable params
			return 0;
	}
	return 0; // fallback
}

// Sums the total number of weight parameters across all layers
int num_params(const AnyLayer *layers, const int num_layers) {
	int n = 0;
	for (int i = 0; i < num_layers; i++) {
		n += num_params(layers[i]);
	}
	return n;
}

// Returns how many outputs (a.k.a. “activation values”) a layer produces
// so that we know how big each layer’s output is.
static int output_size(const AnyLayer &l, int input_size) {
	switch (l.kind) {
		case LayerKind::matmul:
			// MatMul layer’s output is out_features
			return l.matmul.out_features;
		case LayerKind::bias:
			// must match the bias features
			return l.bias.features;
		case LayerKind::relu:
		case LayerKind::sigmoid:
			// these activation layers do not change the size;
			// they match the input's dimension.
			return input_size;
	}
	return 0; // fallback
}

//--------------------------------------
// Forward Declaration
//--------------------------------------
class NetworkImpl final : public Network {
	MemManager *mem_manager_;
	AnyLayer *layers_;
	const int num_layers_;
	const int num_parameters_;

	float *parameters_; // Flattened array of all layer weights
	float *grad_parameters_; // Flattened array of weight gradients

	// We store all intermediate outputs in a single buffer.
	float *buffer_; // forward activations
	float *grad_buffer_; // gradient of each activation in backward pass

	// For convenience in backprop, we keep track of each layer’s output offset
	// and output size in these arrays.
	int *offsets_;
	int *sizes_;

	// The total number of floats we need in buffer_ (and grad_buffer_)
	const int num_values_;

public:
	NetworkImpl(AnyLayer *layers, const int num_layers, MemManager *mem_manager) :
			mem_manager_(mem_manager), layers_(layers), num_layers_(num_layers), num_parameters_(num_params(layers_, num_layers)), parameters_(alloc<float>(mem_manager_, num_parameters_)), grad_parameters_(alloc<float>(mem_manager_, num_parameters_)), offsets_(alloc<int>(mem_manager_, num_layers_ + 1)), sizes_(alloc<int>(mem_manager_, num_layers_)), num_values_(compute_total_values(layers, num_layers)) {
		buffer_ = alloc<float>(mem_manager_, num_values_);
		grad_buffer_ = alloc<float>(mem_manager_, num_values_);

		// Initialize parameters and grads to something (e.g., zero)
		if (parameters_) {
			memset(parameters_, 0, sizeof(float) * num_parameters_);
		}
		if (grad_parameters_) {
			memset(grad_parameters_, 0, sizeof(float) * num_parameters_);
		}
		if (buffer_) {
			memset(buffer_, 0, sizeof(float) * num_values_);
		}
		if (grad_buffer_) {
			memset(grad_buffer_, 0, sizeof(float) * num_values_);
		}

		// Precompute layer offsets in the buffer
		if (offsets_) {
			// offsets_[0] = 0 => the first layer's output is stored at buffer_[0]
			offsets_[0] = 0;
		}
	}

	~NetworkImpl() {
		mem_manager_->release(layers_);
		mem_manager_->release(parameters_);
		mem_manager_->release(grad_parameters_);
		mem_manager_->release(buffer_);
		mem_manager_->release(grad_buffer_);
		mem_manager_->release(offsets_);
		mem_manager_->release(sizes_);
	}

	void init_uniform(RNG &rng, const float min_v, const float max_v) override {
		for (auto i = 0; i < num_parameters_; i++) {
			parameters_[i] = rng.uniform(min_v, max_v);
		}
	}

	void init_xavier(RNG &rng) override {
		float *p = parameters_;

		for (int i = 0; i < num_layers_; i++) {
			const AnyLayer &layer = layers_[i];
			if (layer.kind == LayerKind::matmul) {
				const int in_features{ layer.matmul.in_features };
				const int out_features{ layer.matmul.out_features };
				const int num_features{ in_features * out_features };
				const float alpha{ sqrtf(6.0F / static_cast<float>(in_features + out_features)) };
				for (int j = 0; j < num_features; j++) {
					p[j] = rng.uniform(-alpha, alpha);
				}
				p += num_features;
			} else if (layer.kind == LayerKind::bias) {
				for (int j = 0; j < layer.bias.features; j++) {
					p[j] = rng.uniform(0, 1);
				}
				p += layer.bias.features;
			}
		}
	}

	bool allocated() const {
		return (parameters_ && grad_parameters_ &&
				buffer_ && grad_buffer_ &&
				offsets_ && sizes_);
	}

	const float *forward(const float *input, const int input_rows, int *output_rows) override {
		if (!allocated()) {
			if (output_rows) {
				*output_rows = 0;
			}
			return nullptr;
		}

		// We'll build up the outputs layer by layer in buffer_
		// 'in_ptr' will point to the input for the current layer
		// 'out_ptr' will be the location in buffer_ where we write the layer's output
		const float *in_ptr = input;
		int in_size = input_rows;

		float *out_ptr = buffer_;

		const float *params = parameters_;
		int param_offset = 0; // tracks where we are in parameters_

		for (int i = 0; i < num_layers_; i++) {
			offsets_[i] = static_cast<int>(out_ptr - buffer_); // store offset
			int layer_out_size = output_size(layers_[i], in_size);
			sizes_[i] = layer_out_size; // store the size for backprop

			switch (layers_[i].kind) {
				case LayerKind::matmul: {
					const int out_features = layers_[i].matmul.out_features;
					const int in_features = layers_[i].matmul.in_features;
					// compute out = matvec(W, out_features, in_features, in, out)
					matvec(params, out_features, in_features, in_ptr, out_ptr);
					// advance param pointer by # of weights
					params += out_features * in_features;
					param_offset += out_features * in_features;
					break;
				}
				case LayerKind::bias: {
					const int features = layers_[i].bias.features;
					for (int f = 0; f < features; f++) {
						out_ptr[f] = in_ptr[f] + params[f];
					}
					params += features;
					param_offset += features;
					break;
				}
				case LayerKind::relu: {
					for (int f = 0; f < in_size; f++) {
						float x = in_ptr[f];
						out_ptr[f] = (x > 0.0f) ? x : 0.0f;
					}
					break;
				}
				case LayerKind::sigmoid: {
					for (int f = 0; f < in_size; f++) {
						float x = in_ptr[f];
						out_ptr[f] = 1.0f / (1.0f + expf(-x));
					}
					break;
				}
			}

			// Prepare for next layer
			in_ptr = out_ptr; // next layer’s input is this layer’s output
			in_size = layer_out_size;
			out_ptr += layer_out_size;
		}

		// The final layer offset:
		offsets_[num_layers_] = static_cast<int>(out_ptr - buffer_);

		if (output_rows) {
			*output_rows = in_size;
		}
		// The final output is stored at 'in_ptr' now.
		return in_ptr;
	}

	float backward(const float *target, const int target_rows, const LossKind loss_kind) override {
		if (!allocated()) {
			return static_cast<float>(INFINITY);
		}

		// Check final output size vs. target size
		int final_size = sizes_[num_layers_ - 1];
		if (final_size != target_rows) {
			// mismatch
			return static_cast<float>(INFINITY);
		}

		// 1) Compute the loss and the gradient w.r.t the final output
		float loss = 0.0f;
		float *final_out = buffer_ + offsets_[num_layers_ - 1]; // final layer’s output
		float *final_out_grad = grad_buffer_ + offsets_[num_layers_ - 1]; // gradient wrt final out

		const float eps = 1e-12f; // to avoid log(0) in cross-entropy

		if (loss_kind == LossKind::MSE) {
			// MSE: (1/2)*sum((y - t)^2), derivative w.r.t y is (y - t)
			for (int i = 0; i < final_size; i++) {
				float diff = final_out[i] - target[i];
				loss += 0.5f * diff * diff;
				final_out_grad[i] = diff; // derivative w.r.t. output
			}
		} else if (loss_kind == LossKind::CrossEntropy) {
			// We interpret 'final_out' as raw logits (no sigmoid at final layer).
			// We need to compute softmax(prob), then cross-entropy, and fill final_out_grad.

			// 1) Find the maximum logit to improve numerical stability
			float max_logit = -FLT_MAX;
			for (int i = 0; i < final_size; i++) {
				if (final_out[i] > max_logit) {
					max_logit = final_out[i];
				}
			}

			// 2) Compute denominator = sum of exp(logit - max_logit)
			float sum_exp = 0.0f;
			for (int i = 0; i < final_size; i++) {
				sum_exp += expf(final_out[i] - max_logit);
			}

			// 3) Compute probabilities, cross-entropy loss, and gradient
			loss = 0.0f;
			for (int i = 0; i < final_size; i++) {
				float t = target[i]; // one-hot target
				float logit = final_out[i];
				float p = expf(logit - max_logit) / sum_exp; // softmax probability

				// cross-entropy = - sum_i [ t_i * log(p_i) ]
				// accumulate the total loss
				loss -= t * logf(p + eps);

				// derivative wrt logit = p - t
				final_out_grad[i] = p - t;
			}
		}

		if (isnan(loss)) {
			return loss;
		}

		// 2) Back-prop through each layer in reverse order
		float *params = parameters_;
		float *params_grad = grad_parameters_;

		// We'll track the param offset as we go
		// but since we’re going backwards, let's precompute total param count
		int param_offset_total = 0;
		for (int i = 0; i < num_layers_; i++) {
			param_offset_total += num_params(layers_[i]);
		}

		for (int layer_i = num_layers_ - 1; layer_i >= 0; layer_i--) {
			// size of this layer’s output
			int out_size = sizes_[layer_i];
			// offset in buffer for output
			int out_offset = offsets_[layer_i];
			// offset in grad_buffer_ for output gradient
			float *out_grad = grad_buffer_ + out_offset;
			float *out_data = buffer_ + out_offset;

			// The input to this layer is the output of the previous layer
			// or the original user input if layer_i == 0
			float *in_data = nullptr;
			float *in_grad = nullptr;
			int in_offset = 0;
			int in_size = 0;

			if (layer_i == 0) {
				// The first layer’s input is the user’s input to forward(),
				// which we do NOT store in buffer_, so we can’t compute its gradient
				// unless you want to (like for an autoencoder, etc.).
				// We'll treat it like there's no gradient to pass further back.
				// If you do want input gradients, you'd store them similarly in buffer_
				// or do something else fancy.
				in_data = nullptr;
				in_grad = nullptr;
				in_size = layers_[layer_i].kind == LayerKind::matmul
						? layers_[layer_i].matmul.in_features
						: (layers_[layer_i].kind == LayerKind::bias
										  ? layers_[layer_i].bias.features
										  : out_size);
			} else {
				// The input to this layer is at offsets_[layer_i - 1]
				in_offset = offsets_[layer_i - 1];
				in_size = sizes_[layer_i - 1];
				in_data = buffer_ + in_offset;
				in_grad = grad_buffer_ + in_offset;
			}

			// Now let’s see how many parameters belong to this layer
			int layer_params = num_params(layers_[layer_i]);
			// The param block for layer i is right before param_offset_total in the flattened array
			int param_block_begin = param_offset_total - layer_params;
			float *layer_param = params + param_block_begin;
			float *layer_param_grad = params_grad + param_block_begin;

			switch (layers_[layer_i].kind) {
				case LayerKind::sigmoid: {
					// The derivative of sigmoid is: dsigmoid(x)/dx = y*(1-y),
					// so dIn = dOut * (out * (1-out))
					for (int j = 0; j < out_size; j++) {
						float y = out_data[j];
						float dO = out_grad[j];
						// chain rule
						float dI = dO * y * (1.0f - y);
						out_grad[j] = dI; // re-use out_grad to store the gradient wrt input
					}
					// If there's a previous layer, accumulate out_grad => in_grad
					if (in_grad) {
						// The next layer sees the gradient as out_grad of the next layer
						for (int j = 0; j < out_size; j++) {
							in_grad[j] += out_grad[j];
						}
					}
					break;
				}
				case LayerKind::relu: {
					// dIn = dOut * (out > 0 ? 1 : 0)
					for (int j = 0; j < out_size; j++) {
						float x = out_data[j];
						float dO = out_grad[j];
						float dI = (x > 0.0f) ? dO : 0.0f;
						out_grad[j] = dI;
					}
					if (in_grad) {
						for (int j = 0; j < out_size; j++) {
							in_grad[j] += out_grad[j];
						}
					}
					break;
				}
				case LayerKind::bias: {
					// output[i] = input[i] + bias[i]
					// so dBias[i] = dOut[i], dIn[i] = dOut[i]
					for (int f = 0; f < out_size; f++) {
						float dO = out_grad[f];
						// param gradient
						layer_param_grad[f] += dO;
						// pass back to in_grad if it exists
						if (in_grad) {
							in_grad[f] += dO;
						}
					}
					break;
				}
				case LayerKind::matmul: {
					// out = W * in
					// out size = out_features, in size = in_features
					int out_features = layers_[layer_i].matmul.out_features;
					int in_features = layers_[layer_i].matmul.in_features;

					// The gradient wrt input: dIn = W^T * dOut
					// The gradient wrt param: dW = dOut * in^T
					// But we're in 1D form, so we have to do indexing carefully.

					// 1) Compute input gradient if we have a previous layer
					if (in_grad) {
						for (int in_idx = 0; in_idx < in_features; in_idx++) {
							float sum_grad = 0.0f;
							for (int out_idx = 0; out_idx < out_features; out_idx++) {
								float dO = out_grad[out_idx];
								float w = layer_param[in_idx * out_features + out_idx];
								sum_grad += dO * w;
							}
							in_grad[in_idx] += sum_grad;
						}
					}

					// 2) Compute param gradient
					// layer_param is shaped [out_features, in_features] in row-major?
					// Actually from matvec usage: param[out_idx * in_features + in_idx].
					// dW_{out_idx, in_idx} = dOut[out_idx] * in[in_idx]
					for (int out_idx = 0; out_idx < out_features; out_idx++) {
						float dO = out_grad[out_idx];
						for (int in_idx = 0; in_idx < in_features; in_idx++) {
							// the null pointer check is because we do not store the network input
							float val_in = in_data ? in_data[in_idx] : 0.0f;
							layer_param_grad[in_idx * out_features + out_idx] += dO * val_in;
						}
					}
					break;
				}
			}

			// done with this layer
			// update param_offset_total
			param_offset_total -= layer_params;
		}

		return loss;
	}

	void gradient_descent(const float learning_rate) override {
		if (!allocated()) {
			return;
		}
		for (int i = 0; i < num_parameters_; i++) {
			parameters_[i] -= grad_parameters_[i] * learning_rate;
		}
	}

	void zero_grad() override {
		// Zero out the activation grads
		if (grad_buffer_) {
			memset(grad_buffer_, 0, sizeof(float) * num_values_);
		}
		// Optionally zero out param grads
		if (grad_parameters_) {
			memset(grad_parameters_, 0, sizeof(float) * num_parameters_);
		}
	}

	float *parameters() override {
		return parameters_;
	}

	const float *parameters() const override {
		return parameters_;
	}

	int num_parameters() const override {
		return num_parameters_;
	}

private:
	// Helper: figure out how many “activation” floats we need in total
	// to store all intermediate outputs.
	static int compute_total_values(const AnyLayer *layers, int n) {
		// We'll simulate a forward pass in terms of sizes
		// Start with some “worst-case” guess or just do a pass to sum up
		// all outputs from each layer
		// Actually we do need the input size here in real usage, but we
		// can guess a maximum or store it when forward is actually called.
		// For simplicity, we'll return a big enough chunk, or you could
		// handle that more elegantly in real code.
		// For a “safe” guess, we might do 1,024 or so.
		// But let’s do the sum-of-out_features approach:
		// Obviously, in a real library, you'd do something more robust.
		int total = 0;
		int in_size = 1024; // just a placeholder (or store max somewhere)
		for (int i = 0; i < n; i++) {
			int out_size = output_size(layers[i], in_size);
			total += out_size;
			in_size = out_size;
		}
		return (total > 0) ? total : 1; // can't be zero
	}
};

class BuilderImpl final : public Builder {
	MemManager *mem_manager_;
	AnyLayer *layers_;
	int num_layers_;
	int max_layers_;
	bool failure_;

public:
	BuilderImpl(int max_layers, MemManager *mem_manager) :
			mem_manager_(mem_manager), layers_(alloc<AnyLayer>(mem_manager_, max_layers)), num_layers_(0), max_layers_(layers_ ? max_layers : 0), failure_(false) {
	}

	~BuilderImpl() {
		mem_manager_->release(layers_);
	}

	Builder &reset() override {
		num_layers_ = 0;
		failure_ = false;
		return *this;
	}

	Builder &matmul(const int in_features, const int out_features) override {
		AnyLayer *layer = make_layer();
		if (layer) {
			layer->kind = LayerKind::matmul;
			layer->matmul.in_features = in_features;
			layer->matmul.out_features = out_features;
		}
		return *this;
	}

	Builder &bias(const int features) override {
		AnyLayer *layer = make_layer();
		if (layer) {
			layer->kind = LayerKind::bias;
			layer->bias.features = features;
		}
		return *this;
	}

	Builder &relu() override {
		AnyLayer *layer = make_layer();
		if (layer) {
			layer->kind = LayerKind::relu;
		}
		return *this;
	}

	Builder &sigmoid() override {
		AnyLayer *layer = make_layer();
		if (layer) {
			layer->kind = LayerKind::sigmoid;
		}
		return *this;
	}

	Network *build() override {
		if (failure_) {
			return nullptr;
		}
		NetworkImpl *net = new NetworkImpl(layers_, num_layers_, mem_manager_);
		// We “move” ownership of layers_ to the network
		layers_ = nullptr;
		num_layers_ = 0;
		max_layers_ = 0;
		if (!net->allocated()) {
			// One of the buffers (parameter, etc.) failed to allocate memory
			delete net;
			return nullptr;
		}
		return net;
	}

protected:
	[[nodiscard]] AnyLayer *make_layer() {
		if (num_layers_ >= max_layers_) {
			failure_ = true;
			return nullptr;
		}
		num_layers_++;
		return &layers_[num_layers_ - 1];
	}
};

} // namespace

Builder *Builder::make(int max_layers, MemManager *mem_manager) {
	return new BuilderImpl(max_layers, mem_manager);
}

Builder &Builder::linear(int in_features, int out_features) {
	return matmul(in_features, out_features).bias(out_features);
}

bool VecView::empty() const {
	return (rows == 0);
}

bool Sample::empty() const {
	return input.empty() && target.empty();
}

float train(Network &net, Dataset &d, const float learning_rate, const LossKind loss_kind, RNG *rng, TrainingObserver *observer) {
	const int n{ d.len() };

	const int observer_interval{ n / 100 };

	int *indices = static_cast<int *>(malloc(sizeof(int) * n));
	if (!indices) {
		return INFINITY;
	}

	for (int i = 0; i < n; i++) {
		indices[i] = i;
	}

	// Shuffle if RNG is provided
	if (rng) {
		for (int i = 1; i < n; i++) {
			int j = rng->randint(0, i - 1);
			int tmp = indices[i];
			indices[i] = indices[j];
			indices[j] = tmp;
		}
	}

	float total_loss{ 0.0F };

	for (int i = 0; i < n; i++) {
		const Sample s = d.get_item(indices[i]);
		int out_rows{ 0 };

		const float *out_data = net.forward(s.input.data, s.input.rows, &out_rows);

		net.zero_grad();

		// backprop
		const float loss = net.backward(s.target.data, s.target.rows, loss_kind);

		// update parameters
		net.gradient_descent(learning_rate);

		total_loss += loss;

		if (observer && (i % observer_interval == 0)) {
			const auto avg_loss = total_loss / (i + 1);
			observer->on_training_progress(static_cast<float>(i) / static_cast<float>(n), avg_loss);
		}
	}

	free(indices);

	return total_loss / static_cast<float>(n);
}

} // namespace retronet
