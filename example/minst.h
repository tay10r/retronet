#include <retronet.h>

#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>

class MINSTDataset final : public retronet::Dataset {
	std::vector<float> images_;

	std::vector<float> labels_;

public:
	bool load(const char *image_path, const char *sample_path) {
		images_ = normalize(load(image_path, 0x803));
		labels_ = one_hot(load(sample_path, 0x801), /* number of MINST classes -> */ 10);
		return true;
	}

	retronet::Sample get_item(const int index) const override {
		retronet::Sample s;
		s.input.data = &images_.at(index * (28 * 28));
		s.input.rows = 28 * 28;
		s.target.data = &labels_.at(index * 10);
		s.target.rows = 10;
		return s;
	}

	int len() const override {
		return labels_.size() / 10;
	}

protected:
	static uint32_t decode_u32(const uint8_t *buf) {
		uint32_t x = 0;
		x |= static_cast<uint32_t>(buf[0]) << 24;
		x |= static_cast<uint32_t>(buf[1]) << 16;
		x |= static_cast<uint32_t>(buf[2]) << 8;
		x |= static_cast<uint32_t>(buf[3]);
		return x;
	}

	static std::vector<float> normalize(const std::vector<uint8_t> &data) {
		std::vector<float> out(data.size());
		for (size_t i = 0; i < data.size(); i++) {
			out[i] = static_cast<float>(data[i]) / 255.0F;
		}
		return out;
	}

	static std::vector<float> one_hot(const std::vector<uint8_t> &data, const int dims) {
		std::vector<float> out(data.size() * dims);
		for (size_t i = 0; i < data.size(); i++) {
			for (int j = 0; j < dims; j++) {
				out.at(i * dims + j) = (j == static_cast<int>(data[i])) ? 1.0F : 0.0F;
			}
		}
		return out;
	}

	static std::vector<uint8_t> load(const char *filename, const uint32_t expected_magic) {
		std::ifstream file(filename, std::ios::binary | std::ios::in);
		if (!file.good()) {
			std::cerr << "failed to open " << filename << std::endl;
			return {};
		}
		unsigned char header[4];
		file.read(reinterpret_cast<char *>(header), sizeof(header));
		if (decode_u32(header) != expected_magic) {
			std::cerr << "unexpected magic number in " << filename << std::endl;
			return {};
		}
		const auto dims = header[3];
		std::vector<unsigned char> shape_buffer;
		shape_buffer.resize(dims * 4);
		file.read(reinterpret_cast<char *>(shape_buffer.data()), shape_buffer.size());

		std::vector<int> shape;
		shape.resize(dims);
		for (int i = 0; i < dims; i++) {
			const auto len = decode_u32(&shape_buffer[i * 4]);
			shape[i] = len;
		}

		int total = 1;
		for (const auto s : shape) {
			total *= s;
		}
		std::vector<uint8_t> data;
		data.resize(total);
		file.read(reinterpret_cast<char *>(data.data()), data.size());
		if (file.gcount() != total) {
			std::cerr << "failed to read all the data" << std::endl;
		}
		return data;
	}
};

