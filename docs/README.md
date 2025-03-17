RetroNet
========

This is a C++ library, with no dependencies (except for libc), for designing and training multi-layer perceptrons.
This library is meant for uses cases where:
 - you need a neural network in an environment with constraints (game engines, embedded systems, etc)
 - you want to avoid all the dependencies that usually come with neural network libraries

This library is not a replacement for libraries such as Torch or Tensorflow.
It is an extremely lightweight alternative for use cases where you do not need as many bells and whistles.

### Example

Here is a minimal an example to demonstrate part of the library.
Note that it does not include any code related to training the network (but the library does support training).

```cxx
#include "retronet.h"
int main()
{
    // create the builder, which instantiates a network
    retronet::Builder *builder = retronet::Builder::make();
    // the network design
    builder->reset()
      .linear(28 * 28, 256)
      .relu(),
      .linear(256, 128)
      .relu()
      .linear(128, 10);
    // finalize the network
    retronet::Network *net = builder->build();
    // run the network on a dummy input
    float input[28 * 28];
    int output_rows{};
    float* output = net->forward(input, 28 * 28, &output_rows);
    // cleanup
    delete builder;
    delete net;
    return 0;
}
```

For a more complete example, check out the [Fashion MNIST example](../example/main.cpp).