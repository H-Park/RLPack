//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#include "Dqn1D.h"

namespace dqn {

Dqn1D::Dqn1D(
  int64_t sequenceLength,
  std::vector<int64_t> &channels,
  std::vector<int64_t> &kernelSizes,
  std::vector<int64_t> &strideSizes,
  std::vector<int64_t> &dilationSizes,
  std::string &activation,
  float_t dropout,
  int64_t numActions){

  setupModel(sequenceLength,
             channels,
             kernelSizes,
             strideSizes,
             dilationSizes,
             dropout,
             numActions);

}

Dqn1D::~Dqn1D() = default;

std::vector<int64_t> Dqn1D::get_interims(int64_t seqLength,
                                         std::vector<int64_t> &kernelSizes,
                                         std::vector<int64_t> &stridesSizes,
                                         std::vector<int64_t> &dilationSizes) {
  size_t numBlocks = kernelSizes.size();
  std::vector<int64_t> interimSizes;
  interimSizes.push_back(seqLength);

  for (int64_t idx = 0; idx != numBlocks; idx++) {

    int64_t interimSizeConvLength = floor(
      (((interimSizes[idx] - dilationSizes[idx] * (kernelSizes[idx] - 1) - 1) / stridesSizes[idx])) + 1);

    interimSizes.push_back(interimSizeConvLength);
  }

  return interimSizes;
}
void Dqn1D::setupModel(int64_t &imageDims,
                       std::vector<int64_t> &channels,
                       std::vector<int64_t> &kernelSizes,
                       std::vector<int64_t> &stridesSizes,
                       std::vector<int64_t> &dilationSizes,
                       float_t dropout,
                       int64_t numClasses) {

  size_t numBlocks = channels.size() - 1;

  std::string convModuleName = "conv_";
  std::string maxPoolModuleName = "pool_";

  for (int64_t idx = 0; idx != numBlocks; idx++) {
    torch::nn::Conv2dOptions conv2dOptions = torch::nn::Conv2dOptions(channels[idx],
                                                                      channels[idx + 1],
                                                                      kernelSizes[idx])
                                               .stride(stridesSizes[idx])
                                               .dilation(dilationSizes[idx]);
    auto *convBlock = new torch::nn::Conv2d(conv2dOptions);

    convSubmodules->push_back(*convBlock);

    register_module(convModuleName.append((std::to_string(idx))), *convBlock);
  }
  std::vector<int64_t> interimSizes = get_interims(imageDims,
                                                   kernelSizes,
                                                   stridesSizes,
                                                   dilationSizes);
  int64_t finalSizes = interimSizes.back();

  auto dropoutOptions = torch::nn::DropoutOptions(dropout);
  dropoutSubmodule = new torch::nn::Dropout(dropoutOptions);
  register_module("dropout", *dropoutSubmodule);

  auto flattenOptions = torch::nn::FlattenOptions().start_dim(1).end_dim(-1);
  flattenSubmodule = new torch::nn::Flatten(flattenOptions);
  register_module("flatten", *flattenSubmodule);

  auto linearOptions = torch::nn::LinearOptions(finalSizes * channels.back(), numClasses);
  linearSubmodule = new torch::nn::Linear(linearOptions);
  register_module("linear", *linearSubmodule);
}

torch::Tensor Dqn1D::forward(torch::Tensor x) {
  for (int64_t idx = 0; idx != convSubmodules->size(); idx++) {
    x = convSubmodules[idx]->as<torch::nn::Conv2d>()->forward(x);
    x = relu(x);
  }
  x = flattenSubmodule->ptr()->forward(x);
  x = dropoutSubmodule->ptr()->forward(x);
  x = linearSubmodule->ptr()->forward(x);
  return x;
}
}// namespace dqn
