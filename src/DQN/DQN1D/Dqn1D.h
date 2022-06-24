//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_DQN1D_DQN1D_H_
#define RLPACK_DQN_DQN1D_DQN1D_H_

#include <torch/torch.h>

namespace dqn {
class Dqn1D : public torch::nn::Module {
  torch::nn::ModuleList convSubmodules = torch::nn::ModuleList();
  torch::nn::ReLU relu;
  torch::nn::Flatten *flattenSubmodule = nullptr;
  torch::nn::Dropout *dropoutSubmodule = nullptr;
  torch::nn::Linear *linearSubmodule = nullptr;

  bool usePadding_;

  static std::vector<int64_t> get_interims(int64_t imageDims,
                                    std::vector<int64_t> &kernelSizes,
                                    std::vector<int64_t> &stridesSizes,
                                    std::vector<int64_t> &dilationSizes);

  static std::vector<int64_t> compute_padding(int64_t imageDims,
                                       std::vector<int64_t> &kernelSizes,
                                       std::vector<int64_t> &stridesSizes,
                                       std::vector<int64_t> &dilationSizes);

  void setupModel(int64_t &imageDims,
                  std::vector<int64_t> &channels,
                  std::vector<int64_t> &kernelSizes,
                  std::vector<int64_t> &stridesSizes,
                  std::vector<int64_t> &dilationSizes,
                  float_t dropout,
                  int64_t numClasses);

 public:
  Dqn1D(
    int64_t sequenceLength,
    std::vector<int64_t> &channels,
    std::vector<int64_t> &kernelSizes,
    std::vector<int64_t> &strideSizes,
    std::vector<int64_t> &dilationSizes,
    std::string &activation,
    float_t dropout,
    int64_t numActions,
    bool usePadding = true);

  ~Dqn1D() override;

  torch::Tensor forward(torch::Tensor x);
  void to_double();
};}// namespace dqn

#endif//RLPACK_DQN_DQN1D_DQN1D_H_
