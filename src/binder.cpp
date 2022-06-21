#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "DQN/Agent.cuh"
#include "AgentImpl.cuh"

class GetAgent {
  std::string modelName_;
  pybind11::kwargs modelArgs_;
  pybind11::kwargs agentArgs_;

  std::vector<dqn::Dqn1D> get_dqn1d_model();

 public:
  GetAgent(std::string &modelName, pybind11::dict &modelArgs, pybind11::dict &agentArgs);
  ~GetAgent();

  AgentImpl *getAgent();
};

GetAgent::GetAgent(std::string &modelName, pybind11::dict &modelArgs, pybind11::dict &agentArgs) {
  modelName_ = modelName;
  modelArgs_ = modelArgs;
  agentArgs_ = agentArgs;
}

std::vector<dqn::Dqn1D> GetAgent::get_dqn1d_model() {
  auto sequenceLength = modelArgs_("sequence_length").cast<int64_t>();
  auto channels = modelArgs_("channels").cast<std::vector<int64_t>>();
  auto kernelSizes = modelArgs_("kernel_sizes").cast<std::vector<int64_t>>();
  auto strideSizes = modelArgs_("strides_sizes").cast<std::vector<int64_t>>();
  auto dilationSizes = modelArgs_("dilation_sizes").cast<std::vector<int64_t>>();
  auto activation = modelArgs_("activation").cast<std::string>();
  auto dropout = modelArgs_("dropout").cast<float_t>();
  auto numActions = modelArgs_("num_actions").cast<int64_t>();

  auto *targetModel = new dqn::Dqn1D(sequenceLength,
                                     channels,
                                     kernelSizes,
                                     strideSizes,
                                     dilationSizes,
                                     activation,
                                     dropout,
                                     numActions);
  auto *policyModel = new dqn::Dqn1D(sequenceLength,
                                     channels,
                                     kernelSizes,
                                     strideSizes,
                                     dilationSizes,
                                     activation,
                                     dropout,
                                     numActions);
  std::vector<dqn::Dqn1D> dqn1dModels = {targetModel, policyModel};
  return dqn1dModels;
}

AgentImpl *GetAgent::getAgent() {
  auto gamma = agentArgs_("gamma").cast<float>();
  auto epsilon = agentArgs_("epsilon").cast<float>();
  auto epsilonDecayRate = agentArgs_("epsilon_decay_rate").cast<float>();
  auto memoryBufferSize = agentArgs_("memory_buffer_size").cast<int>();
  auto targetModelUpdateRate = agentArgs_("target_model_update_rate").cast<int>();
  auto policyModelUpdateRate = agentArgs_("policy_model_update_rate").cast<int>();
  auto numActions = agentArgs_("num_actions").cast<int>();
  auto savePath = agentArgs_("save_path").cast<std::string>();

  if (modelName_ == "dqn1d") {
    std::vector<dqn::Dqn1D> models = get_dqn1d_model();

    auto optimizerName = agentArgs_("optimizer").cast<std::string>();
    if (optimizerName == "adam") {

      auto lr = agentArgs_("lr").cast<float>();
      torch::optim::AdamOptions adamOptions = torch::optim::AdamOptions().lr(lr);
      torch::optim::Adam adamOptimizer(models.at(1).parameters(),adamOptions);

      auto *agent = new dqn::Agent<dqn::Dqn1D, torch::optim::Adam>(models.at(0),
                                                                  models.at(1),
                                                                  adamOptimizer,
                                                                  gamma,
                                                                  epsilon,
                                                                  epsilonDecayRate,
                                                                  memoryBufferSize,
                                                                  targetModelUpdateRate,
                                                                  policyModelUpdateRate,
                                                                  numActions,
                                                                  savePath);
      return agent;
    }
  } else {
    throw std::runtime_error("Invalid Model name passed!");
  }
}

GetAgent::~GetAgent() = default;
