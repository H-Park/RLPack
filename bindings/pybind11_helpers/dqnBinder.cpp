#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "../../src/AgentImpl.cuh"
#include "../../src/DQN/Agent.cuh"

class GetDQNAgent {
  std::string modelName_;
  pybind11::kwargs modelArgs_;
  pybind11::kwargs agentArgs_;
  AgentImpl *agent_;

  static torch::Tensor pybind_array_to_torch_tensor(pybind11::array &array, pybind11::tuple &shape);

  std::vector<dqn::Dqn1D*> get_dqn1d_model();

 public:
  GetDQNAgent(pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &agentArgs);
  ~GetDQNAgent();

  AgentImpl *get_agent();
  int train(pybind11::array &stateCurrent,
            pybind11::array &stateNext,
            pybind11::float_ &reward,
            pybind11::int_ &action,
            pybind11::bool_ &done,
            pybind11::tuple &stateCurrentShape,
            pybind11::tuple &stateNextShape);
  int policy(pybind11::array &stateCurrent, pybind11::tuple &stateCurrentShape);
};

GetDQNAgent::GetDQNAgent(pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &agentArgs) {
  modelName_ = modelName.cast<std::string>();
  modelArgs_ = modelArgs;
  agentArgs_ = agentArgs;

  agent_ = get_agent();
}

std::vector<dqn::Dqn1D*> GetDQNAgent::get_dqn1d_model() {
  auto sequenceLength = modelArgs_["sequence_length"].cast<int64_t>();
  auto channels = modelArgs_["channels"].cast<std::vector<int64_t>>();
  auto kernelSizes = modelArgs_["kernel_sizes"].cast<std::vector<int64_t>>();
  auto strideSizes = modelArgs_["strides_sizes"].cast<std::vector<int64_t>>();
  auto dilationSizes = modelArgs_["dilation_sizes"].cast<std::vector<int64_t>>();
  auto activation = modelArgs_["activation"].cast<std::string>();
  auto dropout = modelArgs_["dropout"].cast<float_t>();
  auto numActions = modelArgs_["num_actions"].cast<int64_t>();
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

  targetModel->to_double();
  policyModel->to_double();

  std::vector<dqn::Dqn1D*> dqn1dModels = {targetModel, policyModel};

  return dqn1dModels;
}

AgentImpl *GetDQNAgent::get_agent() {
  auto gamma = agentArgs_["gamma"].cast<float>();
  auto epsilon = agentArgs_["epsilon"].cast<float>();
  auto epsilonDecayRate = agentArgs_["epsilon_decay_rate"].cast<float>();
  auto memoryBufferSize = agentArgs_["memory_buffer_size"].cast<int>();
  auto targetModelUpdateRate = agentArgs_["target_model_update_rate"].cast<int>();
  auto policyModelUpdateRate = agentArgs_["policy_model_update_rate"].cast<int>();
  auto numActions = agentArgs_["num_actions"].cast<int>();
  auto savePath = agentArgs_["save_path"].cast<std::string>();

  if (modelName_ == "dqn1d") {
    std::vector<dqn::Dqn1D*> models = get_dqn1d_model();

    auto optimizerName = agentArgs_["optimizer"].cast<std::string>();

    if (optimizerName == "adam") {

      auto lr = agentArgs_["lr"].cast<pybind11::float_>().cast<float>();
      torch::optim::AdamOptions adamOptions = torch::optim::AdamOptions(lr);
      auto *adamOptimizer = new torch::optim::Adam(models.at(1)->parameters(), adamOptions);
      auto *agent = new dqn::Agent<dqn::Dqn1D*, torch::optim::Adam*>(models.at(0),
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
    } else {
      throw std::runtime_error("Invalid Optimizer name passed!");
    }
  } else {
    throw std::runtime_error("Invalid Model name passed!");
  }
}

int GetDQNAgent::train(pybind11::array &stateCurrent,
                    pybind11::array &stateNext,
                    pybind11::float_ &reward,
                    pybind11::int_ &action,
                    pybind11::bool_ &done,
                    pybind11::tuple &stateCurrentShape,
                    pybind11::tuple &stateNextShape) {

  // Convert pybind::array (numpy arrays) to torch::Tensor.
  auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape);
  auto stateNextTensor_ = pybind_array_to_torch_tensor(stateNext, stateNextShape);

  // Convert other arguments to native C++ types.
  auto reward_ = reward.cast<float>();
  auto action_ = action.cast<int>();
  auto done_ = done.cast<int>();

  action_ = agent_->train(stateCurrentTensor_, stateNextTensor_, reward_, action_, done_);
  return action_;
}

int GetDQNAgent::policy(pybind11::array &stateCurrent, pybind11::tuple &stateCurrentShape) {
  auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape);
  auto action = agent_->policy(stateCurrentTensor_);
  return action;
}

torch::Tensor GetDQNAgent::pybind_array_to_torch_tensor(pybind11::array &array, pybind11::tuple &shape) {
  std::vector<int64_t> shapeVector;
  for (auto &shape_ : shape) {
    shapeVector.push_back(shape_.cast<int>());
  }

  auto tensorOptions = torch::TensorOptions().dtype(torch::kDouble);
  auto tensor = torch::zeros(shapeVector, tensorOptions);
  memmove(tensor.data_ptr(), array.data(), tensor.nbytes());

  return tensor;
}

GetDQNAgent::~GetDQNAgent() = default;

// ---------------------------------------- Declaration of Binding Module ---------------------------------------- //

PYBIND11_MODULE(RLPack, m) {
  m.doc() = "RLPack plugin to bind the GetAgent class to Python interface";
  pybind11::class_<GetDQNAgent>(m, "GetDQNAgent")
    .def(pybind11::init<pybind11::str &, pybind11::dict &, pybind11::dict &>())
    .def("train", &GetDQNAgent::train, "train method to train the agent")
    .def("policy", &GetDQNAgent::policy, "policy method to run the policy (only eval) of the agent")
    .def("__repr__", [](const GetDQNAgent &) { return "<GetDQNAgent>"; });
}
