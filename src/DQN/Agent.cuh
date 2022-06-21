//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_AGENT_CUH_
#define RLPACK_DQN_AGENT_CUH_

#include <random>
#include <torch/torch.h>

#include "../AgentImpl.cuh"
#include "DQN1D/Dqn1D.h"

namespace dqn {
template<class ModelClass, class Optimizer>
class Agent : public AgentImpl {

  ModelClass *targetModel_;
  ModelClass *policyModel_;
  Optimizer *optimizer_;
  float gamma_;
  float epsilon_;
  float epsilonDecayRate_;
  int memoryBufferSize_;
  int targetModelUpdateRate_;
  int policyModelUpdateRate_;
  int numActions_;
  std::string savePath_;

  torch::nn::HuberLoss huberLoss_;

  struct Memory {
   private:
    std::vector<torch::Tensor> stateCurrent_;
    std::vector<torch::Tensor> stateNext_;
    std::vector<torch::Tensor> reward_;
    std::vector<torch::Tensor> action_;
    std::vector<torch::Tensor> done_;

   public:
    Memory();
    ~Memory();
    void push_back(torch::Tensor &stateCurrent,
                   torch::Tensor &stateNext,
                   int64_t reward,
                   int action,
                   int done);
    void push_back(Memory *memory);
    torch::Tensor stack_current_states();
    torch::Tensor stack_next_states();
    torch::Tensor stack_rewards();
    torch::Tensor stack_actions();
    torch::Tensor stack_dones();
    void clear();
    size_t size();
    Memory *at(int index);
  } memoryBuffer;

  int targetModelUpdateCounter = 0;
  int policyModelUpdateCounter = 0;

  void train_policy_model();
  void update_target_model();
  typename Agent<ModelClass, Optimizer>::Memory *load_random_experiences();
  torch::Tensor temporal_difference(torch::Tensor &rewards, torch::Tensor &qValues, torch::Tensor &dones);
  void decay_epsilon();
  void clear_memory();

 public:
  Agent(ModelClass &targetModel,
        ModelClass &policyModel,
        Optimizer &optimizer,
        float gamma,
        float epsilon,
        float epsilonDecayRate,
        int memoryBufferSize,
        int targetModelUpdateRate,
        int policyModelUpdateRate,
        int numActions,
        std::string &savePath);
  ~Agent();

  int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, int64_t reward, int action, int done);
  int policy(torch::Tensor &stateCurrent);
};

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Memory::Memory() = default;

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::push_back(torch::Tensor &stateCurrent,
                                                     torch::Tensor &stateNext,
                                                     int64_t reward,
                                                     int action,
                                                     int done) {
  stateCurrent_.push_back(stateCurrent);
  stateNext_.push_back(stateNext);

  torch::TensorOptions tensorOptions = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor rewardAsTensor = torch::full({1, 1}, reward, tensorOptions);
  reward_.push_back(rewardAsTensor);

  torch::Tensor actionAsTensor = torch::full({1, 1}, action, tensorOptions);
  action_.push_back(actionAsTensor);

  torch::Tensor doneAsTensor = torch::full({1, 1}, done, tensorOptions);
  done_.push_back(doneAsTensor);
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::clear() {
  stateCurrent_.clear();
  stateNext_.clear();
  reward_.clear();
  action_.clear();
  done_.clear();
}

template<class ModelClass, class Optimizer>
typename Agent<ModelClass, Optimizer>::Memory *Agent<ModelClass, Optimizer>::Memory::at(int index) {
  auto *memory = new Memory();
  memory->stateCurrent_ = {stateCurrent_[index]};
  memory->stateNext_ = {stateNext_[index]};
  memory->reward_ = {reward_[index]};
  memory->action_ = {action_[index]};
  memory->done_ = {done_[index]};

  return memory;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::push_back(typename Agent<ModelClass, Optimizer>::Memory *memory) {
  stateCurrent_.push_back(memory->stateCurrent_[0]);
  stateNext_.push_back(memory->stateNext_[0]);
  reward_.push_back(memory->reward_[0]);
  action_.push_back(memory->action_[0]);
  done_.push_back(memory->done_[0]);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_current_states() {
  return torch::stack(stateCurrent_);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_next_states() {
  return torch::stack(stateNext_);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_rewards() {
  return torch::stack(reward_);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_actions() {
  return torch::stack(action_);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_dones() {
  return torch::stack(done_);
}

template<class ModelClass, class Optimizer>
size_t Agent<ModelClass, Optimizer>::Memory::size() {
  return done_.size();
}

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Agent(ModelClass &targetModel,
                                    ModelClass &policyModel,
                                    Optimizer &optimizer,
                                    float gamma, float epsilon,
                                    float epsilonDecayRate,
                                    int memoryBufferSize,
                                    int targetModelUpdateRate,
                                    int policyModelUpdateRate,
                                    int numActions,
                                    std::string &savePath) {
  targetModel_ = &targetModel;
  policyModel_ = &policyModel;
  optimizer_ = &optimizer;

  gamma_ = gamma;
  epsilon_ = epsilon;
  epsilonDecayRate_ = epsilonDecayRate;
  memoryBufferSize_ = memoryBufferSize;
  assert(targetModelUpdateRate > policyModelUpdateRate);

  targetModelUpdateRate_ = targetModelUpdateRate;
  policyModelUpdateRate_ = policyModelUpdateRate;
  numActions_ = numActions;

  savePath_ = savePath;
}

template<class ModelClass, class Optimizer>
int Agent<ModelClass, Optimizer>::train(torch::Tensor &stateCurrent,
                                        torch::Tensor &stateNext,
                                        int64_t reward,
                                        int action,
                                        int done) {

  memoryBuffer.push_back(stateCurrent, stateNext, reward, action, done);
  policyModelUpdateCounter += 1;
  targetModelUpdateCounter += 1;

  if (policyModelUpdateCounter == policyModelUpdateRate_ + 1) {
    train_policy_model();
    policyModelUpdateCounter = 0;
  }

  if (targetModelUpdateCounter == targetModelUpdateRate_) {
    update_target_model();
    targetModelUpdateCounter = 0;
  }

  if (memoryBuffer.size() == memoryBufferSize_) {
    clear_memory();
  }

  if (done == 1) {
    decay_epsilon();
  }

  action = policy(stateCurrent);
  return action;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::train_policy_model() {
  Memory *randomExperiences = load_random_experiences();

  torch::Tensor statesCurrent = randomExperiences->stack_current_states();
  torch::Tensor statesNext = randomExperiences->stack_next_states();
  torch::Tensor rewards = randomExperiences->stack_rewards();
  torch::Tensor actions = randomExperiences->stack_actions();
  torch::Tensor dones = randomExperiences->stack_dones();

  policyModel_->train();

  torch::Tensor tdValue;
  {
    targetModel_->eval();
    torch::NoGradGuard guard;
    torch::Tensor qValuesTarget = targetModel_->forward(statesNext);
    tdValue = temporal_difference(rewards, qValuesTarget, dones);
  }

  torch::Tensor qValuesPolicy = policyModel_->forward(statesCurrent);
  torch::Tensor qValuesPolicyGathered = qValuesPolicy.gather(-1, actions);

  optimizer_->zero_grad();
  torch::Tensor loss = huberLoss_(tdValue, qValuesPolicyGathered);
  loss.backward();
  optimizer_->step();
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::update_target_model() {
  torch::serialize::OutputArchive outputArchive;
  torch::serialize::InputArchive inputArchive;

  policyModel_->save(outputArchive);
  outputArchive.save_to(savePath_);

  inputArchive.load_from(savePath_);
  targetModel_->load(inputArchive);
}

template<class ModelClass, class Optimizer>
typename Agent<ModelClass, Optimizer>::Memory *Agent<ModelClass, Optimizer>::load_random_experiences() {
  std::vector<int> loadedIndices;
  std::vector<torch::Tensor> memoryBufferTensors;
  std::random_device rd;
  std::mt19937 generator(rd());
  int index;

  auto *loadedExperiences = new Memory();

  while (memoryBufferTensors.size() != policyModelUpdateRate_) {
    std::uniform_int_distribution<int> distribution(0, memoryBuffer.size());
    index = distribution(generator);

    Memory *memory = memoryBuffer.at(index);
    loadedExperiences->push_back(memory);
  }

  return loadedExperiences;
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::temporal_difference(torch::Tensor &rewards,
                                                                torch::Tensor &qValues,
                                                                torch::Tensor &dones) {
  torch::Tensor tdValue = rewards + ((gamma_ * qValues.max()) * (1 - dones));
  return tdValue;
}

template<class ModelClass, class Optimizer>
int Agent<ModelClass, Optimizer>::policy(torch::Tensor &stateCurrent) {
  int action;
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distributionP(0, 1);
  std::uniform_real_distribution<int> distributionAction(0, numActions_);
  float p = distributionP(generator);

  if (p < epsilon_) {
    action = distributionAction(generator);
  } else {
    {
      policyModel_->eval();
      torch::NoGradGuard guard;
      torch::Tensor qValues = policyModel_->forward(stateCurrent);
      torch::Tensor actionTensor = qValues.argmax(-1);
      action = actionTensor.item<int>();
    }
  }

  return action;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::decay_epsilon() {
  epsilon_ *= epsilonDecayRate_;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::clear_memory() {
  memoryBuffer.clear();
}

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Memory::~Memory() = default;

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::~Agent() = default;

}// namespace dqn
#endif//RLPACK_DQN_AGENT_CUH_
