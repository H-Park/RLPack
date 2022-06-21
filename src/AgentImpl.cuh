//
// Created by Kartik Rajeshwaran on 2022-06-20.
//

#ifndef RLPACK_SRC_AGENTIMPL_CUH_
#define RLPACK_SRC_AGENTIMPL_CUH_

#include <torch/torch.h>

class AgentImpl{

 public:
  AgentImpl();
  virtual int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, int64_t reward, int action, int done);
  virtual int policy(torch::Tensor &stateCurrent);
};

int AgentImpl::train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, int64_t reward, int action, int done){
  return 0;
}

int AgentImpl::policy(torch::Tensor &stateCurrent) {
  return 0;
}

AgentImpl::AgentImpl() = default;

#endif//RLPACK_SRC_AGENTIMPL_CUH_
