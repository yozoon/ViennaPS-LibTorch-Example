#ifndef TORCH_ESTIMATOR_HPP
#define TORCH_ESTIMATOR_HPP

#include <array>
#include <tuple>
#include <vector>

#include <psSmartPointer.hpp>
#include <psValueEstimator.hpp>

#include <torch/cuda.h>
#include <torch/script.h>

// Class providing nearest neighbors interpolation
template <typename NumericType>
class TorchEstimator : public psValueEstimator<NumericType, 3, 1> {

  static constexpr int InputDim = 3;
  static constexpr int OutputDim = 1;

  using Parent = psValueEstimator<NumericType, InputDim, OutputDim>;

  using typename Parent::InputType;
  using typename Parent::OutputType;

  using Parent::DataDim;
  using Parent::dataSource;

  using DataPtr = typename decltype(dataSource)::element_type::DataPtr;
  using DataVector = std::vector<std::array<NumericType, DataDim>>;

  std::string modelFilename;

  torch::jit::script::Module model;

  DataPtr data = nullptr;
  bool initialized = false;

public:
  TorchEstimator() {}
  TorchEstimator(std::string passedModelFilename)
      : modelFilename(passedModelFilename) {}

  void setModelFilename(std::string passedModelFilename) {
    modelFilename = passedModelFilename;
  }

  bool initialize() override {
    try {
      c10::InferenceMode guard;
      model = torch::jit::load(modelFilename);
    } catch (const c10::Error &e) {
      std::cerr << "Error while loading the Torch model\n";
      return false;
    }

    initialized = true;
    return true;
  }

  std::tuple<OutputType> estimate(const InputType &input) override {
    if (!initialized)
      if (!initialize())
        return {};

    std::array<NumericType, InputDim> x;
    std::copy(input.begin(), input.end(), x.begin());

    // We only want to use the pretrained model in inference mode
    c10::InferenceMode guard;

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);

    std::vector<torch::jit::IValue> inputs = {
        torch::from_blob(x.data(), {static_cast<long>(x.size())},
                         torch::TensorOptions().dtype(torch::kFloat))};

    // Execute the model and turn its output into a tensor.
    auto out = model.forward(inputs).toTensor();

    // std::cout << out.slice() << std::endl;

    OutputType result;
    // std::copy(out.data_ptr<float>(), out.data_ptr<float>() + out.numel(),
    //           result.begin());
    result[0] = out[0].item<NumericType>();
    return {result};
  }
};

#endif