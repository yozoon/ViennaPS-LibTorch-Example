#include <iostream>

#include <psCSVWriter.hpp>
#include <psGeometricModel.hpp>
#include <psSmartPointer.hpp>

#include "TorchEstimator.hpp"

int main(int argc, const char *argv[]) {
  using NumericType = float;

  static constexpr int InputDim = 3;
  static constexpr int TargetDim = 1;

  static constexpr int DataDim = InputDim + TargetDim;

  const int numSamples = 40;

  TorchEstimator<NumericType> estimator;
  estimator.setModelFilename("pretrained_model.pt");
  if (!estimator.initialize())
    return -1;

  psCSVWriter<NumericType, DataDim> writer(
      "output.csv", "x, y, z, data\nValues estimated by torch model "
                    "on rectilinear grid");
  writer.initialize();

  for (int i = 0; i < numSamples; ++i)
    for (int j = 0; j < numSamples; ++j)
      for (int k = 0; k < numSamples; ++k) {
        std::array<NumericType, InputDim> x;

        // // Interpolate
        // x[0] = .3 + i * (.6 - .3) / (numSamples - 1);
        // x[1] = -4. + j * (6. + 4.) / (numSamples - 1);
        // x[2] = -4. + k * (-1. + 4.) / (numSamples - 1);
        // Extrapolate
        x[0] = .1 + i * (.8 - .1) / (numSamples - 1);
        x[1] = -6. + j * (8. + 6.) / (numSamples - 1);
        x[2] = -6. + k * (1. + 6.) / (numSamples - 1);

        // We use structural bindings to directly unpack the tuple of one
        // element
        auto [value] = estimator.estimate(x);

        writer.writeRow({x[0], x[1], x[2], value[0]});
      }
}
