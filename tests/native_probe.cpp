#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <core/session/onnxruntime_cxx_api.h>
#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> infer(const std::string& path, py::array_t<float, py::array::c_style> histories) {
    if (histories.ndim() != 2 || histories.shape(0) < 1 || histories.shape(1) != 2528) {
        throw std::invalid_argument("expected [batch, 2528] float32 histories");
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "parity");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    Ort::Session session(env, path.c_str(), options);
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t count = histories.shape(0);
    const std::array<int64_t, 2> in_shape{count, 2528}, out_shape{count, 19};
    py::array_t<float> actions(out_shape);
    // ORT's external-buffer API takes a mutable pointer even for a read-only model input.
    auto input = Ort::Value::CreateTensor<float>(cpu, const_cast<float*>(histories.data()),
                                                 histories.size(), in_shape.data(), 2);
    auto output = Ort::Value::CreateTensor<float>(cpu, actions.mutable_data(), actions.size(),
                                                  out_shape.data(), 2);
    const char* input_name = "history";
    const char* output_name = "action";
    session.Run(Ort::RunOptions{}, &input_name, &input, 1, &output_name, &output, 1);
    return actions;
}

PYBIND11_MODULE(native_probe, m) {
    if (mj_version() != mjVERSION_HEADER) {
        throw std::runtime_error("MuJoCo headers and runtime differ");
    }
    m.def("infer", &infer);
}
