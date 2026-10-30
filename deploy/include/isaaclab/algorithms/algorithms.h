// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>
#include <stdexcept>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ORT_SEQUENTIAL);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            if (input_type.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Deployment requires float32 ONNX inputs");
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.get());
            name_storage.push_back(std::move(input_name));
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                if (dim<=0) throw std::runtime_error("Deployment requires fixed positive ONNX input dimensions");
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        if (output_type.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Deployment requires float32 ONNX output");
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size()!=2 || output_shape[0]!=1 || output_shape[1]<=0)
            throw std::runtime_error("Deployment requires ONNX output shape [1, actions]");
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.get());
        name_storage.push_back(std::move(output_name));

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            if (input_data.size() != input_sizes[i]) {
                throw std::runtime_error("Observation size does not match ONNX input " + name_str);
            }
            for (float value : input_data) {
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite ONNX observation");
            }
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        for (int64_t i = 0; i < output_shape[1]; ++i) {
            if (!std::isfinite(floatarr[i])) throw std::runtime_error("Non-finite ONNX action");
        }
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

    void validate_deployment(size_t observation_dim, size_t action_dim,
                             const std::map<std::string, std::string>& expected_metadata)
    {
        if (input_shapes.size() != 1 || input_shapes[0] != std::vector<int64_t>{1, static_cast<int64_t>(observation_dim)} ||
            output_shape != std::vector<int64_t>{1, static_cast<int64_t>(action_dim)} ||
            std::string(input_names[0]) != "obs") {
            throw std::runtime_error("Deployment YAML and ONNX tensor contract differ");
        }
        auto metadata = session->GetModelMetadata();
        for (const auto& item : expected_metadata) {
            auto value = metadata.LookupCustomMetadataMapAllocated(item.first.c_str(), allocator);
            if (!value || item.second != value.get()) {
                throw std::runtime_error("ONNX metadata mismatch: " + item.first);
            }
        }
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> name_storage;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};
