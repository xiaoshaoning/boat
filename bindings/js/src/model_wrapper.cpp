#include <napi.h>
#include "boat_bridge.h"
#include "boat/tensor.h"
#include "boat/model.h"
#include "shared_refs.h"
#include <cstring>
#include <vector>
#include <string>

class ModelWrapper : public Napi::ObjectWrap<ModelWrapper> {
    boat_model_t* model_;

public:
    ModelWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<ModelWrapper>(info), model_(nullptr) {
        model_ = boat_model_create();
        if (!model_) {
            Napi::Error::New(info.Env(), "Failed to create model").ThrowAsJavaScriptException();
        }
    }

    ~ModelWrapper() {
        if (model_) boat_model_free(model_);
    }

    Napi::Value Forward(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "forward expects a Tensor argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto* input = BoatTensorFromJS(info[0]);
        if (!input) {
            Napi::TypeError::New(env, "Argument must be a Tensor").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        boat_tensor_t* output = boat_model_forward(model_, input);
        boat_tensor_unref(input);

        if (!output) {
            Napi::Error::New(env, "Model forward failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        return TensorFromBoat(env, output);
    }

    Napi::Value Backward(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "backward expects a Tensor argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto* grad = BoatTensorFromJS(info[0]);
        if (!grad) {
            Napi::TypeError::New(env, "Argument must be a Tensor").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        boat_tensor_t* output = boat_model_backward(model_, grad);
        boat_tensor_unref(grad);

        if (!output) {
            Napi::Error::New(env, "Model backward failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        return TensorFromBoat(env, output);
    }

    Napi::Value Update(const Napi::CallbackInfo& info) {
        float lr = (info.Length() > 0) ? (float)info[0].ToNumber().DoubleValue() : 0.001f;
        boat_model_update(model_, lr);
        return info.Env().Undefined();
    }

    Napi::Value AddLayer(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Error::New(env, "addLayer not yet implemented in JS bindings").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Value Load(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "load expects a file path string").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        std::string path = info[0].As<Napi::String>().Utf8Value();

        boat_model_t* loaded = boat_model_load(path.c_str());
        if (!loaded) {
            Napi::Error::New(env, "Failed to load model").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        if (model_) boat_model_free(model_);
        model_ = loaded;
        return env.Undefined();
    }

    Napi::Value Save(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "save expects a file path string").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        std::string path = info[0].As<Napi::String>().Utf8Value();

        bool ok = boat_model_save(model_, path.c_str());
        if (!ok) {
            Napi::Error::New(env, "Failed to save model").ThrowAsJavaScriptException();
        }
        return env.Undefined();
    }

    static void Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Model", {
            InstanceMethod("forward", &ModelWrapper::Forward),
            InstanceMethod("backward", &ModelWrapper::Backward),
            InstanceMethod("update", &ModelWrapper::Update),
            InstanceMethod("addLayer", &ModelWrapper::AddLayer),
            InstanceMethod("load", &ModelWrapper::Load),
            InstanceMethod("save", &ModelWrapper::Save),
        });
        exports.Set("Model", func);
    }

private:
    static boat_tensor_t* BoatTensorFromJS(Napi::Value val) {
        if (!val.IsObject()) return nullptr;
        auto obj = val.As<Napi::Object>();

        auto data_fn = obj.Get("toFloat32Array");
        if (!data_fn.IsFunction()) return nullptr;

        auto shape_val = obj.Get("shape");
        auto data_val = data_fn.As<Napi::Function>().Call(obj, {});
        if (!data_val.IsTypedArray()) return nullptr;

        auto arr = data_val.As<Napi::Float32Array>();
        size_t n = arr.ElementLength();
        const float* data = arr.Data();

        auto js_shape = shape_val.As<Napi::Array>();
        size_t ndim = js_shape.Length();
        std::vector<int64_t> shape(ndim);
        for (size_t i = 0; i < ndim; i++) {
            shape[i] = (int64_t)js_shape.Get(i).ToNumber().DoubleValue();
        }

        return boat_tensor_from_data(shape.data(), ndim, BOAT_DTYPE_FLOAT32, data);
    }

    static Napi::Object TensorFromBoat(Napi::Env env, boat_tensor_t* t) {
        size_t n = boat_tensor_nelements(t);
        auto arr = Napi::Float32Array::New(env, n);
        const float* src = (const float*)boat_tensor_const_data(t);
        if (src) {
            std::memcpy(arr.Data(), src, n * sizeof(float));
        }

        auto shape = boat_tensor_shape(t);
        size_t ndim = boat_tensor_ndim(t);
        auto js_shape = Napi::Array::New(env, ndim);
        for (size_t i = 0; i < ndim; i++) {
            js_shape.Set(i, Napi::Number::New(env, (double)shape[i]));
        }

        auto result = g_tensor_ctor.New({arr, js_shape}).As<Napi::Object>();

        boat_tensor_unref(t);
        return result;
    }
};

Napi::Object InitModel(Napi::Env env, Napi::Object exports) {
    ModelWrapper::Init(env, exports);
    return exports;
}
