#include <napi.h>
#include "boat_bridge.h"
#include "boat/tensor.h"
#include "boat/ops.h"
#include "shared_refs.h"
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

class TensorWrapper : public Napi::ObjectWrap<TensorWrapper> {
    boat_tensor_t* tensor_;

public:
    TensorWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<TensorWrapper>(info), tensor_(nullptr) {
        Napi::Env env = info.Env();

        if (info.Length() == 0) return;

        // new Tensor(shape: number[], dtype?: string)
        if (info[0].IsArray()) {
            auto js_shape = info[0].As<Napi::Array>();
            size_t ndim = js_shape.Length();
            if (ndim > BOAT_MAX_DIMS || ndim == 0) {
                Napi::RangeError::New(env, "Invalid number of dimensions").ThrowAsJavaScriptException();
                return;
            }
            std::vector<int64_t> shape(ndim);
            for (size_t i = 0; i < ndim; i++) {
                Napi::Value v = js_shape[i];
                double dval = v.IsNumber() ? v.As<Napi::Number>().DoubleValue() : 0.0;
                shape[i] = (int64_t)dval;
            }
            boat_dtype_t dtype = BOAT_DTYPE_FLOAT32;
            if (info.Length() > 1 && info[1].IsString()) {
                dtype = DTypeFromString(info[1].As<Napi::String>().Utf8Value().c_str());
            }
            tensor_ = boat_tensor_create(shape.data(), ndim, dtype, BOAT_DEVICE_CPU);
            if (!tensor_) {
                Napi::Error::New(env, "Failed to create tensor").ThrowAsJavaScriptException();
            }
            return;
        }

        // new Tensor(data: Float32Array, shape?: number[])
        if (info[0].IsTypedArray()) {
            auto arr = info[0].As<Napi::Float32Array>();
            size_t n = arr.ElementLength();
            const float* data = arr.Data();

            if (info.Length() > 1 && info[1].IsArray()) {
                auto js_shape = info[1].As<Napi::Array>();
                size_t ndim = js_shape.Length();
                if (ndim > BOAT_MAX_DIMS) {
                    Napi::RangeError::New(env, "Too many dimensions").ThrowAsJavaScriptException();
                    return;
                }
                std::vector<int64_t> shape(ndim);
                for (size_t i = 0; i < ndim; i++) {
                    shape[i] = (int64_t)js_shape.Get(i).ToNumber().DoubleValue();
                }
                tensor_ = boat_tensor_from_data(shape.data(), ndim, BOAT_DTYPE_FLOAT32, data);
            } else {
                int64_t shape[1] = {(int64_t)n};
                tensor_ = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);
            }
            if (!tensor_) {
                Napi::Error::New(env, "Failed to create tensor from data").ThrowAsJavaScriptException();
            }
            return;
        }

        Napi::TypeError::New(env, "Unsupported Tensor constructor arguments").ThrowAsJavaScriptException();
    }

    ~TensorWrapper() {
        if (tensor_) boat_tensor_unref(tensor_);
    }

    // ---- Properties ----
    Napi::Value Shape(const Napi::CallbackInfo& info) {
        auto shape = boat_tensor_shape(tensor_);
        size_t ndim = boat_tensor_ndim(tensor_);
        auto arr = Napi::Array::New(info.Env(), ndim);
        for (size_t i = 0; i < ndim; i++) {
            arr.Set(i, Napi::Number::New(info.Env(), (double)shape[i]));
        }
        return arr;
    }

    Napi::Value Dtype(const Napi::CallbackInfo& info) {
        auto dt = boat_tensor_dtype(tensor_);
        return Napi::String::New(info.Env(), boat_dtype_name(dt));
    }

    Napi::Value Device(const Napi::CallbackInfo& info) {
        auto dev = boat_tensor_device(tensor_);
        const char* name = (dev == BOAT_DEVICE_CUDA) ? "cuda" : "cpu";
        return Napi::String::New(info.Env(), name);
    }

    Napi::Value Ndims(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), boat_tensor_ndim(tensor_));
    }

    Napi::Value Size(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), (double)boat_tensor_nelements(tensor_));
    }

    Napi::Value Nbytes(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), (double)boat_tensor_nbytes(tensor_));
    }

    // ---- Data Access ----
    Napi::Value ToFloat32Array(const Napi::CallbackInfo& info) {
        size_t n = boat_tensor_nelements(tensor_);
        auto arr = Napi::Float32Array::New(info.Env(), n);
        const float* src = (const float*)boat_tensor_const_data(tensor_);
        if (src) {
            std::memcpy(arr.Data(), src, n * sizeof(float));
        }
        return arr;
    }

    Napi::Value ToString(const Napi::CallbackInfo& info) {
        char* s = boat_tensor_to_string(tensor_);
        if (!s) return info.Env().Undefined();
        Napi::String result = Napi::String::New(info.Env(), s);
        boat_bridge_free(s);
        return result;
    }

    // ---- Manipulation ----
    Napi::Value Reshape(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        if (info.Length() < 1 || !info[0].IsArray()) {
            Napi::TypeError::New(env, "reshape expects an array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto js_shape = info[0].As<Napi::Array>();
        size_t ndim = js_shape.Length();
        std::vector<int64_t> shape(ndim);
        for (size_t i = 0; i < ndim; i++) {
            shape[i] = (int64_t)js_shape.Get(i).ToNumber().DoubleValue();
        }
        boat_tensor_t* result = boat_tensor_reshape(tensor_, shape.data(), ndim);
        if (!result) {
            Napi::Error::New(env, "Reshape failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        return WrapTensor(env, result);
    }

    Napi::Value Transpose(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        if (info.Length() < 2) {
            Napi::TypeError::New(env, "transpose expects two integers").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        size_t dim0 = (size_t)info[0].ToNumber().DoubleValue();
        size_t dim1 = (size_t)info[1].ToNumber().DoubleValue();
        size_t ndim = boat_tensor_ndim(tensor_);
        std::vector<size_t> perm(ndim);
        for (size_t i = 0; i < ndim; i++) perm[i] = i;
        std::swap(perm[dim0], perm[dim1]);

        boat_tensor_t* result = boat_tensor_transpose(tensor_, perm.data(), ndim);
        if (!result) {
            Napi::Error::New(env, "Transpose failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        return WrapTensor(env, result);
    }

    Napi::Value Clone(const Napi::CallbackInfo& info) {
        boat_tensor_t* result = boat_tensor_clone(tensor_);
        if (!result) {
            Napi::Error::New(info.Env(), "Clone failed").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        return WrapTensor(info.Env(), result);
    }

    // ---- Static ops ----
    static Napi::Value Add(const Napi::CallbackInfo& info) {
        return BinaryOp(info, boat_add);
    }
    static Napi::Value Sub(const Napi::CallbackInfo& info) {
        return BinaryOp(info, boat_sub);
    }
    static Napi::Value Mul(const Napi::CallbackInfo& info) {
        return BinaryOp(info, boat_mul);
    }
    static Napi::Value Div(const Napi::CallbackInfo& info) {
        return BinaryOp(info, boat_div);
    }
    static Napi::Value MatMul(const Napi::CallbackInfo& info) {
        return BinaryOp(info, boat_matmul);
    }
    static Napi::Value Relu(const Napi::CallbackInfo& info) {
        return UnaryOp(info, boat_relu);
    }
    static Napi::Value Sigmoid(const Napi::CallbackInfo& info) {
        return UnaryOp(info, boat_sigmoid);
    }

    static Napi::Value Softmax(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        auto* t = GetBoatTensor(info[0]);
        if (!t) return env.Undefined();
        int axis = (info.Length() > 1) ? info[1].ToNumber().Int32Value() : -1;
        boat_tensor_t* result = boat_softmax(t, axis);
        if (!result) {
            Napi::Error::New(env, "softmax failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        return WrapTensor(env, result);
    }

    static void Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Tensor", {
            InstanceMethod("shape", &TensorWrapper::Shape),
            InstanceMethod("dtype", &TensorWrapper::Dtype),
            InstanceMethod("device", &TensorWrapper::Device),
            InstanceMethod("ndim", &TensorWrapper::Ndims),
            InstanceMethod("size", &TensorWrapper::Size),
            InstanceMethod("nbytes", &TensorWrapper::Nbytes),
            InstanceMethod("toFloat32Array", &TensorWrapper::ToFloat32Array),
            InstanceMethod("toString", &TensorWrapper::ToString),
            InstanceMethod("reshape", &TensorWrapper::Reshape),
            InstanceMethod("transpose", &TensorWrapper::Transpose),
            InstanceMethod("clone", &TensorWrapper::Clone),
            StaticMethod("add", &TensorWrapper::Add),
            StaticMethod("sub", &TensorWrapper::Sub),
            StaticMethod("mul", &TensorWrapper::Mul),
            StaticMethod("div", &TensorWrapper::Div),
            StaticMethod("matmul", &TensorWrapper::MatMul),
            StaticMethod("relu", &TensorWrapper::Relu),
            StaticMethod("sigmoid", &TensorWrapper::Sigmoid),
            StaticMethod("softmax", &TensorWrapper::Softmax),
        });
        exports.Set("Tensor", func);
        g_tensor_ctor = Napi::Persistent(func);
        g_tensor_ctor.SuppressDestruct();
    }

    // Public helpers
    static boat_tensor_t* GetBoatTensor(Napi::Value val) {
        if (!val.IsObject()) return nullptr;
        auto obj = val.As<Napi::Object>();
        auto* wrapper = Napi::ObjectWrap<TensorWrapper>::Unwrap(obj);
        return wrapper ? wrapper->tensor_ : nullptr;
    }

    static Napi::Object WrapTensor(Napi::Env env, boat_tensor_t* t) {
        auto obj = g_tensor_ctor.New({}).As<Napi::Object>();
        auto* wrapper = Napi::ObjectWrap<TensorWrapper>::Unwrap(obj);
        if (wrapper) wrapper->tensor_ = t;
        return obj;
    }

private:
    static boat_dtype_t DTypeFromString(const char* name) {
        for (int i = 0; i < BOAT_DTYPE_COUNT; i++) {
            const char* n = boat_dtype_name((boat_dtype_t)i);
            if (n && strcmp(n, name) == 0) return (boat_dtype_t)i;
        }
        return BOAT_DTYPE_FLOAT32;
    }

    static Napi::Value BinaryOp(const Napi::CallbackInfo& info,
                                boat_tensor_t* (*op)(const boat_tensor_t*, const boat_tensor_t*)) {
        Napi::Env env = info.Env();
        if (info.Length() < 2) {
            Napi::TypeError::New(env, "Expected two Tensor arguments").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto* a = GetBoatTensor(info[0]);
        auto* b = GetBoatTensor(info[1]);
        if (!a || !b) {
            Napi::TypeError::New(env, "Arguments must be Tensors").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        boat_tensor_t* result = op(a, b);
        if (!result) {
            const char* err = boat_bridge_last_error_message();
            if (err && err[0]) {
                Napi::Error::New(env, err).ThrowAsJavaScriptException();
            } else {
                Napi::Error::New(env, "Operation failed").ThrowAsJavaScriptException();
            }
            return env.Undefined();
        }
        return WrapTensor(env, result);
    }

    static Napi::Value UnaryOp(const Napi::CallbackInfo& info,
                               boat_tensor_t* (*op)(const boat_tensor_t*)) {
        Napi::Env env = info.Env();
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected a Tensor argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto* a = GetBoatTensor(info[0]);
        if (!a) {
            Napi::TypeError::New(env, "Argument must be a Tensor").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        boat_tensor_t* result = op(a);
        if (!result) {
            Napi::Error::New(env, "Operation failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        return WrapTensor(env, result);
    }
};

Napi::Object InitTensor(Napi::Env env, Napi::Object exports) {
    TensorWrapper::Init(env, exports);
    return exports;
}
