#include <napi.h>
#include "boat_bridge.h"
#include "boat/tensor.h"
#include "boat/model.h"
#include "boat/ops.h"
#include "shared_refs.h"

// Forward declarations from wrapper files
Napi::Object InitTensor(Napi::Env env, Napi::Object exports);
Napi::Object InitModel(Napi::Env env, Napi::Object exports);

// Module-level Tensor constructor reference
Napi::FunctionReference g_tensor_ctor;

// Standalone utilities
Napi::Value BoatVersion(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), boat_bridge_version());
}

Napi::Value DtypeSize(const Napi::CallbackInfo& info) {
    int dtype = info[0].ToNumber().Int32Value();
    return Napi::Number::New(info.Env(), boat_dtype_size((boat_dtype_t)dtype));
}

Napi::Value DtypeName(const Napi::CallbackInfo& info) {
    int dtype = info[0].ToNumber().Int32Value();
    const char* name = boat_dtype_name((boat_dtype_t)dtype);
    return Napi::String::New(info.Env(), name ? name : "unknown");
}

Napi::Value CheckError(const Napi::CallbackInfo& info) {
    const char* msg = boat_bridge_last_error_message();
    if (msg) return Napi::String::New(info.Env(), msg);
    return info.Env().Undefined();
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    boat_bridge_init();

    exports.Set("boatVersion", Napi::String::New(env, boat_bridge_version()));
    exports.Set("dtypeSize", Napi::Function::New(env, DtypeSize));
    exports.Set("dtypeName", Napi::Function::New(env, DtypeName));
    exports.Set("checkError", Napi::Function::New(env, CheckError));

    InitTensor(env, exports);
    InitModel(env, exports);

    return exports;
}

NODE_API_MODULE(boat_napi, InitAll)
