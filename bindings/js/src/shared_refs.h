#ifndef BOAT_JS_SHARED_REFS_H
#define BOAT_JS_SHARED_REFS_H

#include <napi.h>

// Module-level persistent reference to the Tensor constructor.
// Set by TensorWrapper::Init, read by ModelWrapper.
extern Napi::FunctionReference g_tensor_ctor;

#endif // BOAT_JS_SHARED_REFS_H
