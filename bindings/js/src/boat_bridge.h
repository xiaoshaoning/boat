// Thin C bridge to boat.h — compiles as C, exposes clean C-compatible functions
// for use from C++ N-API code. Avoids C/C++ linkage conflicts in boat.h.

typedef int boat_bridge_error_t;  // matches boat_error_t underlying type

#ifdef __cplusplus
extern "C" {
#endif

void            boat_bridge_init(void);
const char*     boat_bridge_version(void);
const char*     boat_bridge_last_error_message(void);
void            boat_bridge_free(void* ptr);

#ifdef __cplusplus
}
#endif
