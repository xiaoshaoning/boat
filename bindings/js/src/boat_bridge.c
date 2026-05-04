// C bridge — compiled as C, includes boat.h without C++ issues
#include "boat.h"

void boat_bridge_init(void) {
    boat_init();
}

const char* boat_bridge_version(void) {
    return boat_get_version_string();
}

const char* boat_bridge_last_error_message(void) {
    return boat_get_last_error_message();
}

void boat_bridge_free(void* ptr) {
    boat_free(ptr);
}
