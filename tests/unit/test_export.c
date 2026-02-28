// test_export.c - Simple test to check if functions are exported
#include <stdio.h>
#include <windows.h>

int main() {
    HMODULE hDll = LoadLibraryA("boat.dll");
    if (!hDll) {
        printf("Failed to load boat.dll\n");
        return 1;
    }

    // Try to get addresses of some key functions
    void* funcs[] = {
        GetProcAddress(hDll, "boat_tensor_create"),
        GetProcAddress(hDll, "boat_tensor_unref"),
        GetProcAddress(hDll, "boat_variable_create"),
        GetProcAddress(hDll, "boat_variable_free"),
        NULL
    };

    const char* names[] = {
        "boat_tensor_create",
        "boat_tensor_unref",
        "boat_variable_create",
        "boat_variable_free",
        NULL
    };

    printf("Checking exported functions from boat.dll:\n");
    for (int i = 0; names[i] != NULL; i++) {
        if (funcs[i]) {
            printf("  ✓ %s: 0x%p\n", names[i], funcs[i]);
        } else {
            printf("  ✗ %s: NOT EXPORTED\n", names[i]);
        }
    }

    FreeLibrary(hDll);
    return 0;
}