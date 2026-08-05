# cJSON (vendored)

Vendored copy of [cJSON](https://github.com/DaveGamble/cJSON) v1.7.18
(MIT license, retained in cjson/cJSON.c and cjson/cJSON.h).

cJSON is bundled so -DBOAT_WITH_HUGGINGFACE=ON builds without a system
cJSON dependency on any platform. The cjson/ subdirectory mirrors the
standard installed layout so #include <cjson/cJSON.h> works unchanged.

To update, replace cjson/cJSON.c and cjson/cJSON.h from the upstream
release tag and bump the version above.