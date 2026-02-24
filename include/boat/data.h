// data.h - Data handling
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_DATA_H
#define BOAT_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

// Dataset structure (opaque)
typedef struct boat_dataset_t boat_dataset_t;

// Data loader structure (opaque)
typedef struct boat_dataloader_t boat_dataloader_t;

// Dataset operations
boat_dataset_t* boat_dataset_create(const void* data, const void* labels, size_t n_samples);
void boat_dataset_free(boat_dataset_t* dataset);
size_t boat_dataset_size(const boat_dataset_t* dataset);

// Data loader operations
boat_dataloader_t* boat_dataloader_create(boat_dataset_t* dataset, size_t batch_size, bool shuffle);
void boat_dataloader_free(boat_dataloader_t* loader);
bool boat_dataloader_next(boat_dataloader_t* loader, void** batch_data, void** batch_labels);

#ifdef __cplusplus
}
#endif

#endif // BOAT_DATA_H