# MATLAB Deep-Learning Layer Mapping

Status: implemented (2026-08-16). M2 of the deep-learning toolbox
integration: how a MATLAB `dlnetwork` (interpreter-side) maps onto boat
layers for graph export (`boat_export`), including exact weight layouts.

## Mapping table

| MATLAB layer | Boat layer / node | Layout notes |
|---|---|---|
| featureInputLayer | placeholder (boat `BOAT_NODE_TYPE_PLACEHOLDER`) | input tensor `[Q, I]` (Q samples, I features) |
| imageInputLayer | placeholder | input tensor `[Q, C, H, W]`; H/W/C from `InputSize` |
| fullyConnectedLayer | `boat_dense_layer` | MATLAB `Weights` `[O x I]` column-major == boat `[I, O]` row-major: **direct linear copy**; `Bias` `[O]` -> boat bias |
| reluLayer / tanhLayer / tansigLayer / sigmoidLayer | `boat_relu_layer` / `boat_tanh_layer` / `boat_sigmoid_layer` | elementwise, no layout change |
| softmaxLayer | `boat_softmax_layer` (axis 1) | softmax over the feature axis of `[Q, R]` |
| dropoutLayer | (identity at inference; node bypassed) | skipped in the exported graph |
| convolution2dLayer | `boat_conv_layer` | MATLAB `Weights` `[K x fh*fw*C]` (row k = filter, c-major then row then col) maps 1:1 to boat `[K, C, fh, fw]` row-major: **direct linear copy**; square filter/stride/symmetric padding required |
| batchNormalizationLayer | `boat_batchnorm2d_layer` (inference mode) | `Scale`/`Offset`/`TrainedMean`/`TrainedVariance` `[C x 1]` -> boat `[C]` tensors |
| maxPooling2dLayer / averagePooling2dLayer | `boat_pool_layer` (+ `boat_pool_layer_set_average`) | `PoolSize`/`Stride`/`Padding` 1:1 (square); average pooling supported since 2026-08 |
| globalAveragePooling2dLayer | `boat_pool_layer` (average, pool_size = spatial, stride 1) | square inputs only |
| flattenLayer | `boat_flatten_layer` | `[Q, C, H, W]` -> `[Q, R]` |
| lstmLayer | `boat_lstm_layer` | MATLAB `InputWeights`/`RecurrentWeights` `[G x D]` column-major == boat `[D, G]` row-major: **direct linear copy**; gate order `[i; f; g; o]` identical; MATLAB single `Bias` -> boat `bias_ih`, `bias_hh` = 0 |
| gruLayer | `boat_gru_layer` | gate order differs: MATLAB `[z; r; h]` vs boat `[r; z; n]` -> **gate-block reorder** on transfer (z<->r blocks, h -> n); single `Bias` -> `bias_ih`, `bias_hh` = 0 |
| concatenationLayer / depthConcatenationLayer | `boat_concat_layer` (dim 1) | MATLAB dim 1 (flat features) and dim 3 (channels) both join on boat dim 1 of `[Q, R]` / `[Q, C, H, W]` |
| additionLayer | `boat_add_layer` | element-wise sum |

## Layout conventions

The interpreter stores matrices as `[R, Q]` **column-major** (R = H*W*C
flattened, channel-major rows; Q = samples/sequence). Boat tensors are
row-major. The transposed shapes align so every boundary copy is a direct
linear copy with a double<->float conversion:

```
[R, Q] (mx)      == [Q, R] (boat)
[H*W*C, Q] (mx)  == [Q, C, H, W] (boat)
[H, Q] (mx rnn)  == [1, Q, H] (boat rnn output)
```

RNN layout adapters (`boat_tensor_reshape`) are inserted at the graph
boundary: `[Q, R] -> [1, Q, R]` before lstm/gru, and `[1, Q, H] -> [Q, H]`
after them.

## GRU gate order

Boat's internal GRU gate vector is `[r; z; n]` (reset, update, candidate);
MATLAB/PyTorch is `[z; r; h]`. The exporter remaps the three G-row blocks on
transfer (`dl_gru_gate_map = {1, 0, 2}`). LSTM is already aligned
(`[i; f; g; o]`).

## GRU semantics (corrected 2026-08)

Boat's GRU now follows the standard formulation:

- candidate: `n = tanh(a_ih[n] + (r .* h_prev) @ W_hh[n] + b_hh[n])`
  (the reset is applied to `h_prev` **before** the recurrent matmul)
- update:  `h = (1 - z) .* h_prev + z .* n`

Both the forward and the analytic backward were corrected; the numerical
gradient check in `tests/unit/test_lstm_gru.c` covers all weight/bias/input
gradients.
