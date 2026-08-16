# Multi-Input Graph Forward (`boat_graph_forward`) + Merge Layers

Status: implemented (2026-08-16).

Boat's sequential `boat_model_forward` walks a layer chain. This document
covers the multi-input generalization: executing **any DAG** in topological
order with several inputs and several outputs, plus the merge layers
(`boat_concat_layer`, `boat_add_layer`) that make concatenation / addition /
skip-connection graphs expressible.

## `boat_graph_forward`

```c
typedef struct {
    const boat_node_t *node;
    boat_tensor_t *tensor; // in: input tensor; out: result tensor (ref'd)
} boat_graph_io_t;

int boat_graph_forward(const boat_graph_t *graph,
                       const boat_graph_io_t *inputs, size_t n_inputs,
                       const boat_graph_io_t *outputs, size_t n_outputs);
```

Execution rules:

- The graph is topologically sorted (Kahn); a cycle is an error.
- **Placeholder** nodes must be bound through `inputs`. An input binding may
  also override an **operation** node (external feed).
- **OUTPUT** nodes pass through their single forward input.
- **CONSTANT / VARIABLE** nodes whose data is a `boat_tensor_t*` resolve to
  that tensor.
- **OPERATION** nodes collect their incoming FORWARD-edge tensors (in edge
  insertion order) and run them through the forward evaluator:
  - the default evaluates `boat_layer_t*` node data via
    `boat_layer_ops.forward` (single input) or `boat_layer_ops.forward_many`
    (multi input), resolving the ops table with `boat_layer_resolve_ops()`
    when needed;
  - `boat_graph_set_forward_fn(graph, fn)` installs a custom evaluator for
    other node conventions.
- Requested outputs are written back with one reference held for the caller;
  all intermediate tensors are freed. Returns 0 on success, nonzero on error.

## Merge layers

```c
boat_layer_t *boat_concat_layer_create(int64_t dim); // 0-based; negative counts from the end
boat_layer_t *boat_add_layer_create(void);
```

Both consume N tensors through the `forward_many` layer signature:

```c
typedef struct { const boat_tensor_t *t; } boat_layer_input_t;

struct boat_layer_ops_t {
    boat_tensor_t *(*forward)(const boat_layer_t *, const boat_tensor_t *);
    boat_tensor_t *(*forward_many)(const boat_layer_t *,
                                   const boat_layer_input_t *, size_t n_inputs);
    /* ... backward / update / free unchanged ... */
};
```

- `boat_concat_layer_forward_many` joins along `dim` via
  `boat_tensor_concatenate`; a single input is an identity clone.
- `boat_add_layer_forward_many` folds `boat_add` over the inputs (broadcasting
  when shapes allow); a single input is an identity clone.

## Example: 2-input concatenation

```c
#include <boat/boat.h>

boat_graph_t *g = boat_graph_create();
boat_node_t *p1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
boat_node_t *p2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);

boat_layer_t *cat = (boat_layer_t *)malloc(sizeof(boat_layer_t));
cat->data = boat_concat_layer_create(0); // join on dim 0
cat->type = BOAT_LAYER_TYPE_CONCAT;
cat->ops  = NULL; // resolved by boat_layer_resolve_ops during execution
boat_node_t *cn = boat_graph_add_node(g, cat, BOAT_NODE_TYPE_OPERATION, NULL);
boat_node_t *out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);

boat_graph_add_edge(g, p1, cn, BOAT_EDGE_DIRECTION_FORWARD);
boat_graph_add_edge(g, p2, cn, BOAT_EDGE_DIRECTION_FORWARD);
boat_graph_add_edge(g, cn, out, BOAT_EDGE_DIRECTION_FORWARD);

int64_t sa[2] = {3, 2}, sb[2] = {2, 2};
const float av[6] = {1, 2, 3, 4, 5, 6};
const float bv[4] = {7, 8, 9, 10};
boat_tensor_t *a = boat_tensor_from_data(sa, 2, BOAT_DTYPE_FLOAT32, av);
boat_tensor_t *b = boat_tensor_from_data(sb, 2, BOAT_DTYPE_FLOAT32, bv);

boat_graph_io_t inputs[2]  = {{p1, a}, {p2, b}};
boat_graph_io_t outputs[1] = {{out, NULL}};
if (boat_graph_forward(g, inputs, 2, outputs, 1) == 0) {
    boat_tensor_t *o = outputs[0].tensor; // [5, 2] = av rows then bv rows
    /* ... use o (one reference owned by the caller) ... */
    boat_tensor_unref(o);
}

/* cleanup: inputs, layers, graph ... */
```
