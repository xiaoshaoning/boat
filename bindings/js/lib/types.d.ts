export type DType =
    | 'float64' | 'float32' | 'float16' | 'float8' | 'float4'
    | 'int64' | 'int32' | 'uint8' | 'int8'
    | 'bits2' | 'bits1' | 'bool' | 'bfloat16';

export type Device = 'cpu' | 'cuda';

export type Shape = readonly number[];

export class Tensor {
    /** Create tensor with given shape and dtype (zero-filled) */
    constructor(shape: Shape, dtype?: DType);

    /** Create tensor from a Float32Array with optional shape */
    constructor(data: Float32Array, shape?: Shape);

    /** Tensor dimensions */
    readonly shape: Shape;
    readonly dtype: DType;
    readonly device: Device;
    readonly ndim: number;
    readonly size: number;
    readonly nbytes: number;

    /** Copy tensor data to a Float32Array */
    toFloat32Array(): Float32Array;

    /** Human-readable string representation */
    toString(): string;

    /** Return a reshaped copy */
    reshape(shape: Shape): Tensor;

    /** Return a transposed copy */
    transpose(dim0: number, dim1: number): Tensor;

    /** Return a clone */
    clone(): Tensor;

    // Static operations
    static add(a: Tensor, b: Tensor): Tensor;
    static sub(a: Tensor, b: Tensor): Tensor;
    static mul(a: Tensor, b: Tensor): Tensor;
    static div(a: Tensor, b: Tensor): Tensor;
    static matmul(a: Tensor, b: Tensor): Tensor;
    static relu(a: Tensor): Tensor;
    static sigmoid(a: Tensor): Tensor;
    static softmax(a: Tensor, axis?: number): Tensor;
}

export class Model {
    constructor();

    forward(input: Tensor): Tensor;
    backward(grad: Tensor): Tensor;
    update(lr?: number): void;

    load(path: string): void;
    save(path: string): void;
}

/** Version string */
export const boatVersion: string;

/** Return the byte size of a dtype */
export function dtypeSize(dtype: number): number;

/** Return the name of a dtype */
export function dtypeName(dtype: number): string;
