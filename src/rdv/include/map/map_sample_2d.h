/* Parameters
grid: deferred
*/

void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight, float y_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += y_weight * mix(src_left.data[i], src_right.data[i], x_weight);
}

FORWARD {
    Tensor grid = load_deferred(parameters.grid);
    for(int i=0; i<OUTPUT_DIM; i++) _output[i] = 0.0;
    vec2 coord = vec2(_input[0], _input[1])*0.5 + 0.5;
    vec2 grid_size = vec2(float(grid.shape[1]), float(grid.shape[0]));
    #ifdef ALIGN_CORNERS
    vec2 index_f = coord * (grid_size - vec2(1.0));
    #else
    vec2 index_f = coord * grid_size - vec2(0.5);
    #endif
    ivec2 index0 = ivec2(floor(index_f));
    ivec2 index1 = index0 + ivec2(1);
    vec2 alpha = index_f - vec2(index0);
    index0 = clamp(index0, ivec2(0), ivec2(grid_size) - ivec2(1));
    index1 = clamp(index1, ivec2(0), ivec2(grid_size) - ivec2(1));
    int stride_x = OUTPUT_DIM * 4;
    int stride_y = int(grid.shape[1]) * stride_x;
    GPUPtr ptr = grid.data_ptr + index0.x * stride_x + index0.y * stride_y;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, (1 - alpha.y));
    stride_y *= (index1.y - index0.y);
    ptr += stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, alpha.y);
}

void scanline_interpolator_bw(MAP_DECL, float_ptr dst_a_grad, float_ptr dst_b_grad,
in float _output_grad[OUTPUT_DIM], float x_weight, float y_weight) {
    for (int i=0; i<OUTPUT_DIM; i++) {
        atomicAdd_f(dst_a_grad, i, y_weight * (1 - x_weight) * _output_grad[i]);
        atomicAdd_f(dst_b_grad, i, y_weight * x_weight * _output_grad[i]);
    }
}

BACKWARD {
    Tensor grid = load_deferred(parameters.grid);
#ifndef INPUT_REQUIRES_GRAD
    if (grid.grad_ptr == 0) return;  // no gradient needed
#else
    NOT_SUPPORTED("Input requires grad");
#endif
    vec2 coord = vec2(_input[0], _input[1])*0.5 + 0.5;
    vec2 grid_size = vec2(float(grid.shape[1]), float(grid.shape[0]));
    #ifdef ALIGN_CORNERS
    vec2 index_f = coord * (grid_size - vec2(1.0));
    #else
    vec2 index_f = coord * grid_size - vec2(0.5);
    #endif
    ivec2 index0 = ivec2(floor(index_f));
    ivec2 index1 = index0 + ivec2(1);
    vec2 alpha = index_f - vec2(index0);
    index0 = clamp(index0, ivec2(0), ivec2(grid_size) - ivec2(1));
    index1 = clamp(index1, ivec2(0), ivec2(grid_size) - ivec2(1));
    int stride_x = OUTPUT_DIM * 4;
    int stride_y = int(grid.shape[1]) * stride_x;
    GPUPtr ptr = grid.grad_ptr + index0.x * stride_x + index0.y * stride_y;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, (1 - alpha.y));
    stride_y *= (index1.y - index0.y);
    ptr += stride_y;
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, alpha.y);
}