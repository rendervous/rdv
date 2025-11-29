/*
t: deferrable_field
*/
BUFFER_DECL(t, OUTPUT_DIM)

FORWARD {
    Tensor t = load_deferred(parameters.t);
    //float_ptr t_data = float_ptr(t.data_ptr);
    BUFFER(t) t_data = BUFFER(t) (t.data_ptr);
    _output = t_data.data;
    //for (int i=0; i<OUTPUT_DIM; i++) _output[i] = t_data.data[i];
}

BACKWARD {
    Tensor t = load_deferred(parameters.t);
    if (t.grad_ptr == 0)
    return;
    float_ptr t_grad = float_ptr(t.grad_ptr);
    for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(t_grad, i, _output_grad[i]);
}