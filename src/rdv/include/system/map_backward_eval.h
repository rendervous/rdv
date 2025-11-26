layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_tensor { float data[MAP_INPUT_DIM]; };
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_tensor { float data[MAP_OUTPUT_DIM]; };

MAIN(tid)
{
    rdv_input_tensor input_tensor = rdv_input_tensor(parameters.input_tensor + MAP_INPUT_DIM * tid.x * 4);
    rdv_output_tensor output_grad_tensor = rdv_output_tensor(parameters.output_grad_tensor + MAP_OUTPUT_DIM * tid.x * 4);
    if (parameters.input_grad_tensor == 0)
    {
        float input_grad[MAP_INPUT_DIM];
        backward(parameters.map, input_tensor.data, output_grad_tensor.data, input_grad);
    }
    else
    {
        rdv_input_tensor input_grad_tensor = rdv_input_tensor(parameters.input_grad_tensor + MAP_INPUT_DIM * tid.x * 4);
        backward(parameters.map, input_tensor.data, output_grad_tensor.data, input_grad_tensor.data);
    }
}