layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_tensor { float data[MAP_INPUT_DIM]; };
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_tensor { float data[MAP_OUTPUT_DIM]; };

MAIN(tid)
{
    rdv_input_tensor input_tensor = rdv_input_tensor(parameters.input_tensor + MAP_INPUT_DIM * tid.x * 4);
    rdv_output_tensor output_tensor = rdv_output_tensor(parameters.output_tensor + MAP_OUTPUT_DIM * tid.x * 4);
    forward(parameters.map, input_tensor.data, output_tensor.data);
}