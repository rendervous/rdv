from typing import Any

import torch as _torch
import os as _os

import vulky as _vk
import typing as _typing
import enum as _enum
from functools import cached_property


__MANUAL_SEED__ : _typing.Optional[int] = None
__SEEDS_TENSOR__ : _typing.Optional[_torch.Tensor] = None
__INCLUDE_PATH__ = _os.path.dirname(__file__).replace('\\','/') + '/include'
__TORCH_DEVICE__ = _torch.device('cuda:0') if _torch.cuda.is_available() else _torch.device('cpu')  #TODO: Check with AMD
__RDV_PATH__ = _os.path.dirname(__file__)

def device() -> _torch.device:
    """
    Gets the torch device visible by vulkan backend.
    Ensure tensors are valid for rdv compute and maps using:
    >>> t.to(rdv.device())
    """
    return __TORCH_DEVICE__


def seed(manual_seed: _typing.Optional[int] = None):
    """
    Sets the seed used for the generation of torch and rdv randoms.
    Useful for replication.

    Example
    -------
    >>> import torch
    >>> import rdv
    >>> rdv.seed(5)
    >>> print(rdv.randn.cast(output_dim=3)(torch.zeros(10,1)))
    """
    global __MANUAL_SEED__
    __MANUAL_SEED__ = manual_seed
    _generate_seeds()


def _generate_seeds() -> _typing.Tuple[int, int, int, int]:
    """
    Creates the new 4 int seed for hybrid-Taus rgn algorithm.
    """
    import numpy as np
    global __MANUAL_SEED__, __SEEDS_TENSOR__
    if __MANUAL_SEED__ is not None:
        np.random.seed(__MANUAL_SEED__)
        # _torch.manual_seed(__MANUAL_SEED__)
        __MANUAL_SEED__ = None
    # if __SEEDS_TENSOR__ is None:
    #     __SEEDS_TENSOR__ = _vk.ivec4(0, 0, 0, 0)
    # _torch.randint(low=129, high=1 << 30, size=(4,), dtype=_torch.int32, out=__SEEDS_TENSOR__)
    __SEEDS_TENSOR__ = np.random.randint(1 << 30 - 129) + 129, np.random.randint(1 << 30 - 129) + 129, np.random.randint(1 << 30 - 129) + 129, np.random.randint(1 << 30 - 129) + 129
    return __SEEDS_TENSOR__


def time_check(tag: str = ""):
    import time
    class MeassurementContext:
        def __enter__(self):
            self.start_time = time.perf_counter()
        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"{tag}:{time.perf_counter() - self.start_time}s")
    return MeassurementContext()

# Read compute template
with open(__RDV_PATH__ + '/include/compute_template.h') as f:
    __COMPUTE_TEMPLATE__ = f.read()


class BACKWARD_IMPLEMENTATIONS(_enum.IntEnum):
    NONE = 0
    """
    The map source code doesn't contains any backward function.
    """
    DEFAULT = 1
    """
    The map source code contains only the default backward function (input, output_grad, input_grad).
    """
    WITH_OUTPUT = 2
    """
    The map source code contains only the output provided backward function (input, output, output_grad, input_grad).
    """
    ALL = 3
    """
    The map source code contains both backward functions, with and without output.
    """


class _DeferredParametersManager:

    rdv_named_tensors_buffer = None
    rdv_named_tensors_info = { }  # map from key to { key_id:int, references:int } representing id and references count
    rdv_named_tensors_free_ids = []  # reusable ids for named tensors
    rdv_deferred_parameters_buffer = None
    rdv_deferred_parameters_info = { }  # map each tuple of name_id and indices to { index: int, references: int } representing index and references count
    rdv_deferred_parameters_free_ids = []  # reusable ids for deferred parameters
    rdv_wrapped_tensors = {}  # wrapped tensors bound

    @classmethod
    def init(cls):
        cls.rdv_named_tensors_buffer = _vk.structured_buffer(16, element_description=dict(
            data=_torch.int64,  # data ptr of the contiguous bound tensor
            grad_data=_torch.int64,
            dim = int,  # dimension of the bound tensor
            shape=[6, int],  # dimensions of the bound tensor
        ), memory=_vk.MemoryLocation.CPU, usage=_vk.BufferUsage.STORAGE)
        cls.rdv_deferred_parameters_buffer = _vk.structured_buffer(16, element_description=dict(
            name_id=int,  # name id of the key for the tensor.
            number_of_indices=int,  # number of indices used in this deferred parameter
            map_dim=int,  # dimension required by the map.
            indices=[4, int],  # indices set for the tensor
        ), memory=_vk.MemoryLocation.CPU, usage=_vk.BufferUsage.STORAGE)
        cls.rdv_is_initialized = True

    @classmethod
    def resolve(cls, key, map_dim, indices) -> int:
        if key not in cls.rdv_named_tensors_info:
            if len(cls.rdv_named_tensors_free_ids) > 0:
                key_id = cls.rdv_named_tensors_free_ids.pop()
            else:
                key_id = len(cls.rdv_named_tensors_info)
            cls.rdv_named_tensors_info[key] = { 'key_id':key_id, 'references': 1}
        else:
            key_id = cls.rdv_named_tensors_info[key]['key_id']
            cls.rdv_named_tensors_info[key]['references'] += 1
        k = (key, map_dim, indices)
        if k not in cls.rdv_deferred_parameters_info:
            if len(cls.rdv_deferred_parameters_free_ids) > 0:
                index = cls.rdv_deferred_parameters_free_ids.pop()
            else:
                index = len(cls.rdv_deferred_parameters_info)
                assert index < 1024*1024
            cls.rdv_deferred_parameters_info[k] = { 'index': index, 'references': 1}
        else:
            index = cls.rdv_deferred_parameters_info[k]['index']
            cls.rdv_deferred_parameters_info[k]['references'] += 1
        # update buffer of deferred parameters with the key_id and indices
        b = cls.rdv_deferred_parameters_buffer.direct_map()
        b.number_of_indices[index] = len(indices)
        for i, idx in enumerate(indices):
            b.indices[i][index] = idx
        b.name_id[index] = key_id
        b.map_dim[index] = map_dim
        return index

    @classmethod
    def free(cls, key, map_dim, indices):
        assert (key, map_dim, indices) in cls.rdv_deferred_parameters_info
        info = cls.rdv_deferred_parameters_info[(key, map_dim, indices)]
        info['references'] -= 1
        if info['references'] == 0:  # deferred info index can be reused
            cls.rdv_deferred_parameters_info.pop((key, map_dim, indices))
            cls.rdv_deferred_parameters_free_ids.append(info['index'])
        info = cls.rdv_named_tensors_info[key]
        info['references'] -= 1
        if info['references'] == 0:  # key info index can be reused
            cls.rdv_named_tensors_info.pop(key)
            cls.rdv_named_tensors_free_ids.append(info['key_id'])

    @classmethod
    def bind(cls, named_tensors: _typing.Dict[str, _torch.Tensor]):
        grads = { }
        b = cls.rdv_named_tensors_buffer.direct_map()
        grad_data = b.grad_data
        data = b.data
        dim = b.dim
        shapes = b.shape
        for key, t in named_tensors.items():
            key_id = cls.rdv_named_tensors_info[key]['key_id']
            wgpu = _vk.wrap_gpu(t, 'in')
            cls.rdv_wrapped_tensors[key_id] = wgpu
            data[key_id] = wgpu.device_ptr
            dim[key_id] = len(t.shape)
            for i, d in enumerate(t.shape):
                shapes[i][key_id] = d
            if t.requires_grad:
                g = _vk.tensor_like(t)
                _torch.zero_(g)
                grad_data[key_id] = g.device_ptr  # valid for a vulkan tensor
                grads[key] = g
        return grads

    @classmethod
    def unbind(cls, named_tensors: _typing.Dict[str, _torch.Tensor]):
        for key, t in named_tensors.items():
            key_id = cls.rdv_named_tensors_info[key]['key_id']
            cls.rdv_wrapped_tensors[key_id].unwrap()


    @classmethod
    def validate(cls, named_tensors: _typing.Dict[str, _torch.Tensor], *deferred):
        pass  # TODO: Implement this to serve as a debug tool


class deferred:
    def __init__(self, key: str, map_dim: _typing.Optional[int] = None, indices: tuple = ()):
        self._key = key
        self._indices = indices
        self._map_dim = map_dim
        if self._map_dim is not None:
            self.id = _DeferredParametersManager.resolve(key, map_dim, indices)

    def cast(self, map_dim: int) -> 'deferred':
        assert isinstance(map_dim, int) and map_dim > 0, "map_dim must be a valid int"
        assert self._map_dim is None or self._map_dim == map_dim, f"Can not change an existing map dim {self._map_dim} for {map_dim}"
        if self._map_dim == map_dim:
            return self
        return deferred(self._key, map_dim, self._indices)

    def __del__(self):
        if self._map_dim is not None:
            _DeferredParametersManager.free(self._key, self._map_dim, self._indices)


def ensure_tensor(t: _typing.Union[_torch.Tensor, deferred], map_dim: int):
    if isinstance(t, _torch.Tensor):
        assert len(t.shape) == map_dim
        return t
    else:
        assert isinstance(t, deferred)
        return t.cast(map_dim=map_dim)


DeferrableField = dict(
    __name__ = 'DeferrableField',
    data=_torch.Tensor,
    shape=[6, int],  # shape of the tensor if directly bound
    deferred_index=int
)
"""
Represents a parameter that receives a tensor or a deferred parameter.
"""


MeshInfo = dict(
    __name__='MeshInfo',
    positions=_torch.Tensor,
    normals=_torch.Tensor,
    coordinates=_torch.Tensor,
    tangents=_torch.Tensor,
    binormals=_torch.Tensor,
    indices=_torch.Tensor
)
"""
Represents a mesh object in a compute.
"""


RaycastableInfo = dict(
    __name__='RaycastableInfo',
    callable_map=_torch.int64,
    explicit_info=_torch.int64,
)
"""
Raycastable objects in rdv are a union between meshes or AABBs (explicit info) and callable maps acting as geometries.
"""


class _DispatcherEngine(object):
    __COMPUTE_IDS__ = 0  # auto-increment id for all different kernel codes generated in the app.
    __COMPUTE_ID_BY_SIGNATURE__ = {}  # Compute id for a map signature. From tuple with signature to id.
    __KERNELS__ = {}  # Kernel code for each compute id.
    __CODENAMES__ = {}  # Map compute codename for each compute id.
    __DIMENSIONS__ = {}  # Input and output dimensions for each compute id.
    __INCLUDE_DIRS__ = {}  # Included dirs for each compute id
    __DEFINED_STRUCTS__ = {}  # All defined structures
    __BUILTIN_STRUCTS__ = []  # Defined structs in core.h
    __ENGINE_OBJECTS__ = None  # Objects to dispatch map evaluation and raycasting
    __EVAL_PIPELINES__ = {}  # From static signature to pipeline object
    __EVAL_MANAGERS__ = {}  # From dispatch signature to pipeline object
    __SYSTEM_BUFFER__ = None  # Object buffer with system values on it.
    __RANDOMS__ = []

    @classmethod
    def generate_map_kernels(cls, compute_ids: _typing.Iterable[int]) -> (str, _typing.Set[str]):
        """
        Generates the code for evaluating all maps provided and their dependencies.
        Returns the code and the list of required dirs to include.
        """
        codes = []
        for k, s in cls.__DEFINED_STRUCTS__.items():
            if k not in cls.__BUILTIN_STRUCTS__:
                codes.append(s)  # append all external defined structs not matter dependences
        include_dirs = set()
        for m in compute_ids:
            include_dirs.update(cls.__INCLUDE_DIRS__[m])
        codes += [cls.__KERNELS__[k] for k in sorted(compute_ids)]
        return '\n\r'.join(codes), include_dirs

    @classmethod
    def create_code_for_dynamic_calls(cls, map: 'Map', input_dim, output_dim, input_requires, bw_uses_output):
        fw_cases = ""
        bw_cases = ""
        children_kernel_ids = set([d.rdv_kernel_id for d in map.children])
        input_grad_parameter = f", inout float _input_grad[{input_dim}]" if input_requires else ""
        output_parameter = f", in float _output[{output_dim}]" if bw_uses_output else ""
        for id in children_kernel_ids:
            code_name = cls.__CODENAMES__[id]
            comp_input_dim, comp_output_dim, comp_input_requires, comp_bw_uses_output = cls.__DIMENSIONS__[id]
            if comp_input_dim == input_dim and comp_output_dim == output_dim and comp_input_requires == input_requires and comp_bw_uses_output == bw_uses_output:
                fw_cases += f"""
                    case {id}: forward({code_name}(buffer_{code_name}(dynamic_map)), _input, _output); break;
                    """
                bw_cases += f"""
                    case {id}: backward({code_name}(buffer_{code_name}(dynamic_map)), _input{output_parameter}, _output_grad{input_grad_parameter}); break;
                    """
        return f"""
    void dynamic_forward (MAP_DECL, GPUPtr dynamic_map, in float _input[{input_dim}], out float _output[{output_dim}]) {{
        for (int i=0; i<{output_dim}; i++) _output[i] = 0.0;//((i^13+15 + int(random()*17))%{output_dim})/float({output_dim});
        if (dynamic_map == 0) {{
            return;
        }}
        int map_id = int_ptr(dynamic_map).data[0];
        switch(map_id)
        {{
        {fw_cases}
        }}  
    }}

    void dynamic_backward(MAP_DECL, GPUPtr dynamic_map, in float _input[{input_dim}]{output_parameter}, in float _output_grad[{output_dim}]{input_grad_parameter})  {{
        if (dynamic_map == 0) return;
        int map_id = int_ptr(dynamic_map).data[0];
        switch(map_id)
        {{
        {bw_cases}
        }}  
    }}
            """

    @classmethod
    def generate_single_map_kernel(cls, map: 'Map', kernel_id: int):
        """
        Generates the kernel of a specific map.
        """
        codename = cls.__CODENAMES__[kernel_id]
        code = ""
        map_object_parameters_code, external_structs, _ = cls.create_code_type_definition(map.rdv_type_definition,
                                                                                              map.rdv_parameters,
                                                                                              is_first_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                # code += struct_code + "\n"  # only save in defined structs
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"#define RDV_CODENAME {codename}"
        code += f"""
    layout(buffer_reference, scalar, buffer_reference_align=8) buffer MAP_BUFFER_NAME {{{map_object_parameters_code}}};
    struct RDV_CODENAME {{ MAP_BUFFER_NAME data; }};
    """
        for g, v in map.rdv_generics.items():
            code += f"#define {g} {v} \n"
        code += f"#define MAP_DECL in RDV_CODENAME _this \n"
        code += f"#define parameters _this.data \n"
        code += f"#include \"./signatures.h\"\n"

        for s in map.rdv_dynamic_requires:  # Generate dynamic access code for all required signatures
            code += cls.create_code_for_dynamic_calls(map, *s)

        code += map.rdv_source_code + "\n"

        code += f"#undef map_object\n"
        code += f"#undef RDV_CODENAME\n"
        code += f"#undef parameters\n"
        for g in map.rdv_generics:
            code += f"#undef {g}\n"
        return code

    @classmethod
    def generate_compute_kernel(cls, compute: 'Compute', task: 'ComputeTask'):
        """
        Generates the kernel of a specific compute and task.
        """
        code = ""
        compute_parameters_code, external_structs, _ = cls.create_code_type_definition(compute.rdv_type_definition,
                                                                                          task.binder,
                                                                                          is_first_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                # code += struct_code + "\n"  # only save in defined structs
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"""
            layout(binding=3, scalar) uniform ComputeParameters {{{compute_parameters_code}}} parameters;
            """
        for g, v in task.rdv_generics.items():
            code += f"#define {g} {v} \n"

        code += compute.rdv_source_code + "\n"
        for g in compute.rdv_generics:
            code += f"#undef {g}\n"
        return code

    @classmethod
    def register_instance(cls, map: 'Map'):  # sets a new or existing compute id for the map
        """
        Registers a map if new signature.
        Returns the id of the map and the code name
        """
        signature = map.rdv_signature
        compute_id = cls.__COMPUTE_ID_BY_SIGNATURE__.get(signature)
        if compute_id is None:
            cls.__COMPUTE_IDS__ += 1
            compute_id = cls.__COMPUTE_IDS__
            cls.__COMPUTE_ID_BY_SIGNATURE__[signature] = compute_id
            cls.__CODENAMES__[compute_id] = f"{(type(map).__name__).replace('_', '')}_{compute_id}"  # 'rdv_map_' + str(instance_id)
            cls.__DIMENSIONS__[compute_id] = (map.input_dim, map.output_dim)
            cls.__KERNELS__[compute_id] = cls.generate_single_map_kernel(map, compute_id)
            cls.__INCLUDE_DIRS__[compute_id] = map.rdv_include_dirs
        return compute_id

    @classmethod
    def create_code_type_definition(cls, type_definition, field_value = None, is_first_block = False):
        if type_definition == Map:
            assert field_value is not None, "Basic structs can not bind explicit maps. Use int64_t if you want a reference."
            return cls.__CODENAMES__[field_value.rdv_kernel_id], {}, []
        if type_definition == _torch.Tensor:
            return 'GPUPtr', {}, []
        if _vk.Layout.is_scalar_type(type_definition):
            if type_definition == int:
                return 'int', {}, []
            if type_definition == float:
                return 'float', {}, []
            return {
                _torch.int32: 'int',
                _torch.float32: 'float',
                _torch.int64: 'GPUPtr'
            }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0]
            t = type_definition[1]
            t_value = None if field_value is None else field_value[0]
            element_decl, inner_structures, element_sizes = cls.create_code_type_definition(t, t_value)
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            inner_structures = {}
            if '__name__' in type_definition.keys():  # external struct
                struct_code = f"struct {type_definition['__name__']} {{"
                for field_id, field_type in type_definition.items():
                    if field_id != '__name__':
                        t, field_inner_structures, sizes = cls.create_code_type_definition(field_type)
                        struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                        inner_structures.update(field_inner_structures)
                struct_code += '};'
                inner_structures[type_definition['__name__']] = struct_code
                return type_definition['__name__'], inner_structures, []
            else:  # block
                assert is_first_block, 'Can not create a nested block. Add a name attribute to the dictionary to make it a struct'
                code = ""
                for field_id, field_type in type_definition.items():
                    f_value = None if field_value is None else getattr(field_value, field_id)
                    t, field_inner_structures, sizes = cls.create_code_type_definition(field_type, f_value)
                    code += t + " " + field_id + ''.join(f"[{size if size > 0 else ''}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
                return code, inner_structures, []
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def create_support_code(cls):
        # Gets vulkan device used
        caps = _vk.support()
        code = ""
        if caps.ray_query:
            code += "#define SUPPORTED_RAY_QUERY\n"
        if caps.atom_float:
            code += "#define SUPPORTED_FLOAT_ATOM_ADD\n"
        return code

    @classmethod
    def dispatch(cls, instance: 'Compute', task: 'ComputeTask', **deferred):
        static_signature = (task.rdv_signature, task.rdv_group_size)
        pipeline, ds = cls.__EVAL_PIPELINES__.get(static_signature, (None, None))
        if pipeline is None: # Create Pipeline if no exist
            pipeline = _vk.pipeline_compute()
            kernel_codes, kernel_dirs = cls.generate_map_kernels(task.binder._compute_ids())
            compute_code = cls.generate_compute_kernel(instance, task)
            code = f"""
#version 460
#extension GL_GOOGLE_include_directive : require
{
cls.create_support_code()
}
#define LOCAL_SIZE_X {task.rdv_group_size[0]}
#define LOCAL_SIZE_Y {task.rdv_group_size[1]}
#define LOCAL_SIZE_Z {task.rdv_group_size[2]}
            """ + __COMPUTE_TEMPLATE__ + kernel_codes + compute_code
            pipeline.load_shader_from_source(code, include_dirs=set([__INCLUDE_PATH__]+instance.rdv_include_dirs + list(kernel_dirs)))
            pipeline.layout(set=0, binding=0, system_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.layout(set=0, binding=1, deferred_buffer=_vk.DescriptorType.STORAGE_BUFFER)
            pipeline.layout(set=0, binding=2, named_tensors_buffer=_vk.DescriptorType.STORAGE_BUFFER)
            pipeline.layout(set=0, binding=3, parameters_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.close()
            ds = pipeline.create_descriptor_set_collection(0, 1)
            # bind system buffer and parameters buffer
            ds[0].update(
                system_buffer=cls.__SYSTEM_BUFFER__,
                deferred_buffer=_DeferredParametersManager.rdv_deferred_parameters_buffer,
                named_tensors_buffer=_DeferredParametersManager.rdv_named_tensors_buffer,
                parameters_buffer=task.rdv_buffer
            )
            cls.__EVAL_PIPELINES__[static_signature] = (pipeline, ds)
        # create manager if no exist
        dispatch_signature = (static_signature, task.rdv_threads)
        manager = cls.__EVAL_MANAGERS__.get(dispatch_signature)
        if manager is None:  # Create manager
            manager = _vk.compute_manager()
            manager.set_pipeline(pipeline)
            manager.bind(ds[0])
            manager.dispatch_threads(*task.rdv_threads, *task.rdv_group_size)
            manager.freeze()
            cls.__EVAL_MANAGERS__[dispatch_signature] = manager
        # update system fields
        # s = _generate_seeds()
        import numpy as np
        # r = _torch.randint(129, 1<<31, size=(4,))
        if len(cls.__RANDOMS__) == 0:
            cls.__RANDOMS__ = list(_torch.randint(129, 1 << 30, size=(1024*32, 4)))
        r = cls.__RANDOMS__.pop()
        t = task.rdv_threads
        with cls.__SYSTEM_BUFFER__ as b:
            b.seeds_x = r[0]
            b.seeds_y = r[1]
            b.seeds_z = r[2]
            b.seeds_w = r[3]
            b.dim_x = t[0]
            b.dim_y = t[1]
            b.dim_z = t[2]
        # _DeferredParametersManager.bind(deferred)
        _vk.submit(manager)

    @classmethod
    def start_session(cls):
        # Define system buffers
        cls.__SYSTEM_BUFFER__ = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                    seeds_x=int,
                                                                                    seeds_y=int,
                                                                                    seeds_z=int,
                                                                                    seeds_w=int,
                                                                                    dim_x=int,
                                                                                    dim_y=int,
                                                                                    dim_z=int
                                                                                ), memory=_vk.MemoryLocation.CPU)
        #initialize deferred buffers
        _DeferredParametersManager.init()
        # Defined structs in common.h
        _, inner_structs, _ = cls.create_code_type_definition(DeferrableField)
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        _, inner_structs, _ = cls.create_code_type_definition(MeshInfo)
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        _, inner_structs, _ = cls.create_code_type_definition(RaycastableInfo)
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        # Add to builtin array to check and dont redefine
        cls.__BUILTIN_STRUCTS__.extend(cls.__DEFINED_STRUCTS__)


def _start_session():
    try:
        __devices = _os.environ['CUDA_VISIBLE_DEVICES'].split(',')  # = str(rdv_device)
        rdv_device = int(__devices[0])
    except:
        rdv_device = 0
    debug = bool(_os.environ.get('RDV_DEBUG', 'False') == 'True')
    _vk.create_device(device=rdv_device, debug=debug)

    if _torch.cuda.is_available():
        _torch.cuda.init()
    _DispatcherEngine.start_session()


class _ComputeMeta(type):
    __COMPUTE_TYPE_COUNTER__ = 0
    """
    Incremental Id for compute types
    """
    __COMPUTE_DYNAMICS_COUNTER__ = 0
    """
    Incremental Id for dynamic computes.
    """

    def __new__(cls, name, bases, dct):
        # Compute type creation
        compute_type = super().__new__(cls, name, bases, dct)
        # Check __extension_info__
        assert '__extension_info__' in dct, 'Derived computes requires a dict __extension_info__ with path or code, parameters, [opt] generics, [opt] include_dirs'
        extension_info = dct['__extension_info__']
        if extension_info is not None:  # is not an abstract node
            extension_path = extension_info.get('path', None)
            extension_code = extension_info.get('code', None)
            extension_generics = extension_info.get('generics', {})
            parameters = extension_info.get('parameters', {})
            assert (extension_path is None or isinstance(extension_path, str) and _os.path.isfile(
                extension_path)), 'path must be a valid file path str'
            include_dirs = extension_info.get('include_dirs', [])
            assert (extension_path is None) != (extension_code is None), 'Either path or code must be provided'
            if extension_path is not None:
                include_dirs.append(_os.path.dirname(extension_path))
                extension_code = f"#include \"{_os.path.basename(extension_path)}\"\n"
            extension_dynamic_requires = extension_info.get('dynamics',
                                                            [])  # List with list of map signatures that can be dispatched dynamically by this map
            if len(extension_dynamic_requires) == 0:
                compute_type.rdv_generics = extension_generics
            else:
                _ComputeMeta.__COMPUTE_DYNAMICS_COUNTER__ += 1
                compute_type.rdv_generics = { **extension_generics, 'RDV_DYNAMIC_ID': _MapMeta.__COMPUTE_DYNAMICS_COUNTER__}
            compute_type.rdv_dynamic_requires = extension_dynamic_requires
            compute_object = {'rdv_kernel_id': int, 'rdv_map_pad0': int, 'rdv_map_pad1': int, 'rdv_map_pad2': int,
                          **parameters}
            def from_type_2_layout_description(p, dynamic_array_size=0, **generics):
                if p == Map:
                    return _torch.int64
                if p == _torch.Tensor:
                    return _torch.int64
                if isinstance(p, list):
                    if isinstance(p[0], str):
                        array_size = generics[p[0]]  # get the size from generics
                    else:
                        array_size = p[0] if p[0] > 0 else dynamic_array_size
                    return [array_size, from_type_2_layout_description(p[1], dynamic_array_size, **generics)]
                if isinstance(p, dict):
                    return {'__name__': p.get('__name__'),
                            **{k: from_type_2_layout_description(v, dynamic_array_size, **generics) for k, v in p.items() if
                               k != '__name__'}}
                return p
            compute_object_layout_builder = lambda s, g: _vk.Layout.from_description(
                _vk.LayoutAlignment.SCALAR,
                description=from_type_2_layout_description(compute_object, s, **g)
            )
            compute_type.rdv_layout_builder = compute_object_layout_builder
            compute_type.rdv_type_definition = compute_object
            compute_type.rdv_source_code = extension_code
            compute_type.rdv_include_dirs = include_dirs
            cls.__COMPUTE_TYPE_COUNTER__ += 1
            compute_type.rdv_type_id = cls.__COMPUTE_TYPE_COUNTER__
        return compute_type


class _MapMeta(_ComputeMeta):
    def __call__(self, *args, **kwargs):
        # map instantiation
        map_instance: Map = super(_MapMeta, self).__call__(*args, **kwargs)
        if not map_instance.is_generic:
            assert all(not m.is_generic for m in map_instance.children), f'A non-generic map {type(map_instance)} can not contains generic submaps'
            compute_id = _DispatcherEngine.register_instance(map_instance)
            map_instance.rdv_kernel_id = compute_id
            map_instance.rdv_buffer_accessor.rdv_kernel_id = compute_id  # set the id to the gpu.
        return map_instance


class MapElement:
    def __init__(self,  type_definition, accessor):
        object.__setattr__(self, '_type_definition', type_definition)
        object.__setattr__(self, '_accessor', accessor)

    def _compute_ids(self) -> _typing.Iterable[int]:
        pass

    def _references(self) -> _typing.Iterable['Map']:
        pass


class MapArray(MapElement):
    def __init__(self, type_definition, accessor):
        super().__init__(type_definition, accessor)
        object.__setattr__(self, '_element_definition', type_definition[1])
        object.__setattr__(self, '_backend_array', [None] * type_definition[0])

    def _compute_ids(self):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        if element_definition == Map:
            first_map: _typing.Optional[Map] = backend_array[0]
            assert first_map is not None and all(m.rdv_kernel_id == first_map.rdv_kernel_id for m in backend_array)
            return (first_map.rdv_kernel_id,)
        if isinstance(element_definition, dict) or isinstance(element_definition, list):
            first_element : _typing.Optional[MapElement] = backend_array[0]
            cids = first_element._compute_ids()
            assert first_element is not None and all(m._compute_ids() == cids for m in backend_array)
            return cids
        return ()

    def _references(self):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        if element_definition == Map:
            return set(backend_array)
        references = set()
        if isinstance(element_definition, dict) or isinstance(element_definition, list):
            for m in backend_array:
                references.update(m._references())
        return references

    def __len__(self):
        backend_array = object.__getattribute__(self, '_backend_array')
        return len(backend_array)

    def __getitem__(self, item):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        accessor = object.__getattribute__(self, '_accessor')
        assert item >= 0 and item < len(backend_array)
        if backend_array[item] is not None:
            return backend_array[item]
        if isinstance(element_definition, list):  # subarray
            value = MapArray(element_definition, accessor[item] if accessor else None)
        elif isinstance(element_definition, dict):  # substructure
            value = MapStruct(element_definition, accessor[item] if accessor else None)
        elif element_definition == Map or element_definition == _torch.int64:
            value = None
        else:
            value = element_definition()
        backend_array[item] = value
        return value

    def __setitem__(self, key, value):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        accessor = object.__getattribute__(self, '_accessor')
        assert key >= 0 and key < len(backend_array)
        assert not isinstance(element_definition, list) and not isinstance(element_definition, dict)

        if accessor is not None:
            accessor_value = value
            if element_definition == _torch.Tensor:
                if not isinstance(value, _vk.GPUPtr):
                    accessor_value = _vk.wrap_gpu(value, 'in')
            accessor[key] = accessor_value
        backend_array[key] = value


class MapStruct(MapElement):
    def __init__(self, type_definition, accessor):
        super().__init__(type_definition, accessor)

    def __getattr__(self, item):
        type_definition = object.__getattribute__(self, '_type_definition')
        accessor = object.__getattribute__(self, '_accessor')
        assert item in type_definition
        try:
            return super().__getattribute__(item)
        except:
            pass

        field_type = type_definition[item]
        if isinstance(field_type, dict):  # sub-structure
            field_value = MapStruct(field_type, getattr(accessor, item) if accessor else None)
        elif isinstance(field_type, list):
            field_value = MapArray(field_type, getattr(accessor, item) if accessor else None)
        elif field_type == Map or field_type == _torch.int64:
            field_value = None
        else:
            field_value = field_type()
        object.__setattr__(self, item, field_value)
        return field_value

    def __setattr__(self, key, value):
        type_definition = object.__getattribute__(self, '_type_definition')
        accessor = object.__getattribute__(self, '_accessor')
        assert key in type_definition
        field_definition = type_definition[key]
        is_deferrable = field_definition == DeferrableField
        assert not isinstance(field_definition, list), "Can not set directly a list, use per-index access"
        assert not isinstance(field_definition, dict) or is_deferrable, "Can not set directly an struct, use per-field access"
        if accessor is not None:
            if is_deferrable:  # Special case for deferred fields
                deferred_field = getattr(accessor, key)
                if isinstance(value, _torch.Tensor):  # Directly bound
                    deferred_field.data = _vk.wrap_gpu(value, 'in')
                    for i, d in enumerate(value.shape):
                        deferred_field.shape[i] = d
                else:
                    assert isinstance(value, deferred)
                    deferred_field.data = _vk.DirectGPUPtr.null()
                    deferred_field.deferred_index = value.id
            else:
                accessor_value = value
                if field_definition == _torch.Tensor:
                    if not isinstance(value, _vk.GPUPtr):
                        accessor_value = _vk.wrap_gpu(value, 'in')
                setattr(accessor, key, accessor_value)
        super().__setattr__(key, value)

    def _compute_ids(self):
        type_definition = object.__getattribute__(self, '_type_definition')
        cids = []
        for key in type_definition:
            field_type = type_definition[key]
            if field_type == DeferrableField:
                continue  # no maps here
            field_value = getattr(self, key)
            if field_type == Map:
                assert field_value is not None
                cids.append(field_value.rdv_kernel_id)
            elif isinstance(field_type, dict) or isinstance(field_type, list):
                cids.extend(field_value._compute_ids())
        return tuple(cids)

    def _references(self):
        type_definition = object.__getattribute__(self, '_type_definition')
        references = set()
        for key in type_definition:
            field_type = type_definition[key]
            if field_type == DeferrableField:
                continue  # no reference here
            field_value = getattr(self, key)
            if field_type == Map or field_type == _torch.int64 and isinstance(field_value, Map):
                assert field_value is not None
                references.add(field_value)
            elif isinstance(field_type, dict) or isinstance(field_type, list):
                references.update(field_value._references())
        return references


class Map(object, metaclass=_MapMeta):
    """
    Base class for all maps.
    A map defines a transform from R^n to R^m.
    Forward and backward operations are solved with a compute shader.
    """

    __extension_info__ = None  # Represent abstract nodes with __extension_info__ None
    rdv_include_dirs = []
    rdv_generics = {}
    """
    Gets the set of generics used by this map. All generics are turn into defines in the code that are
    only valid within the map implementation.
    """
    rdv_dynamic_requires = {}
    """
    Gets the dynamic signatures (input_dim, output_dim) required by this map dynamic accesses.
    """
    rdv_type_id = 0
    """
    Each Map derived class has a unique type_id.
    """
    rdv_layout_builder = None
    """
    A function that receives a final count and retrieves the layout for the object buffer creation.
    """
    rdv_buffer = None
    """
    Vulky uniform buffer with the struct defining the parameters of the map. 
    """
    rdv_buffer_accessor = None
    """
    Object Buffer Accessor to object header and all map parameters.
    """
    rdv_parameters = None
    """
    MapStruct with buffer accessor to the defining the parameters of the map. 
    """
    rdv_type_definition = None
    """
    Dictionary with the definition of parameters of the map.
    """
    rdv_source_code = None
    """
    Code for the specific map kernel.
    """
    rdv_frozen = False
    """
    Once the map is initialized it is frozen to new updates of the parameters.
    """
    def __init__(self,
                 dynamic_length=0,
                 input_dim=None,
                 output_dim=None,
                 input_requires_grad=False,
                 bw_uses_output=False,
                 **generics):
        input_requires_grad_generic = { 'INPUT_REQUIRES_GRAD': 1 } if input_requires_grad else {}
        bw_uses_output_generic = { 'BW_USES_OUTPUT': 1 } if bw_uses_output else {}
        self.rdv_generics = {**self.rdv_generics, **{k: v for k,v in generics.items() if v is not None}, **input_requires_grad_generic, **bw_uses_output_generic, 'INPUT_DIM': input_dim, 'OUTPUT_DIM': output_dim}
        if not self.is_generic:
            layout = type(self).rdv_layout_builder(dynamic_length, self.rdv_generics)
            buffer = _vk.object_buffer(layout).clear()
            object.__setattr__(self, 'rdv_buffer', buffer)
            object.__setattr__(self, 'rdv_buffer_accessor', buffer.accessor)
            object.__setattr__(self, 'rdv_parameters', MapStruct(self.rdv_type_definition, buffer.accessor))

    @cached_property
    def input_requires_grad(self) -> bool:
        return 'INPUT_REQUIRES_GRAD' in self.rdv_generics

    @cached_property
    def bw_uses_output(self) -> bool:
        return 'BW_USES_OUTPUT' in self.rdv_generics

    def clone(self,
              input_dim,
              output_dim,
              input_requires_grad,
              bw_uses_output) -> 'Map':
        raise NotImplementedError()

    def cast(self,
             input_dim=None,
             output_dim=None,
             input_requires_grad=None,
             bw_uses_output=None) -> 'Map':
        changed = False
        promoting = None
        if input_dim is None:
            input_dim = self.input_dim
        if self.input_dim != input_dim:
            assert self.input_dim is None
            changed |= True
        if output_dim is None:
            output_dim = self.output_dim
        if output_dim != self.output_dim:
            assert self.output_dim is None or self.output_dim == 1
            if self.output_dim == 1:
                promoting = output_dim
                output_dim = 1  # keeps output dim 1 but perform a promote
                changed |= False
            else:
                changed |= True
        if input_requires_grad is None:
            input_requires_grad = self.input_requires_grad
        changed |= input_requires_grad != self.input_requires_grad
        if bw_uses_output is None:
            bw_uses_output = self.bw_uses_output
        changed |= bw_uses_output != self.bw_uses_output
        s = self
        if changed:
            s = self.clone(input_dim, output_dim, input_requires_grad, bw_uses_output)
        if promoting is None:
            return s
        return s.promote(promoting)

    @cached_property
    def device_ptr(self):
        return self.rdv_buffer.device_ptr

    @cached_property
    def _compute_ids(self):
        return self.rdv_parameters._compute_ids()

    @cached_property
    def rdv_signature(self):
        assert not self.is_generic, "Signatures represent uniquely a computation unit. Generic maps do not have signatures."
        return (self.rdv_type_id, frozenset(self.rdv_generics.items()), self._compute_ids)

    def __getattr__(self, item):
        if item in self.rdv_type_definition:
            return getattr(self.rdv_parameters, item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        if key in self.rdv_type_definition:
            assert not self.rdv_frozen, "Parameters of a map can only be set during init."
            return setattr(self.rdv_parameters, key, value)
        super().__setattr__(key, value)

    @cached_property
    def input_dim(self):
        """
        Gets the dimension of the input vector.
        If None, the map is generic in the input and should be cast to have a non-generic map.
        """
        return self.rdv_generics.get('INPUT_DIM')

    @cached_property
    def output_dim(self):
        """
        Gets the dimension of the output vector.
        If None, the map is generic in the output and should be cast to have a non-generic map.
        """
        return self.rdv_generics.get('OUTPUT_DIM')

    @cached_property
    def is_generic(self):
        return self.input_dim is None or self.output_dim is None

    @cached_property
    def is_generic_input(self):
        return self.input_dim is None

    @cached_property
    def is_generic_output(self):
        return self.output_dim is None

    @cached_property
    def is_dynamic(self):
        return len(self.rdv_dynamic_requires) > 0

    @cached_property
    def dependences(self) -> _typing.Set[int]:
        """
        Returns this map's dependencies compute ids as a set.
        """
        return set(self._compute_ids)

    @cached_property
    def children(self) -> _typing.Iterable['Map']:
        """
        Returns an iterable for all accessible direct submaps.
        """
        return self.rdv_parameters._references()

    def __call__(self, *args, **kwargs):
        assert not self.is_generic, "Cast the map to a specific dimensions before eval."
        input, = args
        assert not input.requires_grad or self.input_requires_grad, "Cast the map to input_requires_grad=True before eval an input with grad req."
        return _MapEvalFunction.apply(self, input, kwargs.keys(), *kwargs.values())

    def backward(self,
                 input: _torch.Tensor,
                 output_grad: _torch.Tensor,
                 parameters: _typing.Dict[str, _typing.Any],
                 parameters_grad: _typing.Dict[str, _typing.Any]):
        pass


class ComputeTask:
    def __init__(self, signature: tuple, buffer: _vk.ObjectBuffer, map_object: MapStruct, threads: tuple, group_size: tuple = (32, 1, 1), **generics):
        self.rdv_signature = signature
        self.rdv_threads = threads
        self.rdv_group_size = group_size
        self.rdv_buffer = buffer
        self.rdv_object = map_object
        self.rdv_generics = generics

    def save(self, *values):
        self.saved_values = values

    @cached_property
    def binder(self) -> MapStruct:
        return self.rdv_object


class Compute(object, metaclass=_ComputeMeta):
    """
    Base class for compute-based tensor operations in rendervous.
    Derived classes must define two stages: bind and result.
    __extension_info__ must provide parameters: dict and code|path with a MAIN(tid) { } method.
    generics in extension info can be used to define sizes and parameters needed as compile-time constant in the code.
    """
    __extension_info__ = None  # Abstract node
    __instance__ = None
    __OBJECTS__ = { }  # from signature to binders

    @classmethod
    def instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = cls()
        return cls.__instance__

    # def __getattr__(self, item):
    #     accessor = super().__getattribute__('rdv_parameters_buffer_accessor')
    #     if item in accessor._rdv_layout.fields_layout:
    #         return getattr(accessor, item)
    #     return super().__getattribute__(item)
    #
    # def __setattr__(self, key, value):
    #     accessor = super().__getattribute__('rdv_parameters_buffer_accessor')
    #     setattr(accessor, key, value)
    #     super().__setattr__(key, value)

    @classmethod
    def eval(cls, *args, deferred_parameters: _typing.Optional[_typing.Dict[str, _torch.Tensor]] = None, **kwargs):
        if deferred_parameters is not None:
            grads = _DeferredParametersManager.bind(deferred_parameters)
        else:
            grads = {}
        instance = cls.instance()
        compute_task = instance.bind(*args, **kwargs)
        _DispatcherEngine.dispatch(instance, compute_task)
        r = instance.result(compute_task)
        if deferred_parameters is not None:
            _DeferredParametersManager.unbind(deferred_parameters)
            return r, grads
        return r

    @classmethod
    def create_task(cls, threads: _typing.Union[int, tuple], *maps: Map, dynamic_size=0, group_size: tuple = (1024, 1, 1), **generics) -> ComputeTask:
        generics = cls.rdv_generics if len(generics) == 0 else { **cls.rdv_generics, **generics }
        if isinstance(threads, int):
            threads = (threads, 1, 1)
        elif len(threads) == 2:
            threads = (threads[0], threads[1], 1)
        else:
            assert len(threads) == 3
        signature = (cls.rdv_type_id, *(m.rdv_kernel_id if m else 0 for m in maps), frozenset(generics.items()))
        obj, map_struct = cls.__OBJECTS__.get((signature, dynamic_size), (None, None))
        if obj is None:
            layout = cls.rdv_layout_builder(dynamic_size, generics)
            obj = _vk.object_buffer(layout, memory=_vk.MemoryLocation.CPU)
            map_struct = MapStruct(cls.rdv_type_definition, obj.accessor)
            cls.__OBJECTS__[(signature, dynamic_size)] = (obj, map_struct)
        return ComputeTask(signature, obj, map_struct, threads, group_size, **generics)

    def bind(self, *args, **kwargs) -> 'ComputeTask':
        '''
        sets the arguments to the object and return the number of threads to dispatch
        '''
        raise NotImplementedError()

    def result(self, compute_task) -> _typing.Any:
        '''
        :param parameters: accessor to get bound tensors
        :return: resultant output, can be directly tensors or a postprocessing on them.
        '''
        raise NotImplementedError()


class _MapForwardEvalCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/map_forward_eval.h',
        parameters=dict(
            input_tensor=_torch.Tensor,
            output_tensor=_torch.Tensor,
            map=Map
        )
    )

    def bind(self, *args, **kwargs) -> ComputeTask:
        map, input = args
        assert not map.is_generic
        input_dim = input.shape[-1]
        output_dim = map.output_dim
        output = _vk.tensor(*input.shape[:-1], output_dim)
        assert map.input_dim == input_dim
        task = _MapForwardEvalCompute.create_task(input.numel() // input_dim, map, MAP_INPUT_DIM=input_dim, MAP_OUTPUT_DIM=output_dim)
        binder = task.binder
        binder.input_tensor = _vk.wrap_gpu(input, 'in')
        binder.output_tensor = _vk.wrap_gpu(output, 'out')
        binder.map = map
        return task

    def result(self, compute_task: ComputeTask) -> _typing.Any:
        compute_task.binder.input_tensor.unwrap()
        return compute_task.binder.output_tensor.unwrap()


class _MapBackwardEvalCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/map_backward_eval.h',
        parameters=dict(
            input_tensor=_torch.Tensor,
            output_grad_tensor=_torch.Tensor,
            input_grad_tensor=_torch.Tensor,
            map=Map
        )
    )

    def bind(self, *args, **kwargs) -> ComputeTask:
        map, input, output_grad = args
        assert not map.is_generic
        input_dim = input.shape[-1]
        output_dim = output_grad.shape[-1]
        input_grad = _vk.tensor(*input.shape[:-1], input_dim)
        assert map.input_dim == input_dim and map.output_dim == map.output_dim
        task = _MapBackwardEvalCompute.create_task(input.numel() // input_dim, map, MAP_INPUT_DIM=input_dim, MAP_OUTPUT_DIM=output_dim)
        binder = task.binder
        binder.input_tensor = _vk.wrap_gpu(input, 'in')
        binder.output_grad_tensor = _vk.wrap_gpu(output_grad, 'in')
        binder.input_grad_tensor = _vk.wrap_gpu(input_grad, 'out')
        binder.map = map
        return task

    def result(self, compute_task: ComputeTask) -> _typing.Any:
        compute_task.binder.input_tensor.unwrap()
        compute_task.binder.output_grad_tensor.unwrap()
        return compute_task.binder.input_grad_tensor.unwrap()


class _MapEvalFunction(_torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        map, input, names, *deferred_tensors = args
        ctx.map = map
        ctx.names = names
        ctx.save_for_backward(input, *deferred_tensors)
        output, _ = _MapForwardEvalCompute.eval(map, input, deferred_parameters={n: t for n, t in zip(names, deferred_tensors)})
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        output_grad, = grad_outputs
        input, *deferred_tensors = ctx.saved_tensors
        map = ctx.map
        names = ctx.names
        input_grad, grads = _MapBackwardEvalCompute.eval(map, input, output_grad, deferred_parameters={n: t for n, t in zip(names, deferred_tensors)})
        return (
            None, # map
            input_grad,
            None, # names
            *(grads.get(k) for k in names)
        )

