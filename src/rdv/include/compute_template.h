#define MAIN(tid) void rdv_compute(uvec3 tid)

void rdv_compute(uvec3 thread_id);

#include "core.h"

#ifndef LOCAL_SIZE_X
#define LOCAL_SIZE_X 1024
#endif

#ifndef LOCAL_SIZE_Y
#define LOCAL_SIZE_Y 1
#endif

#ifndef LOCAL_SIZE_Z
#define LOCAL_SIZE_Z 1
#endif

layout (local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

layout(binding = 0) uniform rdv_SystemInfo {
    uvec4 rdv_seeds;
    int rdv_dim_x;
    int rdv_dim_y;
    int rdv_dim_z;
};

void main()
{
    if (any(greaterThanEqual(gl_GlobalInvocationID, uvec3(rdv_dim_x, rdv_dim_y, rdv_dim_z))))
        return;
    int index = int(gl_GlobalInvocationID.x + (gl_GlobalInvocationID.z * rdv_dim_y + gl_GlobalInvocationID.y) * rdv_dim_x);
    random_seed(random_spawn(rdv_seeds, index));
    rdv_compute(gl_GlobalInvocationID);
}

