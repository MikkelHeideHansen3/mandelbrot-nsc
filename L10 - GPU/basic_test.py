import pyopencl as cl
import numpy as np

platforms = cl.get_platforms()

print("Platforms found:")
for p in platforms:
    print(" -", p.name)

ctx = cl.create_some_context(interactive=False)

# 👇 DET HER ER DET DU MANGLER
print("Devices found:")
for d in ctx.devices:
    print(" -", d.name)

queue = cl.CommandQueue(ctx)

program = cl.Program(ctx, """
__kernel void test(__global float *a) {
    int gid = get_global_id(0);
    a[gid] = 1.0f;
}
""").build()

a = np.zeros(10, dtype=np.float32)
buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a.nbytes)

program.test(queue, a.shape, None, buf)
cl.enqueue_copy(queue, a, buf)

print("Test kernel:", "PASSED" if np.all(a == 1.0) else "FAILED")