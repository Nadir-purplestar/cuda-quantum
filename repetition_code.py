"""17-qubit, 60-round repetition circuit without between-round resets.

This circuit was run on our configured CUDA-Q/LQCloud system. Running it
elsewhere requires a CUDA-Q build containing the LQCloud backend and these
environment variables:

    LQCLOUD_API_KEY  API key read directly by the backend
    LQCLOUD_URL      LQCloud service URL
    LQCLOUD_QPU      Available exact device name

The device must support at least 17 qubits and mid-circuit Z readout. The
recorded repetition-code runs used QZ01-repetition_code with measurement
type "measure". Initial reset is supplied by the backend; the circuit does
not reset qubits between rounds. All 489 measurement events are retained.

LQCLOUD_SHOTS defaults to 4096 and accepts a positive shot-count override.
The program prints actual CUDA-Q counts once. It does not run a decoder or
estimate error-correction performance.
"""

import os

import cudaq


def build_kernel(initial_state='000000000'):
    import cudaq
    if len(initial_state) != 9 or set(initial_state) - {'0', '1'}:
        raise ValueError('Expected nine binary initial-state digits')
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(17)
    data, ancilla = list(range(0, 17, 2)), list(range(1, 17, 2))
    for index, bit in zip(data, initial_state):
        if bit == '1':
            kernel.x(q[index])
    for round_index in range(60):
        kernel.barrier(q)
        for index in ancilla:
            kernel.h(q[index])
        kernel.barrier(q)
        for anc, dat in zip(ancilla, data[:-1]):
            kernel.cz(q[anc], q[dat])
        kernel.barrier(q)
        for anc, dat in zip(ancilla, data[1:]):
            kernel.cz(q[anc], q[dat])
        kernel.barrier(q)
        for index in ancilla:
            kernel.h(q[index])
        for index in data:
            kernel.y(q[index])
        kernel.barrier(q)
        for index in ancilla:
            kernel.mz(q[index])
        if round_index == 59:
            for index in data:
                kernel.mz(q[index])
    return kernel


def main():
    shots = int(os.environ.get("LQCLOUD_SHOTS", "4096"))
    if shots <= 0:
        raise ValueError("LQCLOUD_SHOTS must be a positive integer")
    cudaq.set_target("lqcloud",
                     url=os.environ["LQCLOUD_URL"],
                     qpu=os.environ["LQCLOUD_QPU"],
                     measurement_type="measure")
    result = cudaq.sample_async(build_kernel(),
                                shots_count=shots,
                                explicit_measurements=True).get()
    print(result)


if __name__ == "__main__":
    main()
