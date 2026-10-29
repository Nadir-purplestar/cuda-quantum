"""17-qubit, 60-round repetition experiment with chronological measurements."""
import json
import os
from collections import Counter

import cudaq


def build_kernel(initial_state="000000000"):
    if len(initial_state) != 9 or set(initial_state) - {"0", "1"}:
        raise ValueError("Expected nine binary initial-state digits")
    initial_bits = [bit == "1" for bit in initial_state]

    @cudaq.kernel
    def repetition() -> list[bool]:
        q = cudaq.qvector(17)
        record = [False for _ in range(489)]
        for index in range(9):
            if initial_bits[index]:
                x(q[2 * index])
        for round_index in range(60):
            barrier(q)
            for index in range(8):
                h(q[2 * index + 1])
            barrier(q)
            for index in range(8):
                z.ctrl(q[2 * index + 1], q[2 * index])
            barrier(q)
            for index in range(8):
                z.ctrl(q[2 * index + 1], q[2 * index + 2])
            barrier(q)
            for index in range(8):
                h(q[2 * index + 1])
            for index in range(9):
                y(q[2 * index])
            barrier(q)
            for index in range(8):
                record[8 * round_index + index] = mz(q[2 * index + 1])
            if round_index == 59:
                for index in range(9):
                    record[480 + index] = mz(q[2 * index])
        return record

    return repetition


def main():
    if not os.environ.get("LQCLOUD_API_KEY"):
        raise RuntimeError("Set LQCLOUD_API_KEY in the environment")
    shots = int(os.environ.get("LQCLOUD_SHOTS", "4096"))
    if shots <= 0:
        raise ValueError("LQCLOUD_SHOTS must be positive")
    cudaq.set_target(
        "lqcloud",
        url=os.environ.get("LQCLOUD_URL", "https://cloud.logicalqubit.com"),
        qpu=os.environ.get("LQCLOUD_QPU", "QZ01-repetition_code"),
        measurement_type="measure",
    )
    records = cudaq.run_async(build_kernel(), shots_count=shots).get()
    if (type(records) is not list or len(records) != shots
            or any(type(record) is not list or len(record) != 489
                   or any(type(bit) is not bool for bit in record)
                   for record in records)):
        raise RuntimeError("Expected 489 chronological booleans per shot")
    counts = Counter("".join("1" if bit else "0" for bit in record)
                     for record in records)
    print(json.dumps(dict(counts=counts), sort_keys=True))
    return records


if __name__ == "__main__":
    main()
