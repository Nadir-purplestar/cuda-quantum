# CUDA-Q / LQCloud hardware results

PASS denotes successful CUDA-Q submission and result retrieval with matching shot totals and measurement-record widths.

## Hardware measurements

| Circuit | Device | Shots | Observed result | Integration check |
|---|---|---:|---|---|
| Two-qubit Bell | MQ02 | 4096 | Counts: `00:2085, 01:30, 10:68, 11:1913`. Correlated-outcome fraction: 3998/4096 (97.6074%). | PASS |
| Three-qubit GHZ | MQ02 | 4096 | Counts: `000:2058, 001:55, 010:5, 011:29, 100:88, 101:61, 110:65, 111:1735`. Correlated-outcome fraction: 3793/4096 (92.6025%). | PASS |
| Nonadjacent CNOT and out-of-order measurements | MQ02 | 4096 | Counts: `000:92, 001:3716, 010:1, 011:55, 100:92, 101:93, 110:46, 111:1`. The circuit used CNOT(0,2) and measurement order [2,0,1]; logical outcome `001` occurred 3716 times. | PASS |
| Terminal partial measurement: three qubits, two measured | MQ02 | 4096 | Counts: `00:102, 01:3989, 11:5`. Only qubits 2 and 0 were measured; logical outcome `01` occurred 3989 times. | PASS |
| Ry/Rz/H interference with same-process asynchronous retrieval | MQ02 | 4096 | Counts: `00:65, 01:3976, 10:1, 11:54`. The circuit used fixed pi rotations and `sample_async(...).get()`; logical outcome `01` occurred 3976 times. | PASS |
| Three-qubit supported-gate circuit | MQ02 | 1000 | Counts: `000:27, 001:28, 010:579, 011:18, 100:36, 101:251, 110:55, 111:6`. Gates included H/X/Y/Z/S/T, rotations, U3, CX/CY/CZ and SWAP. | PASS |
| 17-qubit, 60-round repetition circuit | QZ01-repetition_code | 4096 | Native asynchronous execution returned all 4096 shots as 489-bit measurement records. | PASS |

The repetition circuit starts nine data qubits in zero and uses eight ancillas. It performs 60 rounds with synchronization barriers and no between-round resets. Initial resets are supplied by the backend. Eight ancilla measurements per round plus nine final data measurements produce 489 events in source order. The supplied program returns the measurement counts without running a decoder.

## RY/CX parameter sweep

Each angle was bound at runtime in a separate 4096-shot CUDA-Q execution on MQ02. The circuit applies RY(theta) to qubit 0, CNOT(0,1), a barrier and terminal measurements.

| Angle theta | 00 | 01 | 10 | 11 | Integration check |
|---|---:|---:|---:|---:|---|
| 0 | 4007 | 40 | 36 | 13 | PASS |
| pi/4 | 3391 | 52 | 71 | 582 | PASS |
| pi/2 | 1947 | 57 | 181 | 1911 | PASS |
| 3pi/4 | 591 | 54 | 212 | 3239 | PASS |
| pi | 21 | 53 | 191 | 3831 | PASS |

## Partial readout from ten allocated qubits

Each row was submitted independently through CUDA-Q with 4096 shots on MQ02. The circuit allocates ten qubits, applies H(7), CNOT(7,6), RZ(theta,7) and a barrier, then measures only logical qubits 6 and 7. The table reports the returned two-bit Z-basis populations.

| Angle theta | 00 | 01 | 10 | 11 | Integration check |
|---|---:|---:|---:|---:|---|
| 0 | 1896 | 96 | 46 | 2058 | PASS |
| pi/4 | 1946 | 89 | 63 | 1998 | PASS |
| pi/2 | 1941 | 85 | 57 | 2013 | PASS |
| 3pi/4 | 1921 | 90 | 48 | 2037 | PASS |
| pi | 1900 | 96 | 54 | 2046 | PASS |

## Running the complete repetition example

`repetition_code.py` is self-contained CUDA-Q user code. Running it requires a CUDA-Q build containing the LQCloud backend, a device with at least 17 qubits and mid-circuit readout, and `LQCLOUD_API_KEY` in the environment. The program supplies the public defaults `https://cloud.logicalqubit.com` and `QZ01-repetition_code`; `LQCLOUD_URL` and `LQCLOUD_QPU` are optional overrides.

After setting `LQCLOUD_API_KEY` securely in the environment, run:

```bash
export LQCLOUD_URL="https://cloud.logicalqubit.com"
export LQCLOUD_QPU="QZ01-repetition_code"
python3 repetition_code.py
```

The default is 4096 shots. Set `LQCLOUD_SHOTS` to use another positive shot count. The program uses `run_async(...).get()` with measurement type `measure`, aggregates the genuine per-shot boolean records into counts, and prints those counts once.
