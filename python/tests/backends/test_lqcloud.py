# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from multiprocessing import Process
import os

import cudaq
import pytest
from network_utils import check_server_connection
from utils.mock_qpu.lqcloud import startServer, TEST_API_KEY

skipIfLQCloudNotInstalled = pytest.mark.skipif(
    not cudaq.has_target("lqcloud"),
    reason="Could not find lqcloud in installation")
pytestmark = skipIfLQCloudNotInstalled

PORT = 62452


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    previous_key = os.environ.get("LQCLOUD_API_KEY")
    os.environ["LQCLOUD_API_KEY"] = TEST_API_KEY
    server = Process(target=startServer, args=(PORT,))
    server.start()
    try:
        assert check_server_connection(PORT), "LQCloud mock did not start"
        cudaq.set_target("lqcloud",
                         url=f"http://localhost:{PORT}",
                         qpu="mock_qpu")
        yield
    finally:
        server.terminate()
        server.join()
        cudaq.reset_target()
        if previous_key is None:
            os.environ.pop("LQCLOUD_API_KEY", None)
        else:
            os.environ["LQCLOUD_API_KEY"] = previous_key


def test_lqcloud_sample():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    for result in (cudaq.sample(kernel, shots_count=400),
                   cudaq.sample_async(kernel, shots_count=400).get()):
        assert result.get_total_shots() == 400
        assert set(dict(result.items())) == {"00", "11"}
        assert all(140 <= count <= 260 for _, count in result.items())

    @cudaq.kernel
    def routed():
        q = cudaq.qvector(3)
        barrier(q)
        x(q[0])
        mz(q[0])
        # The chain requires routing 0--2; measuring q[1] keeps all three.
        barrier(q)
        z.ctrl(q[0], q[2])
        barrier(q)
        x(q[0])
        mz(q[0])
        x(q[2])
        mz(q[2])
        mz(q[1])

    for result in (cudaq.sample(routed,
                                shots_count=20,
                                explicit_measurements=True),
                   cudaq.sample_async(routed,
                                      shots_count=20,
                                      explicit_measurements=True).get()):
        assert dict(result.items()) == {"1010": 20}

    collapse = cudaq.make_kernel()
    q = collapse.qalloc(1)
    collapse.h(q[0])
    collapse.mz(q[0])
    collapse.h(q[0])
    collapse.mz(q[0])
    result = cudaq.sample(collapse, shots_count=400, explicit_measurements=True)
    assert result.get_total_shots() == 400
    assert set(dict(result.items())) == {"00", "01", "10", "11"}
    assert all(50 <= count <= 150 for _, count in result.items())

    long_history = cudaq.make_kernel()
    q = long_history.qalloc(1)
    long_history.x(q[0])
    for _ in range(489):
        long_history.mz(q[0])
    result = cudaq.sample_async(long_history,
                                shots_count=20,
                                explicit_measurements=True).get()
    assert dict(result.items()) == {"1" * 489: 20}

    with pytest.raises(ValueError, match="non-empty static qvector"):
        long_history.barrier(q[0])

    unmeasured = cudaq.make_kernel()
    q = unmeasured.qalloc(1)
    unmeasured.h(q[0])
    with pytest.raises(RuntimeError):
        cudaq.sample(unmeasured, shots_count=20, explicit_measurements=True)

    # A later default request must still use the normal terminal result order.
    result = cudaq.sample(kernel, shots_count=400)
    assert result.get_total_shots() == 400
    assert set(dict(result.items())) == {"00", "11"}
    assert all(140 <= count <= 260 for _, count in result.items())
