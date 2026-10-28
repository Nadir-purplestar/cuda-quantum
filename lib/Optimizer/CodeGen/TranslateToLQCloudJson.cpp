/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/Emitter.h"
#include "cudaq/Optimizer/CodeGen/LQCloudJsonEmitter.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

struct CircuitState {
  quake::AllocaOp allocation;
  // Number of qubits in the combined allocation. After the qubit-mapping
  // pass the circuit operates directly on physical lines (the combined
  // allocation is re-indexed accordingly), so instruction indices are
  // emitted as-is; the runtime restores the user's virtual ordering from
  // the mapping_reorder_idx attribute.
  std::size_t numberOfQubits = 0;
  std::set<std::size_t> measuredQubits;
  bool explicitMeasurements = false;
  std::size_t numberOfMeasurements = 0;
  std::set<std::size_t> measurementEvents;
  std::set<std::size_t> synchronizedQubits;
  bool resetPrefixOpen = true;
  std::vector<nlohmann::json> instructions;
};

nlohmann::json instruction(StringRef name, std::vector<std::size_t> qubits,
                           std::vector<std::size_t> clbits = {},
                           std::vector<double> parameters = {}) {
  return {{"name", name.str()},
          {"qubits", std::move(qubits)},
          {"clbits", std::move(clbits)},
          {"params", std::move(parameters)}};
}

std::optional<std::size_t> getQubitIndex(Value value, CircuitState &state) {
  auto extract = value.getDefiningOp<quake::ExtractRefOp>();
  if (!extract || extract.getVeq() != state.allocation.getRefOrVec())
    return std::nullopt;

  std::optional<int64_t> index;
  if (extract.hasConstantIndex())
    index = extract.getConstantIndex();
  else
    index = cudaq::getIndexValueAsInt(extract.getIndex());
  if (!index || *index < 0 ||
      static_cast<std::size_t>(*index) >= state.numberOfQubits)
    return std::nullopt;
  return static_cast<std::size_t>(*index);
}

std::optional<std::vector<std::size_t>>
getNativeQubitIndices(CircuitState &state, quake::OperatorInterface op,
                      StringRef gateName, std::size_t expectedControls,
                      std::size_t expectedTargets,
                      std::size_t expectedParameters) {
  if (op.getControls().size() != expectedControls ||
      op.getTargets().size() != expectedTargets ||
      op.getParameters().size() != expectedParameters) {
    op.emitOpError() << "LQCloud " << gateName << " expects "
                     << expectedControls << " control(s), " << expectedTargets
                     << " target(s), and " << expectedParameters
                     << " parameter(s)";
    return std::nullopt;
  }

  std::vector<std::size_t> indices;
  indices.reserve(expectedControls + expectedTargets);
  for (auto values : {op.getControls(), op.getTargets()}) {
    for (Value value : values) {
      auto index = getQubitIndex(value, state);
      if (!index) {
        op.emitOpError() << "LQCloud " << gateName
                         << " requires constant in-range qubit indices";
        return std::nullopt;
      }
      indices.push_back(*index);
    }
  }
  return indices;
}

LogicalResult emitReset(CircuitState &state, quake::ResetOp reset) {
  if (!state.explicitMeasurements && !state.measuredQubits.empty())
    return reset.emitOpError(
        "LQCloud resets after measurement require explicit measurements");
  auto index = getQubitIndex(reset.getTargets(), state);
  if (!index)
    return reset.emitOpError(
        "LQCloud reset requires a constant in-range qubit index");
  if (!state.resetPrefixOpen) {
    state.instructions.push_back(instruction("reset", {*index}));
    if (state.explicitMeasurements)
      state.synchronizedQubits.erase(*index);
    else
      state.synchronizedQubits.clear();
  }
  return success();
}

LogicalResult emitBarrier(CircuitState &state, quake::BarrierOp barrier) {
  std::vector<std::size_t> qubits;
  for (Value target : barrier.getTargets()) {
    auto index = getQubitIndex(target, state);
    if (!index)
      return barrier.emitOpError(
          "LQCloud barrier requires constant in-range qubit indices");
    qubits.push_back(*index);
  }
  std::sort(qubits.begin(), qubits.end());
  if (std::adjacent_find(qubits.begin(), qubits.end()) != qubits.end())
    return barrier.emitOpError("LQCloud barrier requires distinct qubits");

  // In particular, the first full barrier can also synchronize initial resets.
  // Never merge across a gate or measurement, or join disjoint target sets.
  if (state.instructions.back().at("name") != "barrier" ||
      state.instructions.back().at("qubits") != qubits)
    state.instructions.push_back(instruction("barrier", qubits));
  state.resetPrefixOpen = false;
  if (!state.explicitMeasurements)
    state.synchronizedQubits.clear();
  state.synchronizedQubits.insert(qubits.begin(), qubits.end());
  return success();
}

LogicalResult emitOperator(CircuitState &state, quake::OperatorInterface op) {
  if (!state.explicitMeasurements && !state.measuredQubits.empty())
    return op.emitOpError(
        "LQCloud does not support mid-circuit measurement or quantum "
        "operations after measurement");
  if (op.getNegatedControls())
    return op.emitOpError("LQCloud does not support negated controls");

  if (isa<quake::HOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "H", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("h", std::move(*qubits)));
  } else if (auto x = dyn_cast<quake::XOp>(op.getOperation());
             x && !x.isAdj()) {
    auto qubits = getNativeQubitIndices(state, op, "X", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("x", std::move(*qubits)));
  } else if (auto y = dyn_cast<quake::YOp>(op.getOperation());
             y && !y.isAdj()) {
    auto qubits = getNativeQubitIndices(state, op, "Y", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("y", std::move(*qubits)));
  } else if (auto rz = dyn_cast<quake::RzOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "Rz", 0, 1, 1);
    if (!qubits)
      return failure();
    auto angle = cudaq::getParameterValueAsDouble(rz.getParameter());
    if (!angle || !std::isfinite(*angle))
      return rz.emitOpError("LQCloud requires a finite constant Rz angle");
    if (rz.isAdj())
      *angle = -*angle;
    state.instructions.push_back(
        instruction("rz", std::move(*qubits), {}, {*angle}));
  } else if (auto z = dyn_cast<quake::ZOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "CZ", 1, 1, 0);
    if (!qubits)
      return failure();
    if ((*qubits)[0] == (*qubits)[1])
      return z.emitOpError(
          "LQCloud CZ control and target must be different qubits");
    state.instructions.push_back(instruction("cz", std::move(*qubits)));
  } else {
    return op.emitOpError()
           << "LQCloud does not support operation '" << op->getName() << "'";
  }

  state.resetPrefixOpen = false;
  if (state.explicitMeasurements) {
    for (auto values : {op.getControls(), op.getTargets()})
      for (Value value : values)
        state.synchronizedQubits.erase(*getQubitIndex(value, state));
  } else {
    state.synchronizedQubits.clear();
  }
  return success();
}

LogicalResult emitMeasurement(CircuitState &state, quake::MzOp measurement) {
  if (measurement.getTargets().size() != 1 ||
      !isa<quake::RefType>(measurement.getTargets().front().getType()) ||
      !isa<quake::MeasureType>(measurement.getMeasOut().getType()))
    return measurement.emitOpError("LQCloud requires scalar Z measurements");
  if (!state.explicitMeasurements && measurement.getRegisterName())
    return measurement.emitOpError(
        "LQCloud does not support named measurement registers");

  std::size_t event = 0;
  if (state.explicitMeasurements) {
    StringRef name = measurement.getRegisterName().value_or("");
    if (!name.consume_front("__lqcloud_measurement_") ||
        name.getAsInteger(10, event) || event >= state.numberOfMeasurements ||
        !state.measurementEvents.insert(event).second)
      return measurement.emitOpError(
          "LQCloud requires a unique index for every explicit measurement");
  }

  auto index = getQubitIndex(measurement.getTargets().front(), state);
  if (!index)
    return measurement.emitOpError(
        "LQCloud measurement requires a constant in-range qubit index");
  if (!state.measuredQubits.insert(*index).second &&
      !state.explicitMeasurements)
    return measurement.emitOpError(
        "LQCloud does not support repeated measurements");

  if (state.explicitMeasurements
          ? !state.synchronizedQubits.count(*index)
          : state.measuredQubits.size() == 1 &&
                state.synchronizedQubits.size() != state.numberOfQubits) {
    std::vector<std::size_t> allQubits(state.numberOfQubits);
    std::iota(allQubits.begin(), allQubits.end(), 0);
    state.instructions.push_back(instruction("barrier", allQubits));
    state.synchronizedQubits.insert(allQubits.begin(), allQubits.end());
  }
  state.resetPrefixOpen = false;
  state.synchronizedQubits.erase(*index);
  state.instructions.push_back(instruction(
      "measure", {*index}, {state.explicitMeasurements ? event : *index}));
  return success();
}

LogicalResult emitEntryPoint(nlohmann::json &json, func::FuncOp func) {
  if (func.getBody().getBlocks().size() != 1)
    return func.emitOpError("LQCloud requires a single straight-line block");
  if (func.getNumArguments() != 0 || func.getNumResults() != 0)
    return func.emitOpError(
        "LQCloud requires a fully specialized void entrypoint");
  for (NamedAttribute attribute : func->getAttrs()) {
    StringRef name = attribute.getName().strref();
    // mapping_v2p and mapping_reorder_idx are attached by the qubit-mapping
    // pass: the placement is consumed below and the runtime restores the
    // user's qubit ordering from the reorder indices. Any other mapping
    // attribute carries layout information we cannot interpret.
    if (name.starts_with("mapping") &&
        name != "mapping_v2p" && name != "mapping_reorder_idx")
      return func.emitOpError()
             << "LQCloud does not support mapping attribute '" << name << "'";
  }

  auto allocations = func.getOps<quake::AllocaOp>();
  if (!llvm::hasSingleElement(allocations))
    return func.emitOpError(
        "LQCloud requires exactly one combined quantum allocation");
  auto allocation = *allocations.begin();
  auto allocationType = dyn_cast<quake::VeqType>(allocation.getType());
  if (!allocationType || !allocationType.hasSpecifiedSize() ||
      allocationType.getSize() == 0)
    return allocation.emitOpError(
        "LQCloud requires a non-empty, statically sized quantum allocation");

  CircuitState state;
  state.allocation = allocation;
  state.numberOfQubits = allocationType.getSize();
  auto module = func->getParentOfType<ModuleOp>();
  state.explicitMeasurements = module->hasAttr("lqcloud.explicit_measurements");
  if (state.explicitMeasurements) {
    auto count =
        module->getAttrOfType<IntegerAttr>("lqcloud.measurement_count");
    if (!count || count.getInt() <= 0)
      return func.emitOpError(
          "LQCloud explicit measurement events were not prepared");
    state.numberOfMeasurements = count.getInt();
  }
  for (std::size_t index = 0; index < state.numberOfQubits; ++index)
    state.instructions.push_back(instruction("reset", {index}));
  std::vector<std::size_t> allQubits(state.numberOfQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  state.synchronizedQubits.insert(allQubits.begin(), allQubits.end());
  state.instructions.push_back(instruction("barrier", std::move(allQubits)));
  Block *functionBlock = &func.getBody().front();
  for (Operation &operation : *functionBlock) {
    if (isa<func::ReturnOp, quake::AllocaOp, quake::DiscriminateOp>(operation))
      continue;
    if (auto deallocation = dyn_cast<quake::DeallocOp>(operation)) {
      if (deallocation.getReference() != state.allocation.getRefOrVec())
        return deallocation.emitOpError(
            "LQCloud deallocation does not match the combined allocation");
      state.resetPrefixOpen = false;
      continue;
    }
    if (auto extract = dyn_cast<quake::ExtractRefOp>(operation)) {
      if (!getQubitIndex(extract.getRef(), state))
        return extract.emitOpError(
            "LQCloud requires every qubit reference to use a constant index "
            "into the combined allocation");
      continue;
    }
    if (auto reset = dyn_cast<quake::ResetOp>(operation)) {
      if (failed(emitReset(state, reset)))
        return failure();
      continue;
    }
    if (auto barrier = dyn_cast<quake::BarrierOp>(operation)) {
      if (failed(emitBarrier(state, barrier)))
        return failure();
      continue;
    }
    if (auto measurement = dyn_cast<quake::MzOp>(operation)) {
      if (failed(emitMeasurement(state, measurement)))
        return failure();
      continue;
    }
    if (auto quantumOperator = dyn_cast<quake::OperatorInterface>(operation)) {
      if (failed(emitOperator(state, quantumOperator)))
        return failure();
      continue;
    }
    auto dialect = operation.getName().getDialectNamespace();
    if (dialect == "cc" || dialect == "arith" || dialect == "llvm")
      continue;
    return operation.emitOpError() << "LQCloud does not support operation '"
                                   << operation.getName() << "'";
  }

  if (state.measuredQubits.empty())
    return func.emitOpError(
        "LQCloud requires at least one terminal measurement");

  if (state.explicitMeasurements &&
      state.measurementEvents.size() != state.numberOfMeasurements)
    return func.emitOpError("LQCloud explicit measurement events were lost");

  // Mapping's reorder indices refer to the measured physical qubits in
  // ascending order. Compact only these bits, excluding unmeasured ancillas.
  if (!state.explicitMeasurements)
    for (auto &op : state.instructions)
      if (op.at("name") == "measure") {
        auto qubit = op.at("qubits").at(0).get<std::size_t>();
        op["clbits"] = {std::distance(state.measuredQubits.begin(),
                                      state.measuredQubits.find(qubit))};
      }

  json["n_qubits"] = state.numberOfQubits;
  json["n_clbits"] = state.explicitMeasurements ? state.numberOfMeasurements
                                                : state.measuredQubits.size();
  json["instructions"] = std::move(state.instructions);
  return success();
}

LogicalResult emitModule(nlohmann::json &json, ModuleOp module) {
  func::FuncOp entryPoint;
  for (Operation &operation : module.getBody()->getOperations()) {
    auto func = dyn_cast<func::FuncOp>(operation);
    if (!func)
      return operation.emitOpError()
             << "LQCloud does not support top-level operation '"
             << operation.getName() << "'";
    if (!func->hasAttr(cudaq::entryPointAttrName))
      return func.emitOpError(
          "LQCloud does not support non-entrypoint functions");
    if (entryPoint)
      return module.emitError("LQCloud module has multiple entrypoints");
    entryPoint = func;
  }
  if (!entryPoint)
    return module.emitError("LQCloud module does not contain an entrypoint");
  return emitEntryPoint(json, entryPoint);
}

} // namespace

LogicalResult cudaq::translateToLQCloudJson(Operation *op,
                                            llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("LQCloud translation requires a builtin.module");

  nlohmann::json json;
  if (failed(emitModule(json, module)))
    return failure();
  os << json.dump(2) << '\n';
  return success();
}
