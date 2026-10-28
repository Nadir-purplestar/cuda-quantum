/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace cudaq::opt {
#define GEN_PASS_DEF_LQCLOUDGATEDECOMPOSITION
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

namespace {

constexpr double pi = 3.14159265358979323846;

Value createFloatConstant(IRRewriter &rewriter, Location location, double value,
                          FloatType type) {
  return cudaq::opt::factory::createFloatConstant(location, rewriter, value,
                                                  type);
}

Value createF64Constant(IRRewriter &rewriter, Location location, double value) {
  return createFloatConstant(rewriter, location, value, rewriter.getF64Type());
}

void createH(IRRewriter &rewriter, Location location, Value target) {
  rewriter.create<quake::HOp>(location, ValueRange{target});
}

void createRz(IRRewriter &rewriter, Location location, Value angle,
              Value target) {
  rewriter.create<quake::RzOp>(location, ValueRange{angle}, ValueRange{},
                               ValueRange{target});
}

void rewriteOperator(Operation *operation, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(operation);
  Location location = operation->getLoc();
  auto op = cast<quake::OperatorInterface>(operation);
  if (op.getTargets().size() != 1 ||
      !isa<quake::RefType>(op.getTargets().front().getType()))
    return;
  const auto controls = op.getControls();
  if (!controls.empty() &&
      !(isa<quake::XOp>(operation) && controls.size() == 1 &&
        isa<quake::RefType>(controls.front().getType())) &&
      !(isa<quake::YOp>(operation) && controls.size() == 1 &&
        isa<quake::RefType>(controls.front().getType())))
    return;

  if (isa<quake::XOp>(operation)) {
    // Controlled X is lowered by the public CXToCZ pattern in the subsequent
    // basis conversion; uncontrolled X is already in the native gate set.
    return;
  }

  if (auto y = dyn_cast<quake::YOp>(operation)) {
    if (y.getControls().empty())
      return;
    Value target = y.getTarget();
    Value negativeHalfPi = createF64Constant(rewriter, location, -pi / 2.0);
    Value positiveHalfPi = createF64Constant(rewriter, location, pi / 2.0);
    createRz(rewriter, location, negativeHalfPi, target);
    rewriter.create<quake::XOp>(location, ValueRange{y.getControls().front()},
                                ValueRange{target});
    createRz(rewriter, location, positiveHalfPi, target);
    rewriter.eraseOp(operation);
    return;
  }

  if (auto z = dyn_cast<quake::ZOp>(operation)) {
    if (!z.getControls().empty())
      return;
    Value angle = createF64Constant(rewriter, location, pi);
    createRz(rewriter, location, angle, z.getTarget());
    rewriter.eraseOp(operation);
    return;
  }

  if (auto rx = dyn_cast<quake::RxOp>(operation)) {
    if (!isa<FloatType>(rx.getParameter().getType()))
      return;
    Value angle = rx.getParameter();
    createH(rewriter, location, rx.getTarget());
    createRz(rewriter, location, angle, rx.getTarget());
    createH(rewriter, location, rx.getTarget());
    rewriter.eraseOp(operation);
    return;
  }

  if (auto ry = dyn_cast<quake::RyOp>(operation)) {
    if (!isa<FloatType>(ry.getParameter().getType()))
      return;
    Value angle = ry.getParameter();
    auto angleType = cast<FloatType>(angle.getType());
    Value negativeHalfPi =
        createFloatConstant(rewriter, location, -pi / 2.0, angleType);
    Value positiveHalfPi =
        createFloatConstant(rewriter, location, pi / 2.0, angleType);
    createRz(rewriter, location, negativeHalfPi, ry.getTarget());
    createH(rewriter, location, ry.getTarget());
    createRz(rewriter, location, angle, ry.getTarget());
    createH(rewriter, location, ry.getTarget());
    createRz(rewriter, location, positiveHalfPi, ry.getTarget());
    rewriter.eraseOp(operation);
    return;
  }
}

struct LQCloudGateDecompositionPass
    : public cudaq::opt::impl::LQCloudGateDecompositionBase<
          LQCloudGateDecompositionPass> {
  using LQCloudGateDecompositionBase::LQCloudGateDecompositionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr("lqcloud.explicit_measurements") &&
        !module->hasAttr("lqcloud.measurement_count")) {
      std::size_t count = 0;
      auto result = module.walk([&](quake::MzOp measurement) {
        if (!isa<func::FuncOp>(measurement->getParentOp()) ||
            measurement.getTargets().size() != 1 ||
            !isa<quake::RefType>(measurement.getTargets().front().getType())) {
          measurement.emitOpError(
              "LQCloud explicit measurements require a static straight-line "
              "circuit with scalar Z measurements");
          return WalkResult::interrupt();
        }
        if (measurement.getRegisterName()) {
          measurement.emitOpError(
              "LQCloud does not support named measurement registers");
          return WalkResult::interrupt();
        }
        // The standard measurement name survives memory/value conversions and
        // routing. It is internal event metadata, not a user sub-register.
        measurement.setRegisterName(StringAttr::get(
            &getContext(), "__lqcloud_measurement_" + std::to_string(count++)));
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
      if (count == 0) {
        module.emitError(
            "LQCloud explicit measurements require at least one measurement");
        signalPassFailure();
        return;
      }
      module->setAttr(
          "lqcloud.measurement_count",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    }

    SmallVector<Operation *> operators;
    module.walk([&](Operation *operation) {
      if (isa<quake::OperatorInterface>(operation))
        operators.push_back(operation);
    });

    // Reject the two shapes whose controls public decomposition can lose.
    for (Operation *operation : operators) {
      auto quantumOperator = dyn_cast<quake::OperatorInterface>(operation);
      if (quantumOperator.getNegatedControls()) {
        operation->emitOpError("LQCloud does not support negated controls");
        signalPassFailure();
        return;
      }
      if (isa<quake::SwapOp>(operation) &&
          !quantumOperator.getControls().empty()) {
        operation->emitOpError("LQCloud does not support controlled SWAP");
        signalPassFailure();
        return;
      }
    }

    IRRewriter rewriter(&getContext());
    for (Operation *operation : operators)
      rewriteOperator(operation, rewriter);
  }
};

} // namespace
