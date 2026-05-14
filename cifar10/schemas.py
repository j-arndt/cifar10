"""Pydantic schemas for kernel proposals — the only shape the firewall accepts."""
from __future__ import annotations

import ast
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator, model_validator


class KernelType(str, Enum):
    CONV_BN_RELU_FUSION = "conv_bn_relu_fusion"
    DEPTHWISE_CONV       = "depthwise_conv"
    OPTIMIZER_FUSION     = "optimizer_fusion"
    DATA_PIPELINE        = "data_pipeline"


class KernelProposal(BaseModel):
    kernel_type:          KernelType
    layer_target:         str       = Field(..., max_length=50)
    cuda_kernel_code:     str       = Field(..., min_length=10, max_length=65536)
    pytorch_binding:      str       = Field(..., min_length=10, max_length=32768)
    integration_patch:    str       = Field(..., min_length=5,  max_length=4096)
    rationale:            str       = Field(..., max_length=1000)
    expected_speedup_pct: int       = Field(..., ge=1, le=500)

    @field_validator("pytorch_binding")
    @classmethod
    def must_be_valid_python(cls, v: str) -> str:
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"pytorch_binding is not valid Python: {e}") from e
        return v

    @field_validator("cuda_kernel_code")
    @classmethod
    def must_have_global(cls, v: str) -> str:
        if "__global__" not in v and "def " not in v:
            raise ValueError("cuda_kernel_code must contain __global__ (CUDA) or def (Triton)")
        return v

    model_config = {"str_strip_whitespace": True}


@dataclass
class FirewallResult:
    passed:       bool
    gate_failed:  Optional[str]           = None
    error_message: Optional[str]          = None
    proposal:     Optional[KernelProposal] = None

    def __bool__(self):
        return self.passed
