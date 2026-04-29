# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-29_binary_max_min_quasar_dualpath
#
# Three-operand SFPU test for binary_max_min on Quasar.
#
# Buffer layout (in buffer_A, tile_count_A=2):
#   tile 0 (in0) + tile 1 (in1) both in buffer_A
#   SFPU computes max/min(Dest[0], Dest[1]) → Dest[2]
#   PACK reads result from Dest[2]   (tile_count_res=1)
#
# Two execution paths in the kernel, picked from the format:
#   * 32-bit formats (Float32, Int32) with dest_acc=Yes   → unpack_to_dest=True
#   * Non-32-bit / MX formats with dest_acc=No            → unpack_to_dest=False
#                                                           (UNPACK→SrcA→FPU datacopy→Dest)
#
# Format matrix (§7d, analyzer spec):
#   Float variant: Float16_b, Float32, MxFp8R, MxFp8P (Float16 excluded — see below)
#   Int32 variant: Int32 (IS_UNSIGNED=true deferred — UInt32 not in VALID_QUASAR_DEST_REG_FORMATS)

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.golden_generators import quantize_mx_stimuli
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    IS_MAX_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test, tolerances

# ─── Input preparation ────────────────────────────────────────────────────────


def prepare_binary_max_min_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
) -> tuple:
    """
    Prepare two input tensors for binary max/min with safe value ranges.

    Returns (in0, in1) both clamped to safe representable range for
    the given input format. Both tensors have the same format.

    For float formats: moderate log-uniform positive+negative values.
    For Int32: moderate positive+negative integer values to avoid overflow.
    """
    if input_format == DataFormat.Int32:
        # For Int32: safe range avoids extreme values near INT32_MIN/MAX
        # that could cause undefined behaviour in sign-magnitude comparison
        max_val = 2**28  # well inside INT32 range; exercises both +/-
        in0 = (src_A * max_val).to(torch.int32).to(torch.float32)
        in1 = (src_B * max_val).to(torch.int32).to(torch.float32)
        # Ensure values are representable as int32
        in0 = torch.clamp(in0, -max_val, max_val)
        in1 = torch.clamp(in1, -max_val, max_val)
        return in0, in1

    # Float formats: log-uniform magnitudes, random signs
    torch_fmt = format_dict[input_format]
    finfo = torch.finfo(torch_fmt)

    max_safe = min(finfo.max * 0.9, 1e4)  # conservative; avoids overflow in any format
    min_mag = max(finfo.tiny * 100, 1e-6)  # avoid denormals

    def log_uniform(t: torch.Tensor) -> torch.Tensor:
        """Map t in arbitrary range to log-uniform magnitudes [min_mag, max_safe]."""
        t_f = t.to(torch.float32)
        lo, hi = t_f.min(), t_f.max()
        if hi > lo:
            t_norm = (t_f - lo) / (hi - lo)
        else:
            t_norm = torch.zeros_like(t_f)
        log_lo = torch.log(torch.tensor(min_mag, dtype=torch.float32))
        log_hi = torch.log(torch.tensor(max_safe, dtype=torch.float32))
        return torch.exp(log_lo + t_norm * (log_hi - log_lo))

    mag_A = log_uniform(src_A)
    mag_B = log_uniform(src_B)

    # Assign signs: 50% negative for each
    signs_A = torch.where(src_A.to(torch.float32) < 0, -1.0, 1.0)
    signs_B = torch.where(src_B.to(torch.float32) < 0, -1.0, 1.0)

    in0 = torch.clamp(signs_A * mag_A, -max_safe, max_safe)
    in1 = torch.clamp(signs_B * mag_B, -max_safe, max_safe)
    return in0, in1


# ─── Invalid-combination filter ────────────────────────────────────────────────


def _is_invalid_quasar_combination(
    fmt: FormatConfig, dest_acc: DestAccumulation
) -> bool:
    """
    Returns True if the (fmt, dest_acc) combination is invalid for Quasar SFPU tests.

    Bit-width rule (with the dual-path kernel):
    - 32-bit input  → unpack_to_dest=True path → Dest must be 32-bit (dest_acc=Yes).
    - Non-32-bit input → unpack_to_dest=False (FPU datacopy) path → dest_acc=No keeps
      Dest at the input bit-width and avoids the 32-bit-Dest ELWADD path. (dest_acc=Yes
      is left out of this matrix to keep it tight; the kernel supports it but we don't
      need a second Dest mode for non-32-bit inputs here.)
    - Non-Float32 → Float32 needs dest_acc=Yes.
    - Float32 → Float16 needs dest_acc=Yes.
    - Integer and float formats cannot be mixed in input→output.
    """
    in_fmt = fmt.input_format
    out_fmt = fmt.output_format

    if in_fmt.is_32_bit() != (dest_acc == DestAccumulation.Yes):
        return True

    # Quasar packer: non-Float32 → Float32 needs dest_acc=Yes
    if (
        in_fmt != DataFormat.Float32
        and out_fmt == DataFormat.Float32
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Quasar SFPU: Float32 → Float16 needs dest_acc=Yes
    if (
        in_fmt == DataFormat.Float32
        and out_fmt == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Integer and float cannot be mixed
    if in_fmt.is_integer() != out_fmt.is_integer():
        return True

    return False


# ─── Format lists (§7d coverage) ──────────────────────────────────────────────

# Float variant — §7d pairs covered by the dual-path kernel.
#
# Float16 is excluded: SFPSWAP VEC_MIN_MAX with FP16A in 16-bit Dest does not correctly
# compare negative pairs in the simulator regardless of sfpmem::DEFAULT or FP16A load
# mode. Float16_b (FP32 exponent bias) works correctly.
SFPU_BINARY_MAX_MIN_FLOAT_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.MxFp8R, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.MxFp8P, DataFormat.Float16_b),
]

# Int32 variant: 1 Yes-listed pair from §7d (UInt32 deferred — not in VALID_QUASAR_DEST_REG_FORMATS)
SFPU_BINARY_MAX_MIN_INT32_FORMATS = [
    InputOutputFormat(DataFormat.Int32, DataFormat.Int32),  # Int32 → Int32
]


# ─── Combination generators ────────────────────────────────────────────────────


def generate_binary_max_min_float_combinations(formats_list: List[FormatConfig]):
    """
    Generate (format, dest_acc, implied_math_format, is_max_op, input_dimensions) tuples
    for the float variant (non-Int32 formats).

    §7d Yes-count: 5 pairs. dest_acc filtered per SFPU bit-width rule.
    """
    combinations = []
    for fmt in formats_list:
        in_fmt = fmt.input_format

        # SFPU bit-width rule: 32-bit input → dest_acc=Yes only; else dest_acc=No only
        dest_acc_modes = (
            (DestAccumulation.Yes,) if in_fmt.is_32_bit() else (DestAccumulation.No,)
        )

        for dest_acc in dest_acc_modes:
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                # MX formats require implied_math_format=Yes
                if (
                    in_fmt.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue

                for is_max_op in [True, False]:  # max AND min
                    for input_dimensions in [[32, 32]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                is_max_op,
                                input_dimensions,
                            )
                        )

    return combinations


def generate_binary_max_min_int32_combinations(formats_list: List[FormatConfig]):
    """
    Generate combinations for the int32 variant (Int32 input, IS_UNSIGNED=false).

    §7d Yes-count: 1 pair. Int32 requires dest_acc=Yes (32-bit Dest).
    """
    combinations = []
    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,) if in_fmt.is_32_bit() else (DestAccumulation.No,)
        )

        for dest_acc in dest_acc_modes:
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for is_max_op in [True, False]:  # max AND min
                for input_dimensions in [[32, 32]]:
                    combinations.append((fmt, dest_acc, is_max_op, input_dimensions))

    return combinations


# ─── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=generate_binary_max_min_float_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS
    ),
)
def test_binary_max_min_float_quasar(formats_dest_acc_implied_math_is_max_input_dims):
    """
    Test float variant of binary_max_min (calculate_binary_max_min) on Quasar.

    Verifies element-wise max and min across two Dest tile regions using
    SFPSWAP (sign-magnitude comparison = FP32 total order).

    Golden: torch.maximum / torch.minimum applied element-wise.
    """
    (formats, dest_acc, implied_math_format, is_max_op, input_dimensions) = (
        formats_dest_acc_implied_math_is_max_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
        negative_values=True,
    )

    # Prepare in0 and in1 inputs with safe value ranges
    in0, in1 = prepare_binary_max_min_inputs(src_A, src_B, formats.input_format)

    # Convert to target format
    torch_fmt = format_dict[formats.input_format]
    in0 = in0.to(torch_fmt)
    in1 = in1.to(torch_fmt)

    num_faces = 4

    # MX formats use a 32-element shared block exponent. Quantizing both inputs
    # through the same pack→unpack roundtrip the hardware applies makes the
    # golden compare against what the kernel actually sees in Dest, not the
    # higher-precision bfloat16 source values.
    if formats.input_format.is_mx_format():
        in0_for_golden = quantize_mx_stimuli(
            in0.flatten(), formats.input_format, num_faces
        ).reshape(in0.shape)
        in1_for_golden = quantize_mx_stimuli(
            in1.flatten(), formats.input_format, num_faces
        ).reshape(in1.shape)
    else:
        in0_for_golden = in0
        in1_for_golden = in1

    in0_f32 = in0_for_golden.to(torch.float32)
    in1_f32 = in1_for_golden.to(torch.float32)
    if is_max_op:
        golden_f32 = torch.maximum(in0_f32, in1_f32)
    else:
        golden_f32 = torch.minimum(in0_f32, in1_f32)

    output_torch_fmt = format_dict[formats.output_format]
    golden_tensor = golden_f32.to(output_torch_fmt)

    # buffer_A has 2 tiles: tile 0 = in0, tile 1 = in1, concatenated.
    # StimuliConfig with tile_count_A=2 reads buffer_A[0:1024] as tile 0
    # and buffer_A[1024:2048] as tile 1 (stride = MAX_TILE_ELEMENTS=1024).
    # buffer_B is a dummy (required by StimuliConfig; not written to the kernel).
    buffer_A_combined = torch.cat([in0.flatten(), in1.flatten()])

    # If in0/in1 have fewer than 1024 elements (e.g. partial faces), pad to 1024 each
    max_tile_elements = 1024
    if len(in0.flatten()) < max_tile_elements:
        pad_len = max_tile_elements - len(in0.flatten())
        buffer_A_combined = torch.cat(
            [
                in0.flatten(),
                torch.zeros(pad_len, dtype=in0.dtype),
                in1.flatten(),
                torch.zeros(pad_len, dtype=in1.dtype),
            ]
        )

    # 32-bit input + dest_acc=Yes  → unpack_to_dest=True
    # Everything else (incl. Float16_b, MxFp8R, MxFp8P) → unpack_to_dest=False
    # (UNPACK→SrcA→FPU datacopy→Dest path; required for MX block formats).
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_max_min_quasar_test.cpp",
        formats,
        templates=[
            IS_MAX_OP(is_max_op=is_max_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(2),  # 2 input tiles in buffer_A
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,  # in0 (tile 0) + in1 (tile 1) concatenated
            formats.input_format,
            in1,  # dummy in buffer_B (not used by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,  # kernel packs only the SFPU output (Dest[2])
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)

    # For MX inputs, the kernel's effective precision is bounded by the MX format,
    # not by the (tighter) Float16_b output format. Override the tolerance accordingly.
    custom_atol = None
    custom_rtol = None
    if formats.input_format.is_mx_format():
        mx_tol = tolerances[formats.input_format]
        custom_atol = mx_tol.atol
        custom_rtol = mx_tol.rtol

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), (
        f"Assert against golden failed for is_max_op={is_max_op}, "
        f"format={formats.input_format}->{formats.output_format}, dest_acc={dest_acc}"
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_is_max_input_dims=generate_binary_max_min_int32_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS
    ),
)
def test_binary_max_min_int32_quasar(formats_dest_acc_is_max_input_dims):
    """
    Test int32 variant of binary_max_min (calculate_binary_max_min_int32) on Quasar.

    Verifies element-wise signed max and min for Int32 data using SFPSWAP +
    CC-guarded correction to handle sign-magnitude vs two's-complement discrepancy.

    IS_UNSIGNED=true (UInt32) is deferred — UInt32 not in VALID_QUASAR_DEST_REG_FORMATS.

    Golden: torch.maximum / torch.minimum on int32 values.
    """
    (formats, dest_acc, is_max_op, input_dimensions) = (
        formats_dest_acc_is_max_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
        negative_values=True,
    )

    # Prepare int32 inputs (returns float32-container tensors in [-2^28, 2^28])
    in0_f, in1_f = prepare_binary_max_min_inputs(src_A, src_B, formats.input_format)

    # Convert to actual int32 for packing and golden computation
    in0_int = in0_f.to(torch.int32)
    in1_int = in1_f.to(torch.int32)

    # Golden: element-wise signed max or min in two's-complement int32
    # (the hardware kernel corrects sign-magnitude to two's-complement ordering)
    if is_max_op:
        golden_int = torch.maximum(in0_int, in1_int)
    else:
        golden_int = torch.minimum(in0_int, in1_int)

    golden_tensor = golden_int.to(torch.float32)

    num_faces = 4

    output_torch_fmt = format_dict[formats.output_format]

    # Concatenate in0 and in1 into buffer_A (tile_count_A=2)
    # pack_int32 expects int32 tensors (converts two's-complement → sign-magnitude for HW)
    max_tile_elements = 1024
    buffer_A_combined = torch.cat([in0_int.flatten(), in1_int.flatten()])
    if len(in0_int.flatten()) < max_tile_elements:
        pad_len = max_tile_elements - len(in0_int.flatten())
        buffer_A_combined = torch.cat(
            [
                in0_int.flatten(),
                torch.zeros(pad_len, dtype=torch.int32),
                in1_int.flatten(),
                torch.zeros(pad_len, dtype=torch.int32),
            ]
        )

    # Int32 is 32-bit with dest_acc=Yes → unpack_to_dest=True path.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_max_min_quasar_test.cpp",
        formats,
        templates=[
            IS_MAX_OP(is_max_op=is_max_op),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(2),  # 2 input tiles in buffer_A
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,  # in0 (tile 0) + in1 (tile 1) concatenated
            formats.input_format,
            in1_int,  # dummy in buffer_B (not used by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,  # kernel packs only the SFPU output (Dest[2])
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Assert against golden failed for is_max_op={is_max_op}, int32 format"
