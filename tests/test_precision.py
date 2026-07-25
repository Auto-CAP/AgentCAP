"""Unit tests for precision resolution from HF quantization_config.

Regression guard for the bug where quantized checkpoints (MXFP4/INT4/FP8) were
recorded as bfloat16 because only torch_dtype was consulted.
"""
from agent_cap.utils.precision import resolve_precision_from_config as rp
from agent_cap.utils.precision import normalize_model_id


def test_mxfp4_quant_method():
    assert rp({"torch_dtype": "bfloat16",
               "quantization_config": {"quant_method": "mxfp4"}}) == "mxfp4"


def test_compressed_tensors_int4():
    assert rp({"torch_dtype": "bfloat16", "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"g0": {"weights": {"num_bits": 4, "type": "int"}}},
    }}) == "int4"


def test_compressed_tensors_fp8():
    assert rp({"torch_dtype": "bfloat16", "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"g0": {"weights": {"num_bits": 8, "type": "float"}}},
    }}) == "fp8"


def test_kimi_nested_text_config_no_quant_method():
    # Real kimi-k2.5 shape: quant in text_config, uses `dtype`, no quant_method.
    assert rp({"dtype": "bfloat16", "quantization_config": None, "text_config": {
        "dtype": "bfloat16",
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {"g0": {"weights": {"num_bits": 4, "type": "int"}}},
        },
    }}) == "int4"


def test_dtype_key_alias():
    assert rp({"dtype": "bfloat16"}) == "bfloat16"


def test_fp8_and_nvfp4_methods():
    assert rp({"quantization_config": {"quant_method": "fp8"}}) == "fp8"
    assert rp({"quantization_config": {"quant_method": "nvfp4"}}) == "nvfp4"


def test_dense_falls_back_to_dtype():
    assert rp({"torch_dtype": "bfloat16"}) == "bfloat16"


def test_empty_or_bad_input():
    assert rp({}) is None
    assert rp(None) is None


def test_normalize_model_id():
    assert normalize_model_id("/workspace/models/gpt-oss-120b") == "models/gpt-oss-120b"
    assert normalize_model_id("/llm-cache-pvc/models/unsloth/gpt-oss-120b") == "unsloth/gpt-oss-120b"
    assert normalize_model_id("unsloth/gpt-oss-120b") == "unsloth/gpt-oss-120b"
    assert normalize_model_id("gpt-oss-120b") == "gpt-oss-120b"
