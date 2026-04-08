from dataclasses import dataclass


@dataclass
class APICostConfig:
    model_id: str
    input_price_per_1m: float
    output_price_per_1m: float
    cache_read_price_per_1m: float = (
        0.0  # price for cached input tokens (typically 90% discount)
    )


@dataclass(frozen=True)
class GPUSpec:
    name: str
    price_usd: float
    tdp_watts: float
    vram_gb: int


GPU_SPECS = {
    "H100_SXM": GPUSpec("H100 SXM", price_usd=35000, tdp_watts=700, vram_gb=80),
}


@dataclass
class LocalCostConfig:
    model_id: str
    gpu: str = "H100_SXM"
    prefill_tok_per_sec: float = 0.0
    decode_tok_per_sec: float = 0.0
    gpu_count: int = 1
    gpu_life_years: float = 3.0
    gpu_utilization: float = 1.0
    pue: float = 1.3
    electricity_per_kwh: float = 0.1778

    @property
    def gpu_spec(self) -> GPUSpec:
        return GPU_SPECS[self.gpu]

    @property
    def capex_per_hour(self) -> float:
        return self.gpu_spec.price_usd / (self.gpu_life_years * 8760) * self.gpu_count

    @property
    def opex_per_hour(self) -> float:
        power_w = self.gpu_spec.tdp_watts * self.gpu_utilization * self.pue
        return (power_w / 1000) * self.electricity_per_kwh * self.gpu_count

    @property
    def total_per_hour(self) -> float:
        return self.capex_per_hour + self.opex_per_hour

    @property
    def prefill_cost_per_token(self) -> float:
        if self.prefill_tok_per_sec <= 0:
            return 0.0
        return self.total_per_hour / (self.prefill_tok_per_sec * 3600)

    @property
    def decode_cost_per_token(self) -> float:
        if self.decode_tok_per_sec <= 0:
            return 0.0
        return self.total_per_hour / (self.decode_tok_per_sec * 3600)


@dataclass
class HybridCostResult:
    plan_cost_usd: float
    exec_cost_usd: float
    total_cost_usd: float
    plan_model: str
    exec_model: str


def compute_api_cost(
    config: APICostConfig,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    uncached_tokens = input_tokens - cached_tokens
    input_cost = (uncached_tokens / 1_000_000) * config.input_price_per_1m
    cache_cost = (cached_tokens / 1_000_000) * config.cache_read_price_per_1m
    output_cost = (output_tokens / 1_000_000) * config.output_price_per_1m
    return input_cost + cache_cost + output_cost


def compute_local_cost(
    config: LocalCostConfig, input_tokens: int, output_tokens: int
) -> float:
    return (
        input_tokens * config.prefill_cost_per_token
        + output_tokens * config.decode_cost_per_token
    )


def compute_local_cost_runtime(
    config: LocalCostConfig,
    input_tokens: int,
    output_tokens: int,
    total_prefill_seconds: float,
    total_decode_seconds: float,
) -> float:
    """Compute local cost using runtime-measured timing.

    Cost = hourly_rate × actual_gpu_time_used.
    GPU time = time spent on prefill + time spent on decode.
    """
    del input_tokens, output_tokens
    gpu_seconds = total_prefill_seconds + total_decode_seconds
    gpu_hours = gpu_seconds / 3600
    return config.total_per_hour * gpu_hours


def compute_hybrid_cost(
    plan_config: APICostConfig,
    exec_config,
    plan_input_tokens: int,
    plan_output_tokens: int,
    exec_input_tokens: int,
    exec_output_tokens: int,
) -> HybridCostResult:
    plan_cost = compute_api_cost(plan_config, plan_input_tokens, plan_output_tokens)

    if isinstance(exec_config, APICostConfig):
        exec_cost = compute_api_cost(exec_config, exec_input_tokens, exec_output_tokens)
    else:
        exec_cost = compute_local_cost(
            exec_config, exec_input_tokens, exec_output_tokens
        )

    return HybridCostResult(
        plan_cost_usd=plan_cost,
        exec_cost_usd=exec_cost,
        total_cost_usd=plan_cost + exec_cost,
        plan_model=plan_config.model_id,
        exec_model=exec_config.model_id,
    )
