from dataclasses import dataclass


@dataclass
class APICostConfig:
    model_id: str
    input_price_per_1m: float
    output_price_per_1m: float


@dataclass
class LocalCostConfig:
    model_id: str
    gpu_price_usd: float = 30000.0
    gpu_life_years: float = 3.0
    gpu_tdp_watts: float = 700.0
    gpu_utilization: float = 0.80
    cpu_power_watts: float = 50.0
    pue: float = 1.3
    electricity_per_kwh: float = 0.1778
    throughput_tok_per_sec: float = 50.0
    gpu_count: int = 1

    @property
    def capex_per_hour(self) -> float:
        return self.gpu_price_usd / (self.gpu_life_years * 8760) * self.gpu_count

    @property
    def opex_per_hour(self) -> float:
        power_w = (
            self.gpu_tdp_watts * self.gpu_utilization + self.cpu_power_watts
        ) * self.pue
        return (power_w / 1000) * self.electricity_per_kwh * self.gpu_count

    @property
    def total_per_hour(self) -> float:
        return self.capex_per_hour + self.opex_per_hour

    @property
    def cost_per_token(self) -> float:
        if self.throughput_tok_per_sec <= 0:
            return 0.0
        return self.total_per_hour / (self.throughput_tok_per_sec * 3600)


@dataclass
class HybridCostResult:
    plan_cost_usd: float
    exec_cost_usd: float
    total_cost_usd: float
    plan_model: str
    exec_model: str


def compute_api_cost(
    config: APICostConfig, input_tokens: int, output_tokens: int
) -> float:
    input_cost = (input_tokens / 1_000_000) * config.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * config.output_price_per_1m
    return input_cost + output_cost


def compute_local_cost(
    config: LocalCostConfig, input_tokens: int, output_tokens: int
) -> float:
    total_tokens = input_tokens + output_tokens
    return total_tokens * config.cost_per_token


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


CLAUDE_OPUS_46 = APICostConfig(
    "claude-opus-4-6", input_price_per_1m=5.0, output_price_per_1m=25.0
)
GPT_54 = APICostConfig("gpt-5.4", input_price_per_1m=2.5, output_price_per_1m=15.0)

QWEN3_32B = LocalCostConfig("Qwen/Qwen3-32B", throughput_tok_per_sec=22.3)
QWEN3_4B = LocalCostConfig("Qwen/Qwen3-4B", throughput_tok_per_sec=136.7)
