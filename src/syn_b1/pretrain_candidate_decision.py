"""SYN-B1 T6：只依据冻结验证指标判定复杂候选是否有资格开启最终测试。"""

# 导入 dataclass；它将每个候选的只读验证事实组织为明确字段。
from dataclasses import dataclass
# 导入 Decimal；它避免比较八位小数指标时受到二进制浮点误差影响。
from decimal import Decimal

# 固定T6实现版本；报告通过此标识避免与以后新数据版本混淆。
T6_VERSION = "syn_b1_pretrain_t6_candidate_decision_1_0"
# 固定合同唯一主指标名称；所有候选必须与锁定基线使用同一口径。
PRIMARY_METRIC = "account_normalized_mae"


# 声明T6专用异常；它表示血缘、密封或评价条件失败，而非模型成绩不好。
class CandidateDecisionContractError(ValueError):
    """当候选比较不满足冻结实验合同的硬条件时抛出。"""


# 使用不可变对象保存一个复杂候选的验证事实；它不持有任何最终测试资料。
@dataclass(frozen=True)
class CandidateMetric:
    # 保存机器可读候选名称，供CSV和JSON稳定引用。
    model_name: str
    # 保存中文候选名称，供项目发起人直接阅读结论。
    model_name_cn: str
    # 保存账户归一化MAE；数值越小越好。
    primary_metric: Decimal
    # 保存与锁定基线的Skill Score；仅作辅助说明，不替代主指标。
    skill_score: Decimal
    # 保存验证样本数，保证各候选比较同一批3,904行。
    validation_sample_count: int
    # 保存所有泄漏、资格、复现和密封检查是否通过的合并结论。
    guard_checks_passed: bool
    # 保存复杂度次序；主指标完全相同才用于选择较简单者。
    complexity_rank: int


# 用固定规则判断一个复杂候选是否达到打开最终测试的最低条件。
def is_eligible(candidate: CandidateMetric, locked_baseline_metric: Decimal) -> bool:
    # 若任何硬性护栏失败，则不论指标多好都禁止进入最终测试。
    if not candidate.guard_checks_passed:
        # 返回否，明确资格检查优先于效果比较。
        return False
    # 若验证主指标大于锁定基线，则候选预测误差更大，不满足最低条件。
    if candidate.primary_metric > locked_baseline_metric:
        # 返回否，禁止用最终测试替复杂模型寻找优势。
        return False
    # 只有护栏通过且验证指标不差于基线时才返回是。
    return True


# 在所有复杂候选中执行验证集唯一候选规则；返回选中者或None以及中文原因。
def decide_unique_candidate(candidates: list[CandidateMetric], locked_baseline_metric: Decimal) -> tuple[CandidateMetric | None, str]:
    # 筛选同时满足护栏与主指标门槛的复杂候选。
    eligible_candidates = [candidate for candidate in candidates if is_eligible(candidate, locked_baseline_metric)]
    # 没有合格候选时必须如实关闭最终测试。
    if not eligible_candidates:
        # 返回空候选和合同规定的中文理由。
        return None, "没有复杂候选在验证集主指标上不差于锁定基线；最终测试保持关闭。"
    # 先按主指标升序、再按复杂度升序和名称排序，落实合同的平局规则并保证稳定输出。
    selected = sorted(eligible_candidates, key=lambda item: (item.primary_metric, item.complexity_rank, item.model_name))[0]
    # 返回唯一候选及说明；调用方仍须另行执行最终测试授权门。
    return selected, "候选满足验证集最低条件；仅在获得独立授权后才可开启一次最终测试。"
