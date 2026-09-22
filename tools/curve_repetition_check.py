# 曲线内重复检查只读取真实日表，不改变余额或补造流水。
"""识别连续复制的月内锯齿与跨周重复波形。"""
# 同一检查既可批量调用也可独立复核。
import argparse
# 月份完整性按自然日判断。
from calendar import monthrange
# 组合用于查找三段互相相似的波形。
from itertools import combinations
# 日表读取与精确金额转换复用已有实现。
import synthetic_batch_workflow as flow
# 数值运算仅用于形状统计，不参与记账。
import numpy as np

# 规则先冻结，不以某张新图能否通过来调阈值。
POLICY = {"version": "within_curve_repetition_v2", "monthly_correlation": 0.92, "rolling_correlation": 0.94, "repeated_cycles": 3, "monthly_neighborhood": 6, "period_days": [7, 10, 14, 21, 28, 30, 31, 45, 60, 90, 180, 365], "minimum_direction_share": 0.10, "minimum_oscillation_share": 0.15,
          "whole_curve_monthly_correlation": 0.90, "whole_curve_min_component_months": 7,
          "whole_curve_min_component_share": 0.30, "whole_curve_min_component_pair_density": 0.25}

# 去掉余额水平、规模和线性趋势，防止换金额掩盖相同锯齿。
def shape(changes):
    # 非真实缺失区间不能拿零补齐，本函数只接受已有数列。
    values = np.asarray(changes, dtype=float)
    # 至少一个完整短周期才能比较。
    if len(values) < 7:
        # 太短不冒充通过形状验证。
        return None
    # 单向维持费、停产后的单向下降不会形成心电图式来回波形。
    positive, negative = values[values > 0].sum(), -values[values < 0].sum()
    # 总流动幅度用以排除微小噪声和近乎平直段。
    gross = positive + negative
    # 需要同时有实质上冲和下冲。
    if gross == 0 or min(positive, negative) < POLICY["minimum_direction_share"] * gross:
        # 固定工资或租金本身不构成整条曲线复制。
        return None
    # 净收支累积成局部余额变化，起点不含期初注资。
    level = np.concatenate(([0.0], np.cumsum(values)))
    # 只消除直线趋势，不把波形磨平。
    residual = level - np.linspace(level[0], level[-1], len(level))
    # 只有足够明显的振荡才列入机械波形核查。
    if np.ptp(residual) < POLICY["minimum_oscillation_share"] * gross:
        # 大量小额随机支出不应被微小相关误报。
        return None
    # 将不同月长映射到同一月内进度，而不是伪造实际日期。
    return np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(residual)), residual)

# 相关度只比较已满足明显双向振荡条件的波形。
def similar(a, b, threshold):
    # 平直或单向段不参与锯齿相似判断。
    return a is not None and b is not None and flow.correlation(a, b) >= threshold


def whole_curve_density(months):
    """Find a dense, repeated month-template component across the whole curve.

    The original witness rule catches any local triple.  This companion check
    catches a visibly repeated template spread over a longer horizon, while
    still requiring substantial two-way waves (the ``shape`` gate) and a
    dense graph rather than a single coincidental similar pair.
    """
    eligible = [(month, wave) for month, wave in months if wave is not None]
    names = [month for month, _ in eligible]
    edges = []
    for left, (left_month, left_wave) in enumerate(eligible):
        for right_month, right_wave in eligible[left + 1:]:
            value = flow.correlation(left_wave, right_wave)
            if value is not None and value >= POLICY["whole_curve_monthly_correlation"]:
                edges.append((left_month, right_month, float(value)))
    graph = {name: set() for name in names}
    for left, right, _ in edges:
        graph[left].add(right); graph[right].add(left)
    visited, evidence = set(), []
    for start in names:
        if start in visited:
            continue
        pending, component = [start], []
        visited.add(start)
        while pending:
            node = pending.pop(); component.append(node)
            for adjacent in graph[node]:
                if adjacent not in visited:
                    visited.add(adjacent); pending.append(adjacent)
        component = sorted(component)
        component_set = set(component)
        component_edges = [edge for edge in edges if edge[0] in component_set and edge[1] in component_set]
        possible = len(component) * (len(component) - 1) / 2
        density = len(component_edges) / possible if possible else 0.0
        share = len(component) / len(eligible) if eligible else 0.0
        if (len(component) >= POLICY["whole_curve_min_component_months"]
                and share >= POLICY["whole_curve_min_component_share"]
                and density >= POLICY["whole_curve_min_component_pair_density"]):
            evidence.append({"months": component, "month_count": len(component), "eligible_month_count": len(eligible),
                             "eligible_month_share": share, "similar_pair_count": len(component_edges), "pair_density": density,
                             "minimum_correlation": min(edge[2] for edge in component_edges)})
    return {"policy": {key: POLICY[key] for key in ("whole_curve_monthly_correlation", "whole_curve_min_component_months", "whole_curve_min_component_share", "whole_curve_min_component_pair_density")},
            "eligible_month_count": len(eligible), "similar_pair_count": len(edges), "evidence": evidence,
            "passed": not evidence}

# 每户给出证据位置和明确交付资格。
def check(daily, setup_end="0001-01-01"):
    # 筹建结束之前不参与经营重复形状判断。
    rows = [r for r in daily if r["calendar_date"] > setup_end]
    # 所有金额只取正式日表的真实进出账。
    changes = [flow.delivery.fen(r["inflow_cny"]) - flow.delivery.fen(r["outflow_cny"]) for r in rows]
    # 月内形状优先按自然月对齐。
    groups = {}
    # 保留日期便于定位需要复核的段落。
    for row, value in zip(rows, changes, strict=True):
        # 顺序沿用正式日表。
        groups.setdefault(row["calendar_date"][:7], []).append((row["calendar_date"], value))
    # 仅完整月份作月内形状比较。
    months = []
    # 缺少未来的停止月份不补齐。
    for month, values in groups.items():
        # 核查自然月份长度。
        year, number = map(int, month.split("-"))
        # 不将不完整月冒充完整周期。
        if len(values) == monthrange(year, number)[1]:
            # 保存标准化形状和月份。
            months.append((month, shape([v for _, v in values])))
    # 保存月内重复证据。
    monthly = []
    # 任意连续六个月内若有三个月相互高度相似即阻断。
    for i in range(len(months)):
        # 不只检查相邻月份，防止隔月复制漏检。
        for j, k in combinations(range(i+1, min(i+POLICY["monthly_neighborhood"], len(months))), 2):
            # 三对都相似，避免单个偶然相近就否决样本。
            if all(similar(months[a][1], months[b][1], POLICY["monthly_correlation"]) for a, b in ((i, j), (i, k), (j, k))):
                # 月份和最弱相关度都记录下来。
                monthly.append({"months": [months[n][0] for n in (i, j, k)], "minimum_correlation": min(flow.correlation(months[a][1], months[b][1]) for a, b in ((i,j),(i,k),(j,k)))})
    # 跨周检查不要求波形恰好从每月一日开始。
    rolling = []
    # 覆盖旬、双周、月和较长周期。
    for period in POLICY["period_days"]:
        # 逐日移动起点，避免只检查固定相位。
        for offset in range(max(0, len(changes)-3*period+1)):
            # 真实连续三周期都必须有数据。
            waves = [shape(changes[offset+n*period:offset+(n+1)*period]) for n in range(3)]
            # 同样要求三个周期相互高度相似。
            if all(similar(waves[a], waves[b], POLICY["rolling_correlation"]) for a, b in ((0,1),(0,2),(1,2))):
                # 每个周期长度保留第一处证据，避免报告重复堆叠。
                rolling.append({"start": rows[offset]["calendar_date"], "end": rows[offset+3*period-1]["calendar_date"], "period_days": period})
                # 一处即可否决，无需继续扫描同周期长度。
                break
    # 整条曲线密度检查补足“局部三重模板”无法覆盖的跨期机械锯齿。
    density = whole_curve_density(months)
    # 没有三周期的数据不冒充已充分验证多样性。
    sufficient = len(rows) >= 90
    # 质量门槛与可解释经营停止是两回事。
    rejected = bool(monthly or rolling or not density["passed"])
    return {"policy": POLICY, "passed": not rejected, "sufficient_observation": sufficient,
            "status": "rejected_repeating_waveform" if rejected else "no_repetition_detected" if sufficient else "short_record_requires_review",
            "monthly_evidence": monthly, "rolling_evidence": rolling, "whole_curve_density": density, "actual_days": len(rows),
            "scope": "识别明显双向重复波形及整条曲线的高密度重复模板；单向停产维持费和固定费用本身不否决；短记录不证明长期多样性"}

# 独立复核整个已交付批次，无需再次生成。
def main():
    # 只读取明确汇总与输出位置。
    parser = argparse.ArgumentParser(description="单户曲线内重复门槛")
    # 交付汇总内已有每户日表引用。
    parser.add_argument("--summary", required=True)
    # 旧文件不覆盖，检查结果另存。
    parser.add_argument("--output", required=True)
    # 读取命令参数。
    args = parser.parse_args()
    # 日表路径逐项核对原字节。
    result = {r["sample_id"]: check(flow.delivery.read_csv(flow.reference(r["artifacts"]["日余额"]))) for r in flow.read(flow.locate(args.summary))["delivered"]}
    # 保存可以追溯日期的完整证据。
    flow.stable_write(flow.output_path(args.output), result)
    # 简要报告否决数。
    print({"checked": len(result), "repetition_rejected": sum(not r["passed"] for r in result.values())})

# 被工作流导入时不单独执行。
if __name__ == "__main__":
    # 固定命令入口。
    main()
