"""Readable YAML documents for SYN-B1 user parameter profiles only.

This module owns the file format and Chinese comments.  It deliberately does
not validate business rules or construct a scenario; those responsibilities
remain in ``formal_profile`` and the central generator.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ProfileDocumentError(ValueError):
    """A user profile file cannot be safely read or written."""


def read_profile_document(path: str | Path) -> dict[str, Any]:
    """Read a legacy JSON or the current commented YAML user document."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ProfileDocumentError("找不到参数文件。") from error
    except OSError as error:
        raise ProfileDocumentError("参数文件无法读取。") from error
    try:
        if source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        elif source.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            raise ProfileDocumentError("参数文件必须是.yaml、.yml或历史.json格式。")
    except yaml.YAMLError as error:
        raise ProfileDocumentError(f"参数YAML格式错误：{_yaml_error_line(error)}。") from error
    except json.JSONDecodeError as error:
        raise ProfileDocumentError(f"参数JSON格式错误：第{error.lineno}行。") from error
    if not isinstance(value, Mapping):
        raise ProfileDocumentError("参数文件最外层必须是一组参数。")
    return _normalise(dict(value))


def write_user_profile(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a new readable YAML profile; existing paths are never replaced."""
    target = Path(path)
    if target.exists():
        raise ProfileDocumentError("参数文件已存在，不能覆盖历史文件。")
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise ProfileDocumentError("新参数方案必须使用.yaml格式，便于在记事本中阅读中文说明。")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_annotated_profile(payload), encoding="utf-8")
    return target


def render_annotated_profile(profile: Mapping[str, Any]) -> str:
    # 新月度节点完整写出，避免旧趋势模板丢失业务参数或经营原因。
    if isinstance(profile.get("operating"), Mapping) and "business_nodes" in profile["operating"]:
        # 本分支只负责排版，完整类型及业务校验仍归正式画像解析器。
        timing = "# 收付款delay_days沿用源月15日归集到期月份的现有规则，再按流水规则分配实际日期。\n"
        if profile["operating"]["business_nodes"].get("version") == "physical_contract_business_v1":
            timing = "# settlements逐笔保留确认月份、合同到期日期及整数分金额；兼容delay_days字段不参与完整合同排期。\n"
        return "# 月度经营节点：数量、单价、材料外协、库存及费用均事先声明。\n# 金额单位为元；设备产能为等效件/月；每月业务原因随节点保存。\n" + timing + "# 经营节点和未来阶段仅用于生成与人工核验，不作为模型输入。\n" + yaml.safe_dump(_normalise(dict(profile)), allow_unicode=True, sort_keys=False)
    # 新周期清单同样必须完整写入，不能丢失隐藏默认值。
    if "cycle_orders" in profile:
        # 排版只复制字段，不改变原参数。
        base = {key: value for key, value in profile.items() if key != "cycle_orders"}
        # 原设置保留，追加明确的替代主业说明。
        return render_annotated_profile(base) + "\n# 周期贸易订单替代旧主业销售与采购；原费用、税费及账务公式保留。\n# 金额为分，合同在生成前固定，未来计划不作为模型输入。\n" + yaml.safe_dump({"cycle_orders": _normalise(profile["cycle_orders"])}, allow_unicode=True, sort_keys=False)
    # 新增计划单独保存为中文说明段，不让旧渲染器静默丢字段。
    if "supplementary_orders" in profile:
        # 临时复制仅用于排版，原参数对象不修改。
        base = {key: value for key, value in profile.items() if key != "supplementary_orders"}
        # 历史字段继续复用原注释模板；订单清单完整追加。
        return render_annotated_profile(base) + "\n# 补充订单：先定业务清单，再计算余额；金额单位为分。\n# 规则、金额、忙闲日、实际日期和派生随机编号全部保存，禁止看余额补单。\n# 完整计划仅用于生成和核验，不作为模型可见特征。\n" + yaml.safe_dump({"supplementary_orders": _normalise(profile["supplementary_orders"])}, allow_unicode=True, sort_keys=False)
    if profile.get("schema_version") == "syn_b1_formal_profile_2_0":
        return _render_annotated_profile_v2(profile)
    return _render_annotated_profile_v1(profile)


def _render_annotated_profile_v1(profile: Mapping[str, Any]) -> str:
    """Render one effective profile with comments beside every MVP setting."""
    raw = _normalise(dict(profile))
    account = _mapping(raw, "account")
    initial = _mapping(raw, "initial_capital")
    operating = _mapping(raw, "operating")
    tax = _mapping(raw, "tax")
    transactions = _mapping(raw, "transactions")
    calendar = _mapping(raw, "calendar")

    lines = [
        "# SYN-B1 虚构制造企业参数方案（可直接用记事本修改并保存）",
        "# 每个冒号右侧是实际生效的值；# 开头的文字只是说明，不会参与生成。",
        "# 修改后请回到菜单依次选择“核验”和“预演”；核验通过才可正式生成。",
        "# 已用于正式生成的方案请通过菜单“复制为新版本”后再改，避免历史结果失去依据。",
        "",
        "# 文件版本：固定，不要修改。",
        f"schema_version: {_scalar(raw.get('schema_version'))}",
        "# 虚构企业编号：用于独立保存结果；建议只用字母、数字、下划线或中文。",
        f"sample_id: {_scalar(raw.get('sample_id'))}",
        "# 参数运行标识：同一企业的不同设定请改成新版本，例如 factory_001_v2。",
        f"run_id: {_scalar(raw.get('run_id'))}",
        "# 模拟开始/结束日期：YYYY-MM-DD；期间越长，生成的月度循环越多。",
        f"start_date: {_scalar(raw.get('start_date'))}",
        f"end_date: {_scalar(raw.get('end_date'))}",
        "# 随机种子：相同全部参数加相同种子，结果完全可重现；改种子只改变允许的随机波动。",
        f"random_seed: {_scalar(raw.get('random_seed'))}",
        "",
        "calendar:",
        "  # 预测截止日：不早于模拟结束日；只使用生成当时可知的公开日历。",
        f"  prediction_cutoff: {_scalar(calendar.get('prediction_cutoff'))}",
        "  # 银行日历版本：冻结技术值，不要修改，否则核验会拒绝。",
        f"  version: {_scalar(calendar.get('version'))}",
        "",
        _account_block(account),
        "",
        _initial_capital_block(initial),
        "",
        "# 后续股东注资：没有则保持 []。非空时必须预先写清日期和金额；生成器绝不临时猜测注资。",
        _yaml_block("additional_capital_plans", raw.get("additional_capital_plans", [])),
        "# 添加时请复制下列示例并去掉每行开头的#：capital_item_id为唯一编号；plan_version为版本；due_date为注资日；amount_cny为元。",
        "# - capital_item_id: \"capital_002\"",
        "#   plan_version: \"v1\"",
        "#   due_date: \"2026-06-01\"",
        "#   amount_cny: \"200000.00\"",
        "# 银行借款合同：没有借款就保持 []，不需要填0元假贷款。金额、日期、期限和还款方式必须预先声明。",
        _yaml_block("loan_contracts", raw.get("loan_contracts", [])),
        _loan_examples_block(),
        "",
        "operating:",
        "  # 每月销售确认额（元，不等于当月实际回款）：建议按企业体量填写；过高而回款慢会形成应收和资金压力。",
        f"  base_monthly_sales_cny: {_scalar(operating.get('base_monthly_sales_cny'))}",
        "  # 每月相对首月增加或减少多少百分比：例如5表示第1月为首月金额，第2月为首月的105%，第3月为110%，不按复利膨胀。",
        "  # 建议 -10 到 10；正数逐月增加，负数逐月减少。期间很长且负值过大时，后期销售额可能降到0以下，核验会拒绝。",
        f"  sales_monthly_growth_percent: {_scalar(operating.get('sales_monthly_growth_percent'))}",
        "  # 12个月季节性相对强弱：null表示不指定；启用时必须填满1至12月，程序会把12个月整体归一后再分配。",
        "  # 每月可填0至100；0表示该月无销售，100表示相对高峰。不要把它理解为必须合计100或1200。",
        _yaml_block("  sales_month_weights_percent", operating.get("sales_month_weights_percent")),
        _seasonality_example_block(),
        "  # 毛利率（百分比）：建议 5 到 60；越低采购成本越高，接近0会显著增加资金压力。",
        f"  gross_margin_percent: {_scalar(operating.get('gross_margin_percent'))}",
        "  # 生产周期（天）：建议 7 到 180；越长，资金被存货占用越久。",
        f"  production_cycle_days: {_scalar(operating.get('production_cycle_days'))}",
        "  # 目标存货天数：null表示按生产周期；填0表示不额外保有存货，较大值会提高采购与占资。",
        f"  inventory_target_days: {_scalar(operating.get('inventory_target_days'))}",
        "  # 销售后何时回款：delay_days是销售后延迟天数，share_percent是该批销售占比；各行比例必须合计100。",
        _schedule_block("  collection_schedule", operating.get("collection_schedule", [])),
        "  # 多段回款可复制示例：30%当月收、50%延后30天、20%延后60天。延迟越长，应收和资金压力越大。",
        "  # collection_schedule:",
        "  #   - delay_days: 0",
        "  #     share_percent: \"30\"",
        "  #   - delay_days: 30",
        "  #     share_percent: \"50\"",
        "  #   - delay_days: 60",
        "  #     share_percent: \"20\"",
        "  # 采购后何时付货款：各行比例必须合计100；账期越短，现金流出越早。",
        _schedule_block("  supplier_payment_schedule", operating.get("supplier_payment_schedule", [])),
        "  # 分期付采购款可复制示例：20%立即付、80%延后45天；比例合计必须为100。",
        "  # supplier_payment_schedule:",
        "  #   - delay_days: 0",
        "  #     share_percent: \"20\"",
        "  #   - delay_days: 45",
        "  #     share_percent: \"80\"",
        "  # 每月工资、房租水电和其他日常费用（元）：填0表示没有该项；极端高值可能使账户提前停止。",
        f"  monthly_payroll_cny: {_scalar(operating.get('monthly_payroll_cny'))}",
        f"  monthly_rent_and_utilities_cny: {_scalar(operating.get('monthly_rent_and_utilities_cny'))}",
        f"  monthly_other_operating_expense_cny: {_scalar(operating.get('monthly_other_operating_expense_cny'))}",
        "  # 三类日常费用的付款日期规则：delay_days为月初后延迟天数，各行比例必须合计100。",
        _schedule_block("  payroll_payment_schedule", operating.get("payroll_payment_schedule", [])),
        _schedule_block("  rent_and_utilities_payment_schedule", operating.get("rent_and_utilities_payment_schedule", [])),
        _schedule_block("  other_operating_expense_payment_schedule", operating.get("other_operating_expense_payment_schedule", [])),
        "  # 固定资产事项：没有则保持 []；如增加，必须同时写清购买金额、付款安排和折旧信息。",
        _yaml_block("  fixed_assets", operating.get("fixed_assets", [])),
        _fixed_asset_examples_block(),
        "",
        "tax:",
        "  # 所得税率（百分比）：建议按模拟情景填写；过高会增加应交税款和现金流出。",
        f"  income_tax_rate_percent: {_scalar(tax.get('income_tax_rate_percent'))}",
        "  # 模拟增值税综合实际税负率（百分比）：这是简化模拟，不建立发票和进项抵扣系统。",
        f"  vat_burden_rate_percent: {_scalar(tax.get('vat_burden_rate_percent'))}",
        "  # 附加税比例（百分比）：按模拟增值税计算；过高会放大税费现金流出。",
        f"  surcharge_rate_percent: {_scalar(tax.get('surcharge_rate_percent'))}",
        "  # 三种税费的扣款规则：每一类均可单独开关、设置周期和扣款日。",
        _tax_rules_block("  rules", tax.get("rules", {})),
        "",
        "transactions:",
        "  # 普通流水预设：冻结技术值，不要修改；真正可调的是下方笔数、集中度和金额波动。",
        f"  preset: {_scalar(transactions.get('preset'))}",
        "  default_policy:",
        "    # 每类事项在一个月内拆成多少笔：建议1到10；太高会变得很零碎，太低会过于集中。",
        f"    transaction_count: {_scalar(_mapping(transactions, 'default_policy').get('transaction_count'))}",
        "    # 月内集中方式：uniform为分散；其他可见预设以核验提示为准。",
        f"    month_concentration: {_scalar(_mapping(transactions, 'default_policy').get('month_concentration'))}",
        "    # 单笔金额允许波动（百分比）：建议0到30；0表示同类金额不波动，过高会削弱月度计划的可读性。",
        f"    amount_variation_percent: {_scalar(_mapping(transactions, 'default_policy').get('amount_variation_percent'))}",
        "  # 对个别资金类型另设拆分规则：没有则保持 {}；只填写系统支持的类别，先核验再生成。",
        _yaml_block("  category_overrides", transactions.get("category_overrides", {})),
        _transaction_override_example_block(),
        "",
    ]
    return "\n".join(lines)


def _render_annotated_profile_v2(profile: Mapping[str, Any]) -> str:
    """Reuse all v1 comments while replacing only the version-2 operating block."""
    raw = _normalise(deepcopy(dict(profile)))
    operating = _mapping(raw, "operating")
    regime = _mapping(operating, "regime_plan")
    segments = regime.get("segments") if isinstance(regime.get("segments"), list) else []
    first = segments[0] if segments and isinstance(segments[0], Mapping) else {}
    collection = _mapping(operating, "collection_policy")
    sales_seasonality = _mapping(operating, "sales_seasonality")
    procurement = _mapping(operating, "procurement_inertia")
    opening_expectation = _mapping(procurement, "opening_expectation")

    legacy = deepcopy(raw)
    legacy_operating = {
        key: value for key, value in operating.items()
        if key not in {"regime_plan", "sales_seasonality", "collection_policy", "procurement_inertia"}
    }
    legacy_operating.update({
        "base_monthly_sales_cny": regime.get("base_latent_monthly_sales_cny"),
        "sales_monthly_growth_percent": first.get("monthly_linear_change_percent", "0"),
        "sales_month_weights_percent": sales_seasonality.get("month_weights_percent"),
        "gross_margin_percent": first.get("gross_margin_percent", "0"),
        "collection_schedule": collection.get("base_schedule", []),
    })
    legacy["operating"] = legacy_operating
    rendered = _render_annotated_profile_v1(legacy)
    prefix, remainder = rendered.split("\noperating:\n", 1)
    _, tail = remainder.split("\ntax:\n", 1)

    lines = [
        "operating:",
        "  regime_plan:",
        "    # 经营形态：stable、sustained_growth、sustained_contraction、growth_to_contraction或contraction_to_growth。",
        f"    shape: {_scalar(regime.get('shape'))}",
        "    # 阶段声明覆盖月份YYYY-MM：必须覆盖生成期；参加前向模拟时还须预先覆盖前向月份。",
        f"    declared_through_month: {_scalar(regime.get('declared_through_month'))}",
        "    # 去季节化月销售锚点（元）：必须大于0；这是销售确认基础，不是当月实际回款。",
        f"    base_latent_monthly_sales_cny: {_scalar(regime.get('base_latent_monthly_sales_cny'))}",
        "    # 阶段按月份填写。变化率硬范围-20至20；毛利率-100至100并允许负值；阶段和毛利是真值，禁止入模。",
        _regime_segments_block(segments),
        "",
        "  sales_seasonality:",
        "    # 销售季节性控制各月卖多少；true时1至12月必须填全，硬范围0至300并归一到平均100。",
        f"    enabled: {_scalar(sales_seasonality.get('enabled'))}",
        "    # 70至130只是受控合成建议范围，不是行业典型值；false时必须全部填100。",
        _yaml_block("    month_weights_percent", sales_seasonality.get("month_weights_percent")),
        "",
        "  collection_policy:",
        "    # 基础回款安排：delay_days是销售后延迟天数，各档share_percent必须合计100。",
        _schedule_block("    base_schedule", collection.get("base_schedule", [])),
        "    seasonal_timing:",
        "      # 回款季节按销售月份调整每档延迟；硬范围-60至60天，只移动时间、不改变每批100%金额。",
        f"      enabled: {_scalar(_mapping(collection, 'seasonal_timing').get('enabled'))}",
        "      # -15至30天只是受控合成建议；调整后延迟不得为负，false时1至12月必须全部填0。",
        _yaml_block("      delay_adjustment_days_by_sales_month", _mapping(collection, "seasonal_timing").get("delay_adjustment_days_by_sales_month")),
        "",
        "  procurement_inertia:",
        "    opening_expectation:",
        "      # 固定方式：用首月销售成本的比例表达开局采购预期，避免猜绝对金额。",
        f"      mode: {_scalar(opening_expectation.get('mode'))}",
        "      # 硬范围0至300；100表示等于首月计划销售成本，80至120只是普通合成建议，禁止入模。",
        f"      opening_expected_cogs_ratio_percent: {_scalar(opening_expectation.get('opening_expected_cogs_ratio_percent'))}",
        "    # 每月向上月实际销售成本靠拢的速度：0至100；15-30慢、40-60中、75-100快，均非行业统计。",
        f"    adjustment_speed_percent: {_scalar(procurement.get('adjustment_speed_percent'))}",
        "",
        "  # 生产周期和目标存货天数共同影响库存；采购预期不会读取本月或未来销售。",
        f"  production_cycle_days: {_scalar(operating.get('production_cycle_days'))}",
        f"  inventory_target_days: {_scalar(operating.get('inventory_target_days'))}",
        "  # 供应商付款安排作用于已确认采购，各档比例必须合计100。",
        _schedule_block("  supplier_payment_schedule", operating.get("supplier_payment_schedule", [])),
        "  # 工资、房租水电和其他费用不随销售转折自动缩放；填0表示没有，过高可能触发资金不足停止。",
        f"  monthly_payroll_cny: {_scalar(operating.get('monthly_payroll_cny'))}",
        f"  monthly_rent_and_utilities_cny: {_scalar(operating.get('monthly_rent_and_utilities_cny'))}",
        f"  monthly_other_operating_expense_cny: {_scalar(operating.get('monthly_other_operating_expense_cny'))}",
        "  # 三类费用付款安排：各档比例合计100。",
        _schedule_block("  payroll_payment_schedule", operating.get("payroll_payment_schedule", [])),
        _schedule_block("  rent_and_utilities_payment_schedule", operating.get("rent_and_utilities_payment_schedule", [])),
        _schedule_block("  other_operating_expense_payment_schedule", operating.get("other_operating_expense_payment_schedule", [])),
        "  # 固定资产事项：没有则保持[]；新增时继续使用画像1.0已经冻结的逐字段结构和中文说明。",
        _yaml_block("  fixed_assets", operating.get("fixed_assets", [])),
    ]
    return prefix + "\n" + "\n".join(lines) + "\ntax:\n" + tail


def _regime_segments_block(segments: list[Any]) -> str:
    """Render every editable segment field with an adjacent Chinese guide."""

    lines = ["    segments:"]
    for index, value in enumerate(segments, start=1):
        segment = value if isinstance(value, Mapping) else {}
        boundary = "第一段必须与start_date同月" if index == 1 else "第二段开始月就是转折月"
        lines.extend((
            f"      # 第{index}段：segment_id是仅供追溯的唯一阶段编号，不得与其他阶段重复。",
            f"      - segment_id: {_scalar(segment.get('segment_id'))}",
            f"        # start_month使用YYYY-MM；{boundary}，并且不得晚于declared_through_month。",
            f"        start_month: {_scalar(segment.get('start_month'))}",
            "        # monthly_linear_change_percent是相对本段起点的每月非复利线性变化；正数为增长、负数为收缩，硬范围-20至20。",
            f"        monthly_linear_change_percent: {_scalar(segment.get('monthly_linear_change_percent'))}",
            "        # gross_margin_percent是本段销售毛利率，硬范围-100至100；负值表示销售成本高于销售收入。",
            f"        gross_margin_percent: {_scalar(segment.get('gross_margin_percent'))}",
        ))
    if not segments:
        lines.append("      # 至少填写一个阶段；转折形态必须填写两个阶段。")
        lines.append("      []")
    return "\n".join(lines)


def _account_block(account: Mapping[str, Any]) -> str:
    """Render the active route plus a copy-ready complete secondary-account example."""

    lines = [
        "account:",
        "  # 当前账户角色：primary（主账户）或secondary（次要账户）。",
        f"  role: {_scalar(account.get('role'))}",
        "  # primary_all_v1表示13类资金全部经过当前主账户；次要账户请把routing_preset改为null并完整填写routing_rules。",
        f"  routing_preset: {_scalar(account.get('routing_preset'))}",
    ]
    if isinstance(account.get("routing_rules"), Mapping):
        lines.append(_yaml_block("  routing_rules", account.get("routing_rules")))
    else:
        lines.append("  # routing_rules: 主账户使用上述预设时无需填写。")
    lines.extend((
        "  # 次要账户完整路由示例（复制后去掉#）：enabled控制该类是否经过本账户，routed_share_percent控制经过比例。",
        "  # 比例范围0至100；0表示当前账户完全看不到，100表示全部经过。部分比例只改变当前账户可见金额，不改变企业整体事项。",
        "  # routing_rules:",
        "  #   sales_collection: {enabled: true, routed_share_percent: \"40\"}",
        "  #   supplier_payment: {enabled: true, routed_share_percent: \"20\"}",
        "  #   payroll: {enabled: false, routed_share_percent: \"0\"}",
        "  #   rent_and_utilities: {enabled: true, routed_share_percent: \"100\"}",
        "  #   bank_loan_drawdown: {enabled: false, routed_share_percent: \"0\"}",
        "  #   bank_loan_repayment: {enabled: false, routed_share_percent: \"0\"}",
        "  #   bank_interest_payment: {enabled: false, routed_share_percent: \"0\"}",
        "  #   shareholder_funding: {enabled: true, routed_share_percent: \"100\"}",
        "  #   fixed_asset_payment: {enabled: false, routed_share_percent: \"0\"}",
        "  #   corporate_income_tax_payment: {enabled: true, routed_share_percent: \"100\"}",
        "  #   simulated_vat_payment: {enabled: true, routed_share_percent: \"100\"}",
        "  #   surcharge_payment: {enabled: true, routed_share_percent: \"100\"}",
        "  #   other_declared_event: {enabled: false, routed_share_percent: \"0\"}",
    ))
    return "\n".join(lines)


def _initial_capital_block(initial: Mapping[str, Any]) -> str:
    """Render either manual or random capital without producing contradictory keys."""

    lines = [
        "initial_capital:",
        "  # 首次注资编号和版本：只用于追溯，同一方案内不要重复。",
        f"  capital_item_id: {_scalar(initial.get('capital_item_id'))}",
        f"  plan_version: {_scalar(initial.get('plan_version'))}",
        "  # manual为固定金额；profile_random为在上下限之间按种子抽取一次。新企业必须有且仅有一种首次注资方式。",
        f"  mode: {_scalar(initial.get('mode'))}",
    ]
    if initial.get("mode") == "profile_random":
        lines.extend((
            "  # 随机下限/上限（元）必须都大于0且下限不大于上限；范围越宽，不同种子的起步资金差异越大。",
            f"  minimum_amount_cny: {_scalar(initial.get('minimum_amount_cny'))}",
            f"  maximum_amount_cny: {_scalar(initial.get('maximum_amount_cny'))}",
            "  # 首版只支持uniform（上下限之间均匀抽取）；profile_version用于追溯该随机画像。",
            f"  distribution: {_scalar(initial.get('distribution'))}",
            f"  profile_version: {_scalar(initial.get('profile_version'))}",
        ))
    else:
        lines.extend((
            "  # 固定起步资金（元）：必须大于0；太低会在付不起下一笔流出前停止，太高会使余额看起来过度充裕。",
            f"  amount_cny: {_scalar(initial.get('amount_cny'))}",
        ))
    lines.extend((
        "  # 随机首次注资可复制示例（启用时删除amount_cny，并用下列5行替换mode及金额行）：",
        "  # mode: \"profile_random\"",
        "  # minimum_amount_cny: \"500000.00\"",
        "  # maximum_amount_cny: \"1500000.00\"",
        "  # distribution: \"uniform\"",
        "  # profile_version: \"capital_random_v1\"",
    ))
    return "\n".join(lines)


def _loan_examples_block() -> str:
    """Explain every frozen loan form with examples that remain inert comments."""

    common = (
        "# 以下四份合同示例全部被#注释，不会生效。需要哪一种，就复制一整份到loan_contracts:下面并去掉每行#。",
        "# principal_cny必须大于0；annual_interest_rate_percent允许0至100，但常见情景建议0至15，极高值会明显增加利息和停表风险。",
        "# term_days必须为正整数；也可删掉term_days改填maturity_date: \"YYYY-MM-DD\"，两者只能选一个。期限越短，还款越集中。",
        "# funding_kind可填bank_loan（银行借款）或shareholder_loan（股东借款）；MVP不模拟循环额度、提前还本或自动续贷。",
        "# 示例一：到期一次还本付息",
        "# - loan_contract_id: \"loan_bullet_001\"",
        "#   contract_version: \"v1\"",
        "#   funding_kind: \"bank_loan\"",
        "#   draw_date: \"2026-02-01\"",
        "#   principal_cny: \"\"",
        "#   annual_interest_rate_percent: \"\"",
        "#   repayment_method: \"bullet_principal_and_interest\"",
        "#   term_days: null",
        "# 示例二：每月付息，到期还本金",
        "# - loan_contract_id: \"loan_monthly_interest_001\"",
        "#   contract_version: \"v1\"",
        "#   funding_kind: \"bank_loan\"",
        "#   draw_date: \"2026-02-01\"",
        "#   principal_cny: \"\"",
        "#   annual_interest_rate_percent: \"\"",
        "#   repayment_method: \"monthly_interest_bullet_principal\"",
        "#   term_days: null",
        "# 示例三：每月等额本金，利息随剩余本金下降",
        "# - loan_contract_id: \"loan_equal_principal_001\"",
        "#   contract_version: \"v1\"",
        "#   funding_kind: \"bank_loan\"",
        "#   draw_date: \"2026-02-01\"",
        "#   principal_cny: \"\"",
        "#   annual_interest_rate_percent: \"\"",
        "#   repayment_method: \"equal_principal_monthly\"",
        "#   term_days: null",
        "# 示例四：每月等额还款（每期本金加利息的合计尽量相同）",
        "# - loan_contract_id: \"loan_equal_payment_001\"",
        "#   contract_version: \"v1\"",
        "#   funding_kind: \"bank_loan\"",
        "#   draw_date: \"2026-02-01\"",
        "#   principal_cny: \"\"",
        "#   annual_interest_rate_percent: \"\"",
        "#   repayment_method: \"equal_payment_monthly\"",
        "#   term_days: null",
    )
    return "\n".join(common)


def _seasonality_example_block() -> str:
    return "\n".join((
        "  # 季节性可复制示例：春节附近较低、年末较高。数值仅表示月份之间的相对强弱。",
        "  # sales_month_weights_percent:",
        "  #   \"1\": \"60\"",
        "  #   \"2\": \"40\"",
        "  #   \"3\": \"75\"",
        "  #   \"4\": \"80\"",
        "  #   \"5\": \"85\"",
        "  #   \"6\": \"90\"",
        "  #   \"7\": \"90\"",
        "  #   \"8\": \"95\"",
        "  #   \"9\": \"100\"",
        "  #   \"10\": \"95\"",
        "  #   \"11\": \"100\"",
        "  #   \"12\": \"100\"",
    ))


def _fixed_asset_examples_block() -> str:
    return "\n".join((
        "  # 固定资产金额必须大于0；付款比例合计100。金额越大、付款越早，账户资金压力越大，付不起时会停止并出事实单。",
        "  # asset_type只支持equipment（设备）或factory_or_construction（厂房/建设）。达到可使用状态后才需要寿命和残值率。",
        "  # 设备分期付款可复制示例：",
        "  # fixed_assets:",
        "  #   - event_id: \"equipment_001\"",
        "  #     asset_type: \"equipment\"",
        "  #     purchase_month: \"2026-04\"",
        "  #     purchase_amount_cny: \"300000.00\"",
        "  #     payment_schedule:",
        "  #       - {delay_days: 0, share_percent: \"30\"}",
        "  #       - {delay_days: 60, share_percent: \"70\"}",
        "  #     ready_for_use_date: \"2026-05-01\"",
        "  #     useful_life_months: 60",
        "  #     residual_value_percent: \"5\"",
        "  # 厂房/建设事项写法相同，只需使用新的event_id并把asset_type改为factory_or_construction。",
    ))


def _transaction_override_example_block() -> str:
    return "\n".join((
        "  # 单类覆盖可复制示例；类别不写时继续使用default_policy。",
        "  # month_concentration可填uniform（分散）、early（月初集中）、middle（月中集中）、late（月末集中）。",
        "  # transaction_count必须大于0；amount_variation_percent允许0至99.99，建议0至30，极大值会造成很小与很大单笔并存，但月度总额不变。",
        "  # category_overrides:",
        "  #   sales_collection:",
        "  #     transaction_count: 6",
        "  #     month_concentration: \"middle\"",
        "  #     amount_variation_percent: \"20\"",
        "  #   supplier_payment:",
        "  #     transaction_count: 3",
        "  #     month_concentration: \"late\"",
        "  #     amount_variation_percent: \"10\"",
        "  # 注意：借款、还款、利息、注资和税费使用合同或税费日期，不能被普通流水拆分规则改写。",
    ))


def _yaml_block(key: str, value: Any) -> str:
    if value is None:
        return f"{key}: null"
    rendered = yaml.safe_dump(_normalise(value), allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip()
    if rendered in {"[]", "{}", "null"}:
        return f"{key}: {rendered}"
    indentation = " " * (len(key) - len(key.lstrip()) + 2)
    indented = "\n".join(f"{indentation}{line}" for line in rendered.splitlines())
    return f"{key}:\n{indented}"


def _schedule_block(key: str, value: Any) -> str:
    values = value if isinstance(value, list) else []
    if not values:
        return f"{key}: []"
    indentation = " " * (len(key) - len(key.lstrip()) + 2)
    lines = [f"{key}:"]
    for item in values:
        row = item if isinstance(item, Mapping) else {}
        lines.append(f"{indentation}- delay_days: {_scalar(row.get('delay_days'))}")
        lines.append(f"{indentation}  share_percent: {_scalar(row.get('share_percent'))}")
    return "\n".join(lines)


def _tax_rules_block(key: str, value: Any) -> str:
    rules = value if isinstance(value, Mapping) else {}
    indentation = " " * (len(key) - len(key.lstrip()) + 2)
    nested = indentation + "  "
    lines = [f"{key}:"]
    for name in ("corporate_income_tax", "simulated_vat", "surcharge"):
        rule = rules.get(name) if isinstance(rules.get(name), Mapping) else {}
        title = {
            "corporate_income_tax": "企业所得税",
            "simulated_vat": "模拟增值税",
            "surcharge": "附加税",
        }[name]
        lines.extend((
            f"{indentation}# {title}",
            f"{indentation}{name}:",
            (
                f"{nested}# 是否扣款：所得税为保证利润表和所得税闭环必须保持true；若不从当前账户扣，请改payer_account_role。"
                if name == "corporate_income_tax"
                else f"{nested}# 是否扣款：true为生成该税费，false为不生成；关闭会减少流出。"
            ),
            f"{nested}enabled: {_scalar(rule.get('enabled'))}",
            f"{nested}# 扣款周期：monthly（月度）或quarterly（季度）；季度会让单次流出更集中。",
            f"{nested}frequency: {_scalar(rule.get('frequency'))}",
            f"{nested}# 日期规则：recurring_day为每期固定日；explicit_period_dates为逐期明确日期。",
            f"{nested}schedule_mode: {_scalar(rule.get('schedule_mode'))}",
            f"{nested}# 扣款经过的账户：primary或secondary；必须与当前账户路由设定一致。",
            f"{nested}payer_account_role: {_scalar(rule.get('payer_account_role'))}",
            f"{nested}# 非工作日处理：next_bank_workday为顺延；keep_calendar_date为保留原日。",
            f"{nested}non_bank_day_policy: {_scalar(rule.get('non_bank_day_policy'))}",
        ))
        if rule.get("schedule_mode") == "explicit_period_dates":
            lines.extend((
                f"{nested}# 逐期明确日期：键为2026-01等月份或2026-Q1等季度，值为实际扣款日。期间内每一期都要填写。",
                _yaml_block(f"{nested}explicit_period_dates", rule.get("explicit_period_dates")),
            ))
        else:
            lines.extend((
                f"{nested}# 每期扣款日：范围1至31，建议1至28；太靠月末可能与工资、采购或还款叠加。",
                f"{nested}recurring_debit_day: {_scalar(rule.get('recurring_debit_day'))}",
            ))
        lines.extend((
            f"{nested}# 明确日期模式示例（启用时把schedule_mode改为explicit_period_dates，删除recurring_debit_day，再填完整期间）：",
            f"{nested}# explicit_period_dates:",
            f"{nested}#   \"2026-Q1\": \"2026-04-20\"",
            f"{nested}#   \"2026-Q2\": \"2026-07-20\"",
            f"{nested}#   \"2026-Q3\": \"2026-10-20\"",
            f"{nested}#   \"2026-Q4\": \"2027-01-20\"",
        ))
    return "\n".join(lines)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, Mapping) else {}


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _yaml_error_line(error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    return f"第{mark.line + 1}行" if mark is not None else "请检查缩进和冒号"
