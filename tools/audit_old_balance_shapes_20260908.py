# 本次按用户要求只统计旧样本实际余额，不生成新流水或重做模型测试。
"""只读统计原260家余额的长期走势和10/30日局部变化。"""
# JSON用于读取已冻结的文件清单并保存本次结果。
import json
# 哈希用于核对日表字节是否与原生成时一致。
import hashlib
# 系统模块配置中文终端输出。
import sys
# 路径对象明确本轮输入与全新输出位置。
from pathlib import Path
# 十进制金额只用于精确检查日度余额勾稽。
from decimal import Decimal
# 数组计算批量描述实际轨迹，不调用训练工具。
import numpy as np
# 表格工具读取本地合成CSV并按企业汇总。
import pandas as pd
# 图形库导出静态图片供直接看结论。
import matplotlib
# 不弹图形窗口，输出只保存为文件。
matplotlib.use("Agg")
# 导入正式绘图接口。
import matplotlib.pyplot as plt
# 日期格式用于少量固定编号的余额示例。
import matplotlib.dates as mdates
# 由脚本定位唯一主项目。
ROOT = Path(__file__).resolve().parents[1]
# 旧数据严格只读，本次结果写入独立子目录。
DATA = ROOT / "data" / "synthetic_generated" / "syn_b1_v2"
# 本次统计输出不能覆盖旧研究目录或报告。
OUT = ROOT / "data" / "synthetic_generated" / "balance_review_20260908" / "old_260_audit_v1"
# 可读材料与新候选360户工作分开保存。
DISPLAY = ROOT / "研究材料" / "20260908_旧样本统计与新样本设计"
# 既有清单的五种标签描述销售，不事先当成余额标签。
OLD_SHAPES = {"stable": "销售稳定", "sustained_growth": "销售增长", "sustained_contraction": "销售收缩", "growth_to_contraction": "销售先升后降", "contraction_to_growth": "销售先降后升"}
# 本次描述实际账户余额的七类名称。
SHAPES = ["整体平稳", "持续上行", "持续下行", "先升后降", "先降后升", "反复起伏", "其他混合"]
# 图中文字使用当前系统已有的微软雅黑。
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 11})
# 中文结果直接显示，避免命令行编码影响阅读。
sys.stdout.reconfigure(encoding="utf-8")

# JSON新文件只能创建一次，以便保留运行证据。
def save_json(path, value):
    # 遇到已有文件就报错，不静默覆盖旧统计。
    with path.open("x", encoding="utf-8") as handle:
        # 缩进和中文原文方便人工核验。
        json.dump(value, handle, ensure_ascii=False, indent=2)

# 判断一组余额是否先完成一段明显变动再反向明显变动。
def swings(values, threshold):
    # 初始尚未形成方向，同时保留低点和高点。
    low, high, direction, initial, turns = float(values[0]), float(values[0]), 0, 0, 0
    # 逐点看金额变化，只有达到固定幅度才确认方向或转折。
    for value in values[1:]:
        # 尚无明确方向时更新最初区间。
        if direction == 0:
            # 同时记录截至此刻的最低和最高余额。
            low, high = min(low, value), max(high, value)
            # 从此前低点上涨达到门槛才算建立上行。
            if value - low >= threshold:
                # 初始方向记录一次，后续用于区分高点与低点。
                direction, initial, high = 1, 1, value
            # 从此前高点下跌达到门槛才算建立下行。
            elif high - value >= threshold:
                # 记录首次下行方向。
                direction, initial, low = -1, -1, value
        # 已经上行时继续寻找最高点及后来的真实下降。
        elif direction == 1:
            # 更新这段上涨达到的最高余额。
            high = max(high, value)
            # 从最高点下降足够金额，才算一次转折。
            if high - value >= threshold:
                # 翻转方向并从此点继续记录下降。
                direction, low, turns = -1, value, turns + 1
        # 已经下行时对称检查低点后的回升。
        else:
            # 更新本次下跌最低余额。
            low = min(low, value)
            # 从低点回升足够金额，确认方向改变。
            if value - low >= threshold:
                # 同一转折只计一次，随后进入上涨状态。
                direction, high, turns = 1, value, turns + 1
    # 返回次数和首次方向，供统计而非预测使用。
    return turns, initial

# 从季度平均余额判断长期形状，不让单日付款决定两年标签。
def long_shape(quarters, reference):
    # 季度高低差小于典型余额10%时才标整体平稳。
    if np.ptp(quarters) <= 0.10 * reference:
        # 这里平稳只描述本轮账户，不代表销售也平稳。
        return "整体平稳"
    # 长期转折的两侧变动均须达到典型余额10%。
    turns, initial = swings(quarters, 0.10 * reference)
    # 多次转折如实登记为反复起伏。
    if turns >= 2:
        # 多次上下波动与单次转折分开。
        return "反复起伏"
    # 一次转折按首次方向决定是先升还是先降。
    if turns == 1:
        # 返回可与图上季度方向核对的名称。
        return "先升后降" if initial == 1 else "先降后升"
    # 未出现明显反转时，以起末季度变化判断累计方向。
    if quarters[-1] - quarters[0] > 0.10 * reference:
        # 仍允许其间有不足10%的小回撤。
        return "持续上行"
    # 下行用完全对称的金额门槛。
    if quarters[0] - quarters[-1] > 0.10 * reference:
        # 长期下降不得由销售标签直接决定。
        return "持续下行"
    # 不强行把难归类的轨迹塞入期望类型。
    return "其他混合"

# 主过程先锁定原200家加60家清单，再逐户只读计算。
def main():
    # 创建独立输出目录；重跑必须换版本而非覆盖。
    OUT.mkdir(parents=True, exist_ok=False)
    # 面向用户的说明和图片放在主项目研究材料中。
    DISPLAY.mkdir(parents=True, exist_ok=True)
    # 明确引用两批旧生成结果，不把试生成目录凑入260家。
    batches = [("_pretrain_formal_200x24m_v1", "p0_formal_24m_v1", "原200家"), ("_pretrain_formal_high_revenue_60x24m_v1", "p0_formal_high_revenue_24m_v1", "后增60家")]
    # 暂存逐户汇总和日期窗口汇总，不改写任何日表。
    records, windows, sources, frames = [], [], [], {}
    # 开始前把公式和样本边界写明，避免看结果后选阈值。
    save_json(OUT / "统计前方法登记.json", {"企业范围": "原200家及后增60家，全部260家；只描述旧数据，不重算模型成绩", "开业处理": "长期形状及局部窗口从第91个自然日开始；完整两年结果另列", "比较金额": "每户第91日以后所有自然日日终余额的中位数；仅供事后描述，不是训练特征", "长期形状": "开业后第4至24月共7个季度的平均日余额；明显变动门槛为该户比较金额10%", "局部窗口": "每个可用银行工作日起点到未来10或30个银行工作日，包含起点共11或31个余额", "局部幅度": "窗口内最高余额减最低余额，再除以该户比较金额", "局部转折": "先上涨再下降或先下降再上涨，两侧均达到门槛", "局部门槛": [0.02, 0.05, 0.10, 0.25], "汇总顺序": "先逐户统计比例，再对企业等权平均；重叠窗口不是独立企业"})
    # 两批企业使用各自正式运行编号。
    for folder, run_id, group in batches:
        # 画像清单只用于核对旧设计分类，不作为新模型输入。
        allocation_path = DATA / folder / "profile_allocation_manifest.csv"
        # 读取清单并保留中文与空值。
        allocation = pd.read_csv(allocation_path, keep_default_na=False)
        # 每个冻结清单有200或60户，异常立即阻断统计。
        assert len(allocation) == (200 if group == "原200家" else 60)
        # 所有企业都进入统计，不能只挑看起来平稳的样本。
        for entry in allocation.to_dict("records"):
            # 由已保存企业号构造原正式运行目录。
            sid = entry["sample_id"]
            # 明确这户唯一日表路径。
            directory = DATA / sid / run_id
            # 读取原始清单中该日表的冻结哈希。
            manifest = json.loads((directory / "generation_manifest.json").read_text(encoding="utf-8"))
            # 对每户原始日表先算哈希再读数据。
            daily_path = directory / "account_daily_total.csv"
            # 保存当前字节指纹用于最后再次检查。
            digest = hashlib.sha256(daily_path.read_bytes()).hexdigest()
            # 当前数据必须仍等于生成时的原始数据。
            assert digest == manifest["files"]["account_daily_total.csv"], sid
            # 以字符串读取，金额检查到分不经浮点近似。
            raw = pd.read_csv(daily_path, dtype=str, keep_default_na=False)
            # 将元转成分并在源表上核对每日期末余额。
            exact = {c: raw[c].map(lambda x: int(Decimal(x) * 100)) for c in ["inflow_cny", "outflow_cny", "ending_balance_cny"]}
            # 从零余额加每天流入减流出，必须逐日精确匹配。
            assert np.array_equal((exact["inflow_cny"] - exact["outflow_cny"]).cumsum(), exact["ending_balance_cny"])
            # 两年730个自然日必须完整连续。
            dates = pd.to_datetime(raw.calendar_date)
            # 自然日覆盖不能因无交易日而减少。
            assert len(raw) == 730 and dates.iloc[0] == pd.Timestamp("2025-01-01") and dates.iloc[-1] == pd.Timestamp("2026-12-31") and dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
            # 单独保存日期索引，方便季度及自然日统计。
            frame = raw.copy()
            # 用普通元金额只做形状比例计算，不改原CSV。
            frame["balance"] = exact["ending_balance_cny"].to_numpy() / 100
            # 日期转换为索引后不再依赖字符串排序。
            frame.index = dates
            # 第91日起诊断，去掉首次注资和建库存的启动阶段。
            mature = frame.iloc[90:]
            # 每户典型余额固定，不在每个窗口换不同分母。
            reference = max(float(mature.balance.median()), 0.01)
            # 七个完整季度的日终余额均值可直接复算。
            quarters = mature.balance.resample("QS").mean().to_numpy()
            # 日历标记来自原日表，绝不重新推断银行工作日。
            bank = mature.loc[mature.is_bank_workday.str.lower() == "true"]
            # 在完整原表算差分后再取成熟期间，保留第91日真实变化。
            flat = frame.balance.diff().loc[mature.index].eq(0)
            # 每家形成一行明确可解释的指标。
            row = {"企业编号": sid, "原批次": group, "原始分组": entry["split_group"], "原销售形状": OLD_SHAPES[entry["shape"]], "实际长期余额形状": long_shape(quarters, reference), "典型余额元": reference, "全期交易笔数": int(raw.transaction_count.astype(int).sum()), "全期有交易自然日": int(raw.transaction_count.astype(int).gt(0).sum()), "全期余额不变日比例": float(frame.balance.diff().iloc[1:].eq(0).mean()), "成熟期无交易银行工作日比例": float(bank.transaction_count.astype(int).eq(0).mean()), "成熟期余额不变自然日比例": float(flat.mean()), "成熟期余额不变银行工作日比例": float(frame.balance.diff().loc[bank.index].eq(0).mean()), "成熟期活跃日平均笔数": float(bank.loc[bank.transaction_count.astype(int).gt(0)].transaction_count.astype(int).mean()), "首季度均值元": float(quarters[0]), "末季度均值元": float(quarters[-1]), "长期起末变化比例": float((quarters[-1] - quarters[0]) / reference), "长期季度高低差比例": float(np.ptp(quarters) / reference)}
            # 最长连续不变的自然日长度含区段第一个余额点。
            lengths = frame.balance.groupby(frame.balance.diff().ne(0).cumsum()).size()
            # 这项帮助解释为什么短图常看成水平线。
            row["最长连续相同余额自然日"] = int(lengths.max())
            # 保存全量逐户结果，不用聚合数掩盖企业差异。
            records.append(row)
            # 每户的历史销售设计和新余额分类可以交叉核对。
            frames[sid] = frame
            # 登记旧日表的路径和校验指纹。
            sources.append({"企业编号": sid, "相对位置": daily_path.relative_to(ROOT).as_posix(), "原日表SHA256": digest})
            # 原银行日顺序作为滚动10和30日唯一时间线。
            values = bank.balance.to_numpy()
            # 两种期限都按所有起点统计，不挑某个月。
            for horizon in [10, 30]:
                # 每个窗口包含当日起点和horizon个未来工作日。
                array = np.lib.stride_tricks.sliding_window_view(values, horizon + 1)
                # 最高减最低刻画整段是否真有波动。
                amplitudes = np.ptp(array, axis=1) / reference
                # 终点变化与路径波动分开，否则先升后降容易被误判平稳。
                endpoints = (array[:, -1] - array[:, 0]) / reference
                # 四档门槛并列，避免单一5%定义左右结论。
                for threshold in [0.02, 0.05, 0.10, 0.25]:
                    # 每段按相同固定金额门槛检查可见反向变化。
                    reversals = np.array([swings(v, threshold * reference)[0] >= 1 for v in array])
                    # 企业内先算各项比例，随后企业等权汇总。
                    windows.append({"企业编号": sid, "原批次": group, "期限工作日": horizon, "门槛比例": threshold, "窗口数": len(array), "整段幅度小于等于门槛比例": float((amplitudes <= threshold).mean()), "终点上涨超过门槛比例": float((endpoints > threshold).mean()), "终点下降超过门槛比例": float((endpoints < -threshold).mean()), "终点接近起点比例": float((np.abs(endpoints) <= threshold).mean()), "先变动再明显反向比例": float(reversals.mean())})
    # 明确唯一企业共260家，防止重复运行目录混入。
    assert len(records) == 260 and len({r["企业编号"] for r in records}) == 260
    # 全量和208户开发子集都单独展示，旧52户在本次只做数据描述。
    enterprise = pd.DataFrame(records)
    # 保存逐户结果，读者能从总体数字追到具体账户。
    enterprise.to_csv(OUT / "260家逐户余额统计.csv", index=False, encoding="utf-8-sig")
    # 保存企业内比例，重叠窗口不当独立样本。
    window_table = pd.DataFrame(windows)
    # 原始窗口统计保留四档门槛。
    window_table.to_csv(OUT / "逐户10日30日四档统计.csv", index=False, encoding="utf-8-sig")
    # 锁定当前研究使用过的208户开发清单，便于分开查看。
    dev_path = ROOT / "公开材料" / "GitHub公开仓库" / "enterprise-balance-forecast-demo" / "public_materials" / "evidence" / "source_extracts" / "development_input_manifest.csv"
    # 本表只用于分组，不读取或重新计算模型预测。
    dev_ids = set(pd.read_csv(dev_path).enterprise_id)
    # 开发企业数量和旧总体交集必须准确。
    assert len(dev_ids) == 208 and dev_ids.issubset(set(enterprise["企业编号"]))
    # 对全量、原200、后增60和开发208分别给结论。
    groups = {"全部260家": enterprise, "原200家": enterprise.loc[enterprise["原批次"] == "原200家"], "后增60家": enterprise.loc[enterprise["原批次"] == "后增60家"], "开发208家": enterprise.loc[enterprise["企业编号"].isin(dev_ids)]}
    # 汇总结果仅包含普通数字和数量，容易阅读。
    summary = {}
    # 逐组按企业等权统计。
    for group, table in groups.items():
        # 明确只选择当前组的企业内部比例。
        subset = window_table.loc[window_table["企业编号"].isin(table["企业编号"])]
        # 几个常见期限门槛按组内企业等权平均。
        averaged = subset.groupby(["期限工作日", "门槛比例"])[["整段幅度小于等于门槛比例", "终点上涨超过门槛比例", "终点下降超过门槛比例", "终点接近起点比例", "先变动再明显反向比例"]].mean().reset_index()
        # 同时保留典型企业的笔数和最长水平段长度。
        summary[group] = {"企业数": len(table), "全期交易笔数合计": int(table["全期交易笔数"].sum()), "每家两年交易笔数中位数": float(table["全期交易笔数"].median()), "实际长期余额形状": table["实际长期余额形状"].value_counts().to_dict(), "原销售形状": table["原销售形状"].value_counts().to_dict(), "银行工作日无交易比例_企业平均": float(table["成熟期无交易银行工作日比例"].mean()), "银行工作日余额不变比例_企业平均": float(table["成熟期余额不变银行工作日比例"].mean()), "最长水平段天数_企业中位数": float(table["最长连续相同余额自然日"].median()), "不同期限不同门槛": averaged.to_dict("records")}
    # 交叉表直接揭示销售形状与余额形状是否对应。
    pd.crosstab(enterprise["原销售形状"], enterprise["实际长期余额形状"]).to_csv(OUT / "销售设计与实际余额形状对照.csv", encoding="utf-8-sig")
    # 本次最终输出之前重新读取日表指纹以确认只读完成。
    for source in sources:
        # 任何输入被修改都终止交付。
        assert hashlib.sha256((ROOT / source["相对位置"]).read_bytes()).hexdigest() == source["原日表SHA256"]
    # 保存全部来源定位，便于以后按文件重算。
    pd.DataFrame(sources).to_csv(OUT / "只读来源核对清单.csv", index=False, encoding="utf-8-sig")
    # JSON汇总作为说明文件唯一数字来源。
    save_json(OUT / "汇总结论.json", summary)
    # 长期形状分组画一张简单的水平柱状图。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    # 第一块图说明全260家并非都只有平稳余额。
    counts = enterprise["实际长期余额形状"].value_counts().reindex(SHAPES, fill_value=0)
    # 使用统一颜色以便关注实际家数。
    bars = axes[0].barh(SHAPES, counts, color="#235782")
    # 柱末直接标家数，无需读者估算。
    axes[0].bar_label(bars, fmt="%.0f家", padding=4)
    # 留白确保数字不被边界截断。
    axes[0].set(xlim=(0, counts.max() * 1.2), xlabel="企业数", title="完整期间：实际余额长期形状")
    # 按读者通常从上到下的顺序列类别。
    axes[0].invert_yaxis()
    # 第二块对比原200和后60的实际交易密度。
    densities = [summary[g]["银行工作日无交易比例_企业平均"] * 100 for g in ["原200家", "后增60家", "全部260家"]]
    # 三组统一0至100百分比，避免视觉放大。
    bars = axes[1].bar(["原200家", "后增60家", "全部260家"], densities, color=["#b86a31", "#5b8d81", "#235782"])
    # 直接把无交易比例标在柱上。
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4)
    # 写明是无交易，而非模型预测不变。
    axes[1].set(ylim=(0, 100), ylabel="银行工作日没有交易的比例", title="为什么看几天常像一条水平线")
    # 全图下方简短记录口径，细节见Markdown。
    fig.suptitle("原260家样本只读统计｜开业前三个月单独剔除后观察\n长期形状看季度均值；右图先逐户统计，再让每家企业同等分量", fontsize=13)
    # 保存一张可直接查看的汇总图。
    fig.savefig(DISPLAY / "01_旧260家余额形状与交易密度.png", dpi=150)
    # 释放图形资源。
    plt.close(fig)
    # 原200与后60各取固定前三号，只用于示例而不代表总体。
    selected = ["syn_b1_p0_001", "syn_b1_p0_002", "syn_b1_p0_003", "syn_b1_p0_201", "syn_b1_p0_202", "syn_b1_p0_203"]
    # 用三行两列提供六户全期曲线，满足先看少量原样本的需求。
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    # 依编号确定，不按看完结果的好坏挑选。
    for ax, sid in zip(axes.flat, selected):
        # 读取刚才内存中的原日度表，源路径已登记。
        frame = frames[sid]
        # 只换算元到万元，保留每个自然日。
        ax.plot(frame.index, frame.balance / 10000, linewidth=1, color="#235782")
        # 标题写实际余额形状而非销售设计。
        label = enterprise.set_index("企业编号").loc[sid, "实际长期余额形状"]
        # 全图零起点让不同企业的绝对变化幅度可识别。
        ax.set(title=f"{sid}｜{label}", ylabel="每日余额（万元）", ylim=(0, None))
        # 半年刻度避免日期重叠。
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        # 月份和年份足够表达完整两年。
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        # 淡网格方便追踪每日金额。
        ax.grid(alpha=0.2)
    # 明确六例只是固定编号，不能替代260家统计。
    fig.suptitle("旧样本完整余额曲线：原200家和后60家各取编号前3户\n全部是原日表逐日余额；未重新生成、未修改，金额单位为万元", fontsize=14)
    # 保留图上依据所对应的原始文件位置在来源清单。
    fig.savefig(DISPLAY / "02_旧样本固定六户完整余额.png", dpi=150)
    # 关闭最后一张图。
    plt.close(fig)
    # 输出完整汇总便于写最终通俗结论。
    print(json.dumps(summary, ensure_ascii=False, indent=2))

# 直接调用时才开始只读统计。
if __name__ == "__main__":
    # 不提供任何训练或新生成分支。
    main()
