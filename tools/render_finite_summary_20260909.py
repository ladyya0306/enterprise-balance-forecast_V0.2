# 声明本工具只读取既有日表并汇总绘图。
"""从有限已生成CSV制作9面板和6面板余额汇总图。"""

# 导入命令行解析器以要求明确验证包位置。
import argparse
# 导入CSV库以读取既有自然日日表。
import csv
# 导入哈希库以登记图源字节摘要。
import hashlib
# 导入JSON库以读取绘图清单和写入侧车。
import json
# 导入系统模块以关闭字节码缓存。
import sys
# 在导入本地模块前关闭字节码写入。
sys.dont_write_bytecode = True
# 导入日期类型以计算实际月末点。
from datetime import date
# 导入日历模块以识别真实自然月最后一天。
import calendar
# 导入精确金额类型以避免金额显示误差。
from decimal import Decimal
# 导入路径类型以固定输出位置。
from pathlib import Path
# 导入换行工具以排版中文背景。
import textwrap
# 导入Matplotlib主模块以固定无界面后端。
import matplotlib
# 强制使用Agg后端。
matplotlib.use("Agg")
# 导入绘图接口以输出PNG。
import matplotlib.pyplot as plt
# 导入日期刻度工具以降低长期间横轴密度。
import matplotlib.dates as mdates

# 固定中文字体优先级。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
# 关闭中文图中的负号乱码。
plt.rcParams["axes.unicode_minus"] = False


# 返回文件原始字节的稳定摘要。
def sha(path: Path) -> str:
    # 读取原始字节计算SHA-256。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 只写不存在的文件以保护已审阅图件。
def write_once(path: Path, text: str) -> None:
    # 已存在表示不可覆盖。
    if path.exists():
        # 明确报告冲突路径。
        raise FileExistsError(f"拒绝覆盖：{path}")
    # 创建目标父目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入UTF-8侧车内容。
    path.write_text(text, encoding="utf-8")


# 读取单户日表并返回实际日期和万元余额。
def read_daily(path: Path, start: str, end: str) -> tuple[list[date], list[float], list[Decimal]]:
    # 以BOM兼容编码打开中央日表。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 读取全部日表记录。
        rows = list(csv.DictReader(handle))
    # 仅保留清单限定的实际日期窗口。
    rows = [row for row in rows if start <= row["calendar_date"] <= end]
    # 缺少实际记录时拒绝空图。
    if not rows:
        # 报告无数据的清单条目。
        raise ValueError(f"日表在指定期间为空：{path}")
    # 转换自然日横轴。
    dates = [date.fromisoformat(row["calendar_date"]) for row in rows]
    # 转换为万元显示值。
    values = [float(Decimal(row["ending_balance_cny"]) / Decimal("10000")) for row in rows]
    # 保留原始精确元金额供侧车登记。
    exact_values = [Decimal(row["ending_balance_cny"]) for row in rows]
    # 返回真实日表序列。
    return dates, values, exact_values


# 绘制一个企业面板，只展示实际CSV已有期间。
def draw_panel(axis, panel: dict) -> dict[str, object]:
    # 固定清单给出的绝对CSV路径。
    csv_path = Path(panel["csv_path"])
    # 读取实际日期和日终余额。
    dates, values, exact_values = read_daily(csv_path, panel["period_start"], panel["period_end"])
    # 绘制蓝色真实日终余额线。
    axis.plot(dates, values, color="#1f5f99", linewidth=1.0)
    # 找到每月最后一个实际日的索引。
    month_end_indices = [index for index, item in enumerate(dates) if item.day == calendar.monthrange(item.year, item.month)[1]]
    # 绘制橙色实际月末点及其连线。
    axis.plot([dates[index] for index in month_end_indices], [values[index] for index in month_end_indices], color="#d66b2c", marker="o", markersize=2.8, linewidth=0.8)
    # 找到最低余额的实际位置。
    low_index = min(range(len(values)), key=values.__getitem__)
    # 设置带运行编号的中文标题。
    axis.set_title(panel["title"], fontsize=11, loc="left")
    # 设置万元纵轴。
    axis.set_ylabel("万元", fontsize=10)
    # 横轴精确限制在实际日表范围。
    axis.set_xlim(dates[0], dates[-1])
    # 每六个月保留一个日期刻度。
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    # 使用简短年月格式显示刻度。
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # 添加轻量网格支持读数。
    axis.grid(alpha=0.2)
    # 写出最低和期末实际余额。
    # 将诊断和画像压缩为轴下三行，避免遮挡真实曲线。
    annotation = f"最低 {values[low_index]:,.2f}万（{dates[low_index]}）｜期末 {values[-1]:,.2f}万\n" + textwrap.fill(panel["diagnostic_text"], width=40, break_on_hyphens=False) + "\n" + textwrap.fill(panel["profile_cn"], width=40, break_on_hyphens=False)
    # 在轴下预留区域写注释。
    axis.text(0.01, -0.20, annotation, transform=axis.transAxes, fontsize=10.5, va="top", clip_on=False)
    # 返回可供侧车审阅的真实图源事实。
    return {"csv_path": str(csv_path), "csv_sha256": sha(csv_path), "run_id": panel["run_id"], "actual_start": dates[0].isoformat(), "actual_end": dates[-1].isoformat(), "lowest_cny": str(exact_values[low_index]), "ending_cny": str(exact_values[-1])}


# 绘制一页有限面板并登记所有实际图源。
def render_page(pack: Path, page: dict) -> None:
    # 读取页面面板集合。
    panels = page["panels"]
    # 九面板固定3行3列，其余六面板固定3行2列。
    rows, cols = (3, 3) if len(panels) == 9 else (3, 2)
    # 非九或六面板拒绝绘制避免误排版。
    if len(panels) not in (9, 6):
        # 报告不支持的面板数。
        raise ValueError("每页只支持9或6个面板")
    # 创建20英寸宽页面。
    figure, axes = plt.subplots(rows, cols, figsize=(20, 14.5), sharey="row" if cols == 2 else False)
    # 展平坐标轴方便逐面板排版。
    flat_axes = list(axes.flat)
    # 保存各面板图源事实。
    sources = []
    # 逐面板绘制真实余额。
    for axis, panel in zip(flat_axes, panels, strict=True):
        # 添加单面板图源记录。
        sources.append(draw_panel(axis, panel))
    # 写入整页标题。
    figure.suptitle(page["title"], fontsize=21, fontweight="bold", y=0.98)
    # 写入整页副标题。
    figure.text(0.5, 0.945, page["subtitle"], ha="center", fontsize=11)
    # 调整页面留白避免文字与面板重叠。
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.11, top=0.89, hspace=0.78, wspace=0.25)
    # 固定图片输出目录。
    image_dir = pack / "图片"
    # 创建图片目录。
    image_dir.mkdir(exist_ok=True)
    # 固定不可覆盖的PNG目标。
    output = image_dir / page["filename"]
    # 已存在时拒绝覆盖。
    if output.exists():
        # 报告历史图件冲突。
        raise FileExistsError(f"拒绝覆盖：{output}")
    # 输出清晰PNG。
    figure.savefig(output, dpi=180, bbox_inches="tight")
    # 关闭图形释放资源。
    plt.close(figure)
    # 写入图源摘要侧车。
    write_once(image_dir / f"{page['filename']}.图源SHA256.json", json.dumps({"filename": page["filename"], "sources": sources, "matplotlib_backend": matplotlib.get_backend()}, ensure_ascii=False, indent=2) + "\n")


# 解析清单并制作所有明确列出的页面。
def main() -> None:
    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="有限已生成日表的汇总余额图")
    # 要求传入验证包位置。
    parser.add_argument("--pack", required=True)
    # 解析用户参数。
    args = parser.parse_args()
    # 固定验证包路径。
    pack = Path(args.pack)
    # 读取包内绘图清单。
    manifest = json.loads((pack / "汇总绘图清单.json").read_text(encoding="utf-8-sig"))
    # 逐页生成PNG。
    for page in manifest["pages"]:
        # 绘制一页有限样本图。
        render_page(pack, page)


# 仅在命令行直接调用时执行绘图。
if __name__ == "__main__":
    # 启动只读绘图入口。
    main()
