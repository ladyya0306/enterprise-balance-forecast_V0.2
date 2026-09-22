# 批次总览复用既有余额面板，不临时编写每户绘图代码。
"""从已交付正式日表制作固定八图总览，所有样本都展示。"""
# 命令指定已完成汇总和原执行名单。
import argparse
# 绘图保存不参与流水生成。
import matplotlib.pyplot as plt
# 固定读取与指纹。
import synthetic_batch_workflow as flow
# 复用已有实际日表面板，包括精确最低和期末金额。
from render_finite_summary_20260909 import draw_panel

# 固定入口每批都可调用。
def main():
    # 不自动寻找或筛选好看的曲线。
    parser = argparse.ArgumentParser(description="已交付批次余额总览")
    # 冻结名单提供企业背景。
    parser.add_argument("--manifest", required=True)
    # 读取明确参数。
    args = parser.parse_args()
    # 只读本批。
    manifest = flow.read(flow.locate(args.manifest))
    # 所有总览保存在同一生产批次内。
    root = flow.output_path(manifest["output"]) / "余额总览"
    # 交付完成后才能制作整批总览。
    summary = flow.read(flow.output_path(manifest["output"]) / "批次汇总.json")
    # 原顺序展示，不能按漂亮程度排序。
    rows = {r["sample_id"]: r for r in manifest["rows"]}
    # 字体沿用已验证的本机中文字体。
    if flow.delivery.CHINESE_FONT is not None:
        # 注册字体供复用面板的普通标题调用。
        flow.delivery.font_manager.fontManager.addfont(str(flow.delivery.CHINESE_FONT_PATH))
        # 固定字体避免中文方框。
        plt.rcParams["font.sans-serif"] = [flow.delivery.CHINESE_FONT.get_name()]
    # 报告索引包括每一页。
    links = ["# 本批全部余额图", "", "蓝线为真实日余额，橙线仅连接真实月末点。所有图均展示；停止、短记录及未过门槛者不隐藏。", ""]
    # 每页八户，便于逐页完整查看。
    for offset in range(0, len(summary["delivered"]), 8):
        # 新图固定命名。
        target = root / f"余额总览_{offset//8+1:02d}.png"
        # 每页绑定真实来源，不重复画已完成的图。
        page = summary["delivered"][offset:offset+8]
        # 首次才渲染。
        if not target.exists():
            # 固定四行两列，最后一页不足则隐藏空坐标。
            figure, axes = plt.subplots(4, 2, figsize=(19, 20))
            # 记录所有图源证据。
            evidence = []
            # 保留原逐户顺序。
            for axis, receipt in zip(axes.flat, page):
                # 画像与完整预算从明确引用读取。
                scenario = flow.read(flow.reference(rows[receipt["sample_id"]]["inputs"]["scenario"]))
                # 正式日表是唯一图源。
                csv = flow.reference(receipt["artifacts"]["日余额"])
                # 标题用用户能识别的位置，注明质量状态。
                title = rows[receipt["sample_id"]]["combination_id"] + " " + receipt["sample_id"].split("_")[-1].upper()
                # 复用原面板，不另算或修饰余额。
                evidence.append(draw_panel(axis, {"csv_path": str(csv), "period_start": scenario["operating_start"], "period_end": scenario["source_budget"]["period"]["end_date"], "title": title, "run_id": receipt["run_id"], "diagnostic_text": "资金不足停止" if receipt["stop"] else "记录至期末", "profile_cn": "曲线内重复：" + ("未过门槛" if not receipt["curve_quality"]["passed"] else "未检出；短记录另核" if not receipt["curve_quality"]["sufficient_observation"] else "未检出")}))
            # 最后一页的空位不用虚构企业填满。
            for axis in list(axes.flat)[len(page):]:
                # 只隐藏空坐标。
                axis.set_visible(False)
            # 留出每户金额说明的位置。
            figure.subplots_adjust(hspace=0.7, wspace=0.2, top=0.94, bottom=0.08)
            # 页面标题明确是实际日表。
            figure.suptitle(f"实际余额总览 第{offset//8+1}页", fontsize=20)
            # 原生产根内建立图目录。
            root.mkdir(parents=True, exist_ok=True)
            # 保存清晰原图，不重写任何CSV。
            figure.savefig(target, dpi=140, bbox_inches="tight")
            # 及时释放图形内存。
            plt.close(figure)
            # 侧车绑定图及每户数据来源。
            flow.stable_write(target.with_suffix(".json"), {"sources": evidence, "png_sha256": flow.sha(target)})
        # Markdown内嵌可直接浏览。
        links.extend([f"![第{offset//8+1}页]({target.as_posix()})", ""])
    # 总览入口不需要用户逐个寻找文件。
    flow.stable_write(root / "查看全部余额图.md", "\n".join(links))

# 只有显式调用才渲染。
if __name__ == "__main__":
    # 固定命令入口。
    main()
