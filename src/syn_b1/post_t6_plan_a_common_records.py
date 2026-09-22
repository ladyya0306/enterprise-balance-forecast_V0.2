# 模块说明：本文件集中实现Plan A输入边界和PA0的只读核对。
"""Plan A输入边界与PA0只读核对的共同实现。"""

# 导入CSV标准库；本模块逐户读取A1登记清单而不接触逐笔流水。
import csv
# 导入哈希标准库；本模块用同一算法核对已保存文件是否被改写。
import hashlib
# 导入JSON标准库；本模块读取既有阶段报告并写出机器可读的保护结果。
import json
# 从pathlib导入Path；本模块以统一方式处理中文目录和相对路径。
from pathlib import Path


# 固定Plan A版本；边界或文件选择改变时必须新建版本而不能覆盖本次记录。
PLAN_A_VERSION = "syn_b1_post_t6_plan_a_three_view_path_audit_2_0_corrected"
# 固定本阶段只可使用的开发池企业数；52户最终企业不属于本阶段输入。
DEVELOPMENT_ENTERPRISE_COUNT = 208
# 固定唯一允许的B2方法编号；其它候选不能混入既有路线评价。
SELECTED_B2_CONFIGURATION = "lstm_direct_selected_window_h16_scaled_l1"
# 固定三次独立训练的起始编号；缺少或替换任何一次都必须停止。
SELECTED_B2_TRAINING_SEEDS = (2026083102, 2026083103, 2026083104)
# 固定四个历史时间检查编号；它们用于后续公平汇总而不是挑选最好一段。
SELECTED_B2_FOLDS = ("A", "B", "C", "D")
# 固定本轮输出目录名；已有同名目录时一律不覆盖。
OUTPUT_DIRECTORY_NAME = "_post_t6_plan_a_three_view_path_audit_v2_corrected"


# 定义专用异常；把输入边界错误与模型好坏明确区分开。
class PlanAInputError(ValueError):
    # 类说明：此异常表示程序触碰了不允许的来源或发现已保存证据不一致。
    """当Plan A试图读取未允许来源或发现证据不一致时抛出。"""


# 计算一个文件的SHA-256指纹；文件内容任何变化都会改变结果。
def sha256_file(path: Path) -> str:
    # 创建SHA-256累计器。
    digest = hashlib.sha256()
    # 以二进制只读方式打开已获准文件。
    with path.open("rb") as handle:
        # 分块读取，避免大表一次占满内存。
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            # 把当前块加入累计指纹。
            digest.update(block)
    # 返回小写十六进制指纹，便于同既有报告逐字比较。
    return digest.hexdigest()


# 把一个路径转换为相对于项目根目录的规范表示；根外路径一律拒绝。
def project_relative_path(project_root: Path, path: Path) -> str:
    # 解析根目录和目标路径，统一大小写、点号和Windows分隔符。
    resolved_root = project_root.resolve()
    # 解析目标路径；此操作只处理路径文字，不读取文件内容。
    resolved_path = path.resolve()
    # 尝试将目标表达为根目录以内的相对路径。
    try:
        # 得到不含盘符的相对路径。
        relative = resolved_path.relative_to(resolved_root)
    # 根目录以外的路径不属于本项目，必须阻断。
    except ValueError as error:
        # 抛出含中文原因的专用错误。
        raise PlanAInputError(f"输入路径不在项目目录内：{path}") from error
    # 返回用正斜杠表示的稳定键，避免Windows反斜杠影响清单比较。
    return relative.as_posix()


# 判断一条相对路径是否属于人工备注、逐笔流水、最终测试、模型检查点或数据库。
def forbidden_reason(relative_path: str) -> str | None:
    # 将路径拆成小写部分，便于不受大小写影响地检查。
    parts = tuple(part.lower() for part in Path(relative_path).parts)
    # 取得小写文件名，供精确文件规则使用。
    filename = parts[-1] if parts else ""
    # 人工备注和逐笔流水无论放在哪里都禁止进入Plan A。
    if filename == "cash_flow_review_notes.csv":
        # 返回普通读者可理解的拒绝原因。
        return "人工备注表只供人工查看，不能作为评价输入"
    # 逐笔流水会重新引入交易多少等未授权分组，必须在打开前拒绝。
    if filename == "transactions_total.csv":
        # 返回逐笔流水拒绝原因。
        return "逐笔流水不在Plan A允许范围内"
    # 最终测试目录任何文件都不能作为开发池复审输入。
    if "_post_t6_c3b_final_test_v1" in parts:
        # 返回最终测试拒绝原因。
        return "最终52户测试结果永久隔离，Plan A不得读取"
    # 受限目录含画像等非公开信息，不能被评价程序打开。
    if "_restricted" in parts:
        # 返回受限目录拒绝原因。
        return "受限目录不属于公开每日余额评价范围"
    # 模型权重文件不能被Plan A用来重训、续训或重新预测。
    if filename.endswith((".pt", ".pth", ".ckpt")) or "checkpoint" in parts or "checkpoints" in parts:
        # 返回模型检查点拒绝原因。
        return "模型检查点不属于只读评价输入"
    # 数据库文件也不能作为隐含输入或写入目标。
    if filename.endswith((".db", ".sqlite", ".sqlite3")):
        # 返回数据库拒绝原因。
        return "数据库不属于Plan A允许输入"
    # 未验收失败目录不能混入正式评价证据。
    if any("failed_unaccepted" in part for part in parts):
        # 返回失败目录拒绝原因。
        return "未验收失败目录不能作为正式评价输入"
    # 未命中任何禁止规则时返回空值。
    return None


# 创建只允许显式登记文件的读取守卫；守卫先拒绝后才允许任何打开操作。
class AllowedInputReader:
    # 用项目根目录和显式允许路径集合初始化守卫。
    def __init__(self, project_root: Path, allowed_relative_paths: set[str]) -> None:
        # 保存已解析项目根目录，所有路径比较都以它为准。
        self.project_root = project_root.resolve()
        # 保存稳定排序后的允许路径集合，避免调用者凭目录名扩大范围。
        self.allowed_relative_paths = frozenset(sorted(allowed_relative_paths))

    # 在任何文件系统读取前检查路径是否禁止且是否被明确列入清单。
    def assert_allowed(self, path: Path) -> str:
        # 先把输入转为项目内稳定相对路径；根外路径会立即拒绝。
        relative_path = project_relative_path(self.project_root, path)
        # 先按禁止规则拒绝，保证禁止文件不会被随后打开。
        reason = forbidden_reason(relative_path)
        # 命中禁止规则时立即停止。
        if reason is not None:
            # 给出具体拒绝原因，方便人工复核。
            raise PlanAInputError(f"拒绝读取 {relative_path}：{reason}")
        # 未在显式允许清单中的文件即使看起来相似也不得打开。
        if relative_path not in self.allowed_relative_paths:
            # 拒绝未登记文件，防止目录级误读。
            raise PlanAInputError(f"拒绝读取未登记文件：{relative_path}")
        # 返回已通过检查的稳定相对路径。
        return relative_path

    # 只读取得UTF-8文本；先执行守卫以保证不先碰禁止文件。
    def read_text(self, path: Path) -> str:
        # 先完成路径边界检查。
        self.assert_allowed(path)
        # 只在检查通过后读取文本。
        return path.read_text(encoding="utf-8")

    # 只读取得CSV行；先执行守卫以保证不先碰禁止文件。
    def read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        # 先完成路径边界检查。
        self.assert_allowed(path)
        # 只在检查通过后以UTF-8读取CSV。
        with path.open("r", encoding="utf-8", newline="") as handle:
            # 返回带字段名的全部行，PA0仅用于208行清单。
            return list(csv.DictReader(handle))

    # 只读计算一个文件指纹；先执行守卫以保证不先碰禁止文件。
    def hash_file(self, path: Path) -> str:
        # 先完成路径边界检查。
        self.assert_allowed(path)
        # 只在检查通过后读取二进制内容计算指纹。
        return sha256_file(path)


# 检查既有报告中登记的每个输出文件指纹；返回可写入PA0的核对结果。
def verify_report_outputs(reader: AllowedInputReader, directory: Path, report_filename: str) -> list[dict[str, str]]:
    # 组合报告的完整路径。
    report_path = directory / report_filename
    # 通过守卫读取报告文字后解析JSON。
    report = json.loads(reader.read_text(report_path))
    # 报告必须存在输出指纹字典，否则不能证明保存结果未变。
    if not isinstance(report.get("output_sha256"), dict):
        # 阻断缺少输出指纹的报告。
        raise PlanAInputError(f"既有报告缺少输出指纹：{project_relative_path(reader.project_root, report_path)}")
    # 初始化每个已核对文件的记录列表。
    verified_rows: list[dict[str, str]] = []
    # 按文件名稳定排序，保证重复运行时清单顺序一致。
    for filename, expected_hash in sorted(report["output_sha256"].items()):
        # 组合当前输出的完整路径。
        output_path = directory / filename
        # 计算守卫许可后的实际指纹。
        actual_hash = reader.hash_file(output_path)
        # 指纹不同意味着历史结果已经变动，不能继续评价。
        if actual_hash != expected_hash:
            # 抛出同时含文件名和预期值的错误。
            raise PlanAInputError(f"既有输出指纹不一致：{project_relative_path(reader.project_root, output_path)}")
        # 保存通过核对的一行，后续写入输入证据清单。
        verified_rows.append({"relative_path": project_relative_path(reader.project_root, output_path), "sha256": actual_hash, "source": report_filename, "purpose_cn": "既有阶段已保存结果的指纹核对"})
    # 返回完整通过列表。
    return verified_rows
