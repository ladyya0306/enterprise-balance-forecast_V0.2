# 模块说明：本文件的函数只处理元数据和哈希，绝不读取最终答案。
"""C3A最终测试前预检的纯函数；只处理元数据和哈希，绝不读取最终答案。"""

# 导入哈希库，用于把文件和密封企业清单固定成可复核指纹。
import hashlib
# 导入JSON库，用于读取和生成运行包元数据。
import json
# 导入数据类工具，用不可变对象表达密封清单摘要。
from dataclasses import dataclass
# 导入Path，统一表达本地文件路径而不打开最终流水。
from pathlib import Path
# 导入Any，标记来自JSON和CSV的结构化数据。
from typing import Any, Iterable, Mapping


# 定义本模块专用异常，让预检失败不会被误报为最终测试结果。
class C3APreflightError(ValueError):
    # 异常说明：用于区分预检失败与任何最终测试结果。
    """表示C3A密封、血缘或模型包核验失败。"""


# 用不可变摘要保存最终企业的数量和不暴露编号的哈希。
@dataclass(frozen=True)
# 类说明：保存最终企业的脱敏密封摘要。
class FinalCohortSeal:
    # 类文档说明：对象只含数量和哈希，不保留企业明细。
    """最终52户的仅元数据密封摘要。"""

    # 保存最终企业数，必须等于冻结的52。
    enterprise_count: int
    # 保存仅最终企业身份与已有日度表哈希组成的整体指纹。
    enterprise_ids_sha256: str
    # 保存运行目录与日度表哈希组成的血缘指纹。
    lineage_sha256: str


# 计算单个文件的SHA-256，任何内容漂移都会改变返回值。
def sha256_file(path: Path) -> str:
    # 函数说明：返回指定文件的内容指纹。
    """返回文件内容的SHA-256十六进制摘要。"""
    # 以二进制方式读取文件，避免文本编码影响指纹。
    payload = path.read_bytes()
    # 构造SHA-256算法对象并返回稳定的十六进制摘要。
    return hashlib.sha256(payload).hexdigest()


# 核验一个结果目录中所有已登记输出是否仍与其哈希清单相同。
def verify_output_hashes(output_directory: Path, hash_manifest: Mapping[str, str]) -> None:
    # 函数说明：逐份比对结果包的登记指纹。
    """逐个核验清单文件，发现缺失或变化立即拒绝。"""
    # 依次读取相对路径和期望摘要，不依赖目录枚举顺序。
    for relative_path, expected_hash in hash_manifest.items():
        # 把清单相对路径连接到唯一结果目录。
        file_path = output_directory / relative_path
        # 若文件不存在，说明模型包已不完整。
        if not file_path.is_file():
            # 抛出明确异常而不是静默跳过。
            raise C3APreflightError(f"C3A缺少已登记模型包文件：{relative_path}")
        # 重新计算文件指纹，确认没有被替换。
        actual_hash = sha256_file(file_path)
        # 不相等就阻断C3A，禁止携带变动后的运行包进入最终测试。
        if actual_hash != expected_hash:
            # 报出相对文件名，便于定位但不暴露最终答案。
            raise C3APreflightError(f"C3A模型包哈希不一致：{relative_path}")


# 从只含血缘字段的T2清单构建最终企业摘要；不读取任何企业日度文件。
def build_final_cohort_seal(rows: Iterable[Mapping[str, str]], expected_count: int) -> FinalCohortSeal:
    # 函数说明：从最终企业元数据生成不泄漏明细的摘要。
    """验证最终企业元数据并返回不含企业编号明文的密封摘要。"""
    # 初始化收集最终企业的稳定文本行。
    identity_rows: list[str] = []
    # 初始化收集最终企业运行血缘的稳定文本行。
    lineage_rows: list[str] = []
    # 逐行处理清单；此处行只包含编号、路径和既有哈希，不含余额或标签。
    for row in rows:
        # 只选择明确标记为final_test的企业。
        if row.get("split_group") != "final_test":
            # 非最终企业属于开发池，不进入最终密封摘要。
            continue
        # 读取企业编号，用于检查最终企业是否重复。
        enterprise_id = row.get("enterprise_id", "")
        # 读取已登记运行目录，用于锁定企业来源位置。
        run_directory = row.get("run_directory", "")
        # 读取既有日度表指纹；本行只读哈希文本，不打开日度表。
        daily_hash = row.get("account_daily_total_sha256", "")
        # 读取T1资格状态，保证最终企业均是合格输入。
        qualification = row.get("source_qualification", "")
        # 编号、路径和日度表指纹缺一不可，否则无法追溯最终考卷。
        if not enterprise_id or not run_directory or len(daily_hash) != 64:
            # 阻断不完整的密封清单。
            raise C3APreflightError("C3A最终企业清单缺少编号、运行目录或日度表指纹")
        # 最终企业必须已通过T1资格审计。
        if qualification != "T1_passed":
            # 拒绝不合格企业被带入最终测试。
            raise C3APreflightError("C3A最终企业未通过T1资格审计")
        # 保存身份行，用企业编号检测重复但不把编号写入C3A输出。
        identity_rows.append(enterprise_id)
        # 保存血缘行，让同一企业换目录或换日度表指纹都会被发现。
        lineage_rows.append(f"{enterprise_id}|{run_directory}|{daily_hash}")
    # 企业编号必须唯一，重复会导致52户计数失真。
    if len(identity_rows) != len(set(identity_rows)):
        # 阻断重复最终企业。
        raise C3APreflightError("C3A最终企业清单存在重复编号")
    # 企业数必须恰等于冻结数量，不能少户或临时补户。
    if len(identity_rows) != expected_count:
        # 报出期望值和实际值，便于人工核验。
        raise C3APreflightError(f"C3A最终企业数不等于冻结值：{len(identity_rows)} != {expected_count}")
    # 先排序再拼接，保证CSV原始行序变化不会改变同一集合的摘要。
    identity_payload = "\n".join(sorted(identity_rows)).encode("utf-8")
    # 对血缘行也排序，固定其集合指纹。
    lineage_payload = "\n".join(sorted(lineage_rows)).encode("utf-8")
    # 返回只含数量和摘要的不可变对象，不输出52户明细。
    return FinalCohortSeal(len(identity_rows), hashlib.sha256(identity_payload).hexdigest(), hashlib.sha256(lineage_payload).hexdigest())


# 核验C2F报告中的范围边界，避免把错误模型包送进最终测试。
def assert_c2f_report(report: Mapping[str, Any], selected_method_id: str) -> None:
    # 函数说明：确认C2F交付物可作为唯一最终运行包。
    """验证C2F唯一候选、三模型和未开启最终测试的承诺。"""
    # C2F本身必须成功完成。
    if report.get("status") != "passed":
        # 失败的定型包不可进入C3A。
        raise C3APreflightError("C3A发现C2F报告不是通过状态")
    # 方法编号必须与冻结唯一候选完全相同。
    if report.get("selected_method_id") != selected_method_id:
        # 禁止用其他模型替换已锁定候选。
        raise C3APreflightError("C3A发现C2F方法编号与唯一候选不一致")
    # C2F合同固定训练三个种子。
    if report.get("model_count") != 3:
        # 少于或多于三种子都表示运行包不再符合合同。
        raise C3APreflightError("C3A发现C2F模型数量不是固定的3个")
    # C2F绝不能提前读取最终测试。
    if report.get("final_test_opened") is not False:
        # 若已开启，C3A应停止而不是继续伪装成预检。
        raise C3APreflightError("C3A发现C2F已开启最终测试")
    # C2F必须明确锁定最终模型包。
    if report.get("final_model_locked") is not True:
        # 未锁定表示还可能被重选或替换。
        raise C3APreflightError("C3A发现C2F最终模型包未锁定")


# 构造C3B未来必须遵守的一次性锁内容；此函数不产生预测或分数。
def build_one_time_lock(seal: FinalCohortSeal, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    # 函数说明：构造以后C3B必须遵守的只读考试锁。
    """返回不含最终答案的一次性最终测试锁。"""
    # 用字典保存以后C3B需要逐项匹配的冻结事实。
    return {
        # 固定锁编号，便于后续日志引用同一份考试锁。
        "lock_id": "POST-T6-C3B-ONE-TIME-LOCK-20260901",
        # 标记C3A已通过但C3B尚未获授权。
        "c3a_status": "passed",
        # 明确C3B仍需要项目发起人单独授权。
        "c3b_separate_owner_authorization_required": True,
        # 明确当前没有授权执行C3B。
        "c3b_authorized_now": False,
        # 固定最终企业数量。
        "final_enterprise_count": seal.enterprise_count,
        # 保存不含明文企业编号的最终考卷身份摘要。
        "final_enterprise_ids_sha256": seal.enterprise_ids_sha256,
        # 保存运行目录与日度表指纹的血缘摘要。
        "final_lineage_sha256": seal.lineage_sha256,
        # 保存模型、合同、代码和上游清单的指纹。
        "locked_source_sha256": dict(source_hashes),
        # 规定预测必须早于任何标签读取。
        "prediction_must_precede_label_reveal": True,
        # 规定最终测试只能开启一次。
        "final_test_open_once_only": True,
        # 声明C3A尚未读取最终标签。
        "final_labels_read": False,
        # 声明C3A尚未创建最终特征。
        "final_features_created": False,
        # 声明C3A尚未创建最终预测。
        "final_predictions_created": False,
        # 声明C3A尚未计算最终指标。
        "final_metrics_computed": False,
        # 声明C3A没有开启最终测试。
        "final_test_opened": False,
        # 结束一次性锁字典，后续只能整体核验而不能替换单项。
    }
