# 模块字符串说明本文件只检查教学注释的结构存在性，不代替人工判断注释质量。
"""检查新增或实质修改代码是否逐条带有紧邻中文教学注释。"""

# 导入JSON标准库；审计结果需要形成机器可读报告。
import json
# 导入正则表达式标准库；它用于识别注释中是否至少含有一个中文字符。
import re
# 导入Python分词标准库；它能区分真实注释与字符串中的井号。
import tokenize
# 从数据类模块导入装饰器；它让单条失败记录拥有明确字段。
from dataclasses import asdict, dataclass
# 从内存流模块导入字节流；Python分词器从这里读取UTF-8源码。
from io import BytesIO
# 从路径模块导入Path；所有待审计文件都以平台无关路径处理。
from pathlib import Path

# 编译中文字符范围；至少命中一个中文字符才算中文教学注释。
CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# 冻结支持的可执行文件后缀；其它文件需要人工复核但不会被本工具误判。
SUPPORTED_SUFFIXES = frozenset({".py", ".bat", ".ps1", ".js", ".ts"})


# 数据类保存一条缺失注释；路径、行号、源码和原因可直接写入验收报告。
@dataclass(frozen=True)
class TeachingCommentFinding:  # 定义不可变失败记录，防止汇总过程中意外改写审计事实。
    # 保存发生问题的文件路径。
    path: str
    # 保存一开始计数的问题行号。
    line_number: int
    # 保存去除首尾空白后的源码，方便用户定位。
    source_line: str
    # 保存为什么没有通过结构检查。
    reason_cn: str


# 判断文本中是否含中文；返回布尔值供各语言检查器共用。
def contains_chinese(text: str) -> bool:
    # 使用预编译正则搜索，找到任一中文字符即返回真。
    return CHINESE_PATTERN.search(text) is not None


# 向上寻找紧邻的非空物理行；它用于判断代码上方是否有一一对应的中文注释。
def previous_nonblank_line(lines: list[str], one_based_line_number: int) -> str:
    # 从当前行的前一行转换成零基索引开始搜索。
    index = one_based_line_number - 2
    # 只要索引仍在文件范围内就继续向上跳过空行。
    while index >= 0:
        # 读取当前候选物理行并去除两侧空白。
        stripped = lines[index].strip()
        # 非空行就是紧邻的有效上一行，立即返回。
        if stripped:
            # 返回去除空白后的上一行文本。
            return stripped
        # 空行不承载注释关系，索引继续向上一行移动。
        index -= 1
    # 文件开头没有上一行时返回空字符串。
    return ""


# 检查Python文件；分词结果确保字符串里的井号不会被当成真实注释。
def audit_python_file(path: Path) -> tuple[int, list[TeachingCommentFinding]]:
    # 读取完整UTF-8源码；读取失败由调用方作为明确异常处理。
    source = path.read_text(encoding="utf-8")
    # 按物理行保留源码；审计报告需要返回原行内容。
    lines = source.splitlines()
    # 初始化所有真实Python注释，键为一开始计数的物理行号。
    comments_by_line: dict[int, list[str]] = {}
    # 初始化含有语法记号的可执行物理行集合。
    executable_lines: set[int] = set()
    # 把UTF-8源码编码成字节流交给Python官方分词器。
    token_stream = tokenize.tokenize(BytesIO(source.encode("utf-8")).readline)
    # 逐个读取词法记号，分别登记注释和可执行内容。
    for token in token_stream:
        # 真实注释记号单独保存，不把它计入可执行行。
        if token.type == tokenize.COMMENT:
            # 同一物理行理论上只有一个真实注释，列表结构仍保留稳健性。
            comments_by_line.setdefault(token.start[0], []).append(token.string)
            # 注释处理完毕后进入下一个记号。
            continue
        # 编码、换行、缩进和结束记号不代表独立可执行源码。
        if token.type in {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            # 跳过不需要教学注释的结构记号。
            continue
        # 多行字符串只登记起始行；内部文字不是独立Python语句。
        executable_lines.add(token.start[0])
    # 初始化缺失中文注释的失败记录。
    findings: list[TeachingCommentFinding] = []
    # 按行号排序检查，确保报告顺序稳定可复现。
    for line_number in sorted(executable_lines):
        # 合并同一行的真实注释，检查行末中文注释是否存在。
        same_line_comment = " ".join(comments_by_line.get(line_number, []))
        # 取得上一条非空物理行，检查行上方中文注释是否紧邻。
        previous_line = previous_nonblank_line(lines, line_number)
        # 上一行只有以井号开头才是Python注释行。
        previous_is_chinese_comment = previous_line.startswith("#") and contains_chinese(previous_line)
        # 行末注释或上一行注释任一含中文，就满足结构硬门。
        if contains_chinese(same_line_comment) or previous_is_chinese_comment:
            # 当前代码行已经有紧邻中文说明，继续检查下一行。
            continue
        # 安全取得源码行；异常分词行号不会让审计器崩溃。
        source_line = lines[line_number - 1].strip() if line_number <= len(lines) else ""
        # 记录缺失原因，人工复核可以继续判断语义质量。
        findings.append(TeachingCommentFinding(str(path), line_number, source_line, "可执行行缺少同一行或上一非空行的中文注释"))
    # 返回可执行行总数和全部失败记录。
    return len(executable_lines), findings


# 识别非Python文件的整行注释前缀；只做结构检查，不尝试解析完整语言语法。
def line_comment_prefixes(suffix: str) -> tuple[str, ...]:
    # BAT允许REM和双冒号作为整行说明。
    if suffix == ".bat":
        # 返回统一转为小写后使用的BAT注释前缀。
        return ("rem ", "rem\t", "::")
    # PowerShell使用井号作为行注释。
    if suffix == ".ps1":
        # 返回PowerShell注释前缀。
        return ("#",)
    # JavaScript和TypeScript使用双斜线作为行注释。
    if suffix in {".js", ".ts"}:
        # 返回脚本语言注释前缀。
        return ("//",)
    # 不支持的后缀没有合法前缀。
    return ()


# 检查BAT、PowerShell、JavaScript或TypeScript物理行是否由中文注释紧邻说明。
def audit_line_oriented_file(path: Path) -> tuple[int, list[TeachingCommentFinding]]:
    # 读取UTF-8文件并拆成物理行。
    lines = path.read_text(encoding="utf-8").splitlines()
    # 读取当前语言支持的整行注释前缀。
    prefixes = line_comment_prefixes(path.suffix.lower())
    # 初始化可执行行计数。
    executable_count = 0
    # 初始化失败记录。
    findings: list[TeachingCommentFinding] = []
    # 逐行检查并保留一开始计数的行号。
    for line_number, line in enumerate(lines, start=1):
        # 去除首尾空白以判断空行和整行注释。
        stripped = line.strip()
        # 空行不属于可执行代码。
        if not stripped:
            # 跳过空行。
            continue
        # 统一小写只用于识别大小写不敏感的REM前缀。
        lowered = stripped.lower()
        # 整行注释不计入可执行行。
        if any(lowered.startswith(prefix) for prefix in prefixes):
            # 跳过注释行。
            continue
        # 当前非空非注释行计为可执行行。
        executable_count += 1
        # 读取上一条非空物理行。
        previous_line = previous_nonblank_line(lines, line_number)
        # 统一上一行小写以识别BAT的REM。
        previous_lowered = previous_line.lower()
        # 上一行必须同时满足注释前缀和中文内容。
        previous_is_chinese_comment = any(previous_lowered.startswith(prefix) for prefix in prefixes) and contains_chinese(previous_line)
        # 有紧邻中文注释时继续检查下一行。
        if previous_is_chinese_comment:
            # 当前行通过结构硬门。
            continue
        # 记录当前文件、行号、源码和失败原因。
        findings.append(TeachingCommentFinding(str(path), line_number, stripped, "可执行行上方缺少紧邻中文整行注释"))
    # 返回可执行行总数和失败记录。
    return executable_count, findings


# 审计显式传入的文件列表；只检查本卡变更文件，避免追溯改写历史代码。
def audit_paths(paths: list[Path]) -> dict[str, object]:
    # 初始化所有文件的机器可读结果。
    file_results: list[dict[str, object]] = []
    # 初始化全部失败记录。
    all_findings: list[TeachingCommentFinding] = []
    # 初始化可执行行总数。
    total_executable_lines = 0
    # 按规范化路径排序，保证同一输入得到稳定报告顺序。
    for path in sorted((item.resolve() for item in paths), key=lambda item: str(item).lower()):
        # 拒绝不存在的文件，防止验收清单漏文件却被当成通过。
        if not path.is_file():
            # 记录不存在文件并继续汇总其它文件。
            finding = TeachingCommentFinding(str(path), 0, "", "待审计文件不存在")
            # 把不存在文件加入总失败列表。
            all_findings.append(finding)
            # 保存当前文件失败结果。
            file_results.append({"path": str(path), "supported": False, "executable_line_count": 0, "passed_line_count": 0, "finding_count": 1})
            # 继续处理下一个路径。
            continue
        # 取得小写后缀以选择语言检查器。
        suffix = path.suffix.lower()
        # 不支持的文件明确记录为未审计，而不是伪装成通过。
        if suffix not in SUPPORTED_SUFFIXES:
            # 保存未支持状态供人工验收处理。
            file_results.append({"path": str(path), "supported": False, "executable_line_count": 0, "passed_line_count": 0, "finding_count": 0})
            # 继续处理下一个文件。
            continue
        # Python使用官方分词器，其它支持语言使用行结构检查器。
        executable_count, findings = audit_python_file(path) if suffix == ".py" else audit_line_oriented_file(path)
        # 累加全部可执行行数量。
        total_executable_lines += executable_count
        # 累加当前文件失败记录。
        all_findings.extend(findings)
        # 计算当前文件通过行数。
        passed_count = executable_count - len(findings)
        # 保存当前文件结构审计结果。
        file_results.append({"path": str(path), "supported": True, "executable_line_count": executable_count, "passed_line_count": passed_count, "finding_count": len(findings)})
    # 只有没有任何失败记录时才允许结构审计通过。
    passed = not all_findings
    # 返回可直接序列化的完整报告。
    return {"status": "passed" if passed else "failed", "file_count": len(file_results), "total_executable_line_count": total_executable_lines, "total_finding_count": len(all_findings), "files": file_results, "findings": [asdict(item) for item in all_findings], "manual_semantic_review_still_required": True}


# 把审计报告转换为稳定的UTF-8友好JSON文本。
def render_audit_report(report: dict[str, object]) -> str:
    # 禁止ASCII转义中文并固定缩进，方便普通用户阅读和Git复核。
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
