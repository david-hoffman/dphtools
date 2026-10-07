"""Select conservative verification scopes from explicit ownership and Git changes."""

import ast
import json
from pathlib import Path
import subprocess

TIERS = {"fast": 1, "library": 2, "doctor": 3, "release": 4, "full": 5}
DOMAINS = ("library", "doctor", "release", "verification")
SAFE_IMPORTS = {
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "skimage",
    "mpl_toolkits",
    "math",
    "numbers",
    "operator",
    "functools",
    "itertools",
    "collections",
    "logging",
    "warnings",
    "typing",
    "textwrap",
    "pytest",
}
RISK_NAMES = {
    "__import__",
    "__getattr__",
    "__dir__",
    "__version__",
    "get_versions",
    "importlib",
    "metadata",
    "pkg_resources",
    "subprocess",
    "multiprocessing",
    "os",
    "sys",
    "pathlib",
    "Path",
    "venv",
    "pip",
    "versioneer",
    "getattr",
    "eval",
    "exec",
    "globals",
    "locals",
    "compile",
    "open",
    "load",
    "save",
    "memmap",
}


def inventory(root):
    """Load explicit test ownership; reject unassigned runtime and test helpers."""
    from verification import owned_sources

    mapping = json.loads((root / "tools/verification-domains.json").read_text())
    sources = owned_sources(root)
    tests = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")}
    assigned = [p for group in mapping["tests"].values() for p in group]
    known = set(assigned) | set(mapping["support"])
    runtime = {p for group in mapping["runtime"].values() for p in group}
    unknown = sorted(
        (tests - known)
        | (set(sources) - runtime - {p for p in sources if p.startswith("dphtools/")})
    )
    if len(assigned) != len(set(assigned)) or known - tests:
        raise ValueError("Test ownership is duplicate or names absent files")
    return mapping, sources, unknown


def scope(root, name, classification=None):
    """Inventory the complete measured domain closure and disclose its complement."""
    from verification import owned_sources

    root = Path(root)
    all_sources = owned_sources(root)
    paths = ["dphtools", "tests"]
    domains = list(DOMAINS)
    unvalidated_tests = []
    sources = all_sources
    if name in ("library", "doctor", "release"):
        mapping, all_sources, unknown = inventory(root)
        if unknown:
            return scope(root, "full", {"reason": "Unassigned files", "files": unknown})
        domains = ["library"]
        if name == "doctor":
            domains.append("doctor")
        if name == "release":
            domains.extend(("doctor", "release"))
        selected = {p for d in domains for p in mapping["tests"][d]}
        for domain in domains:
            selected.update(mapping["shared_tests"].get(domain, []))
        paths = ["dphtools", *sorted(selected)]
        unvalidated_tests = sorted(
            {p for group in mapping["tests"].values() for p in group} - selected
        )
        sources = sorted(
            p
            for p in all_sources
            if p.startswith("dphtools/")
            or any(p in mapping["runtime"][d] for d in domains if d != "library")
        )
    if name == "fast":
        domains, sources, paths = [], [], []
        unvalidated_tests = ["dphtools", "tests"]
    return {
        "version": "1.0",
        "name": name,
        "tier": TIERS[name],
        "domains": domains,
        "coverage_claim": "global" if name == "full" else ("none" if name == "fast" else "scoped"),
        "measured_sources": sources,
        "unvalidated_sources": sorted(set(all_sources) - set(sources)),
        "selected_test_paths": paths,
        "unvalidated_test_paths": unvalidated_tests,
        "classification": classification,
    }


PLAIN_NODES = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def plain_expression(node):
    """Recognize only literal built-in expressions, without names or implicit object hooks."""
    return node is None or all(isinstance(part, PLAIN_NODES) for part in ast.walk(node))


def changed_functions_safe(before, after):
    """Permit only a small supported syntax set with unchanged import/initialization inputs."""
    old, new = ast.parse(before), ast.parse(after)
    dumps = lambda node: ast.dump(node, include_attributes=False)
    for tree in (old, new):
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(alias.name not in SAFE_IMPORTS for alias in node.names):
                    return False
            elif isinstance(node, ast.ImportFrom):
                if (
                    node.level
                    or node.module not in SAFE_IMPORTS
                    or any(alias.name == "*" for alias in node.names)
                ):
                    return False
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.decorator_list or node.name in RISK_NAMES:
                    return False
                body = node.body
                node.body = []
                header_nodes = (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.arguments,
                    ast.arg,
                    ast.Name,
                    *PLAIN_NODES,
                )
                if any(not isinstance(part, header_nodes) for part in ast.walk(node)):
                    return False
                defaults = [*node.args.defaults, *node.args.kw_defaults]
                arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                arguments += [
                    arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None
                ]
                annotations = [node.returns, *(arg.annotation for arg in arguments)]
                if not all(plain_expression(value) for value in defaults):
                    return False
                if not all(
                    plain_expression(value)
                    or isinstance(value, ast.Name)
                    and value.id
                    in {
                        "int",
                        "float",
                        "complex",
                        "bool",
                        "str",
                        "bytes",
                        "object",
                        "tuple",
                        "list",
                        "dict",
                        "set",
                    }
                    for value in annotations
                ):
                    return False
                for statement in body:
                    if isinstance(statement, ast.Return):
                        if not plain_expression(statement.value):
                            return False
                    elif isinstance(statement, ast.Assert):
                        if not plain_expression(statement.test) or not plain_expression(
                            statement.msg
                        ):
                            return False
                    elif isinstance(statement, ast.Pass):
                        continue
                    elif not (
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)
                    ):
                        return False
            elif isinstance(node, ast.Assign):
                if not all(isinstance(target, ast.Name) for target in node.targets):
                    return False
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return False
            elif not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return False
    return dumps(old) == dumps(new)


def shared_fixture_change(before, after):
    """Reject changed fixture definitions or imports shared by test execution."""
    trees = [ast.parse(text) for text in (before, after)]
    imports = [
        [
            ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        for tree in trees
    ]
    if imports[0] != imports[1]:
        return True
    fixtures = []
    for tree in trees:
        aliases = {"fixture"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pytest":
                aliases.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "fixture"
                )
        fixtures.append(
            [
                ast.dump(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(
                    isinstance(part, ast.Attribute)
                    and part.attr == "fixture"
                    or isinstance(part, ast.Name)
                    and part.id in aliases
                    for decorator in node.decorator_list
                    for part in ast.walk(decorator)
                )
            ]
        )
    return fixtures[0] != fixtures[1]


def git(root, *arguments):
    """Read Git metadata without fetching, modifying refs, or contacting remotes."""
    return subprocess.check_output(["git", *arguments], cwd=root, text=True, encoding="utf-8")


def classify(root, base):
    """Choose the highest approved trigger, falling back to full for uncertainty."""
    root = Path(root)
    reasons = []
    details = {"requested_base": base, "reasons": reasons}
    try:
        mapping, owned, unknown = inventory(root)
        if unknown:
            raise ValueError("Unassigned files: " + ", ".join(unknown))
        if git(root, "status", "--porcelain", "--untracked-files=normal").strip():
            raise ValueError("Checkout has uncommitted files")
        if not base:
            raise ValueError("No integration base supplied")
        base_sha = git(root, "rev-parse", "--verify", base + "^{commit}").strip()
        head = git(root, "rev-parse", "HEAD").strip()
        git(root, "merge-base", "--is-ancestor", base_sha, head)
        details.update(base_sha=base_sha, head_sha=head)
        entries = git(
            root, "diff", "--name-status", "--no-renames", "-z", base_sha, head, "--"
        ).split("\0")[:-1]
        changed = list(zip(entries[::2], entries[1::2]))
        details["changes"] = [{"status": status, "path": path} for status, path in changed]
        names = set()
        for status, path in changed:
            if status != "M":
                raise ValueError("Addition, deletion, or rename: " + path)
            before = git(root, "show", base_sha + ":" + path)
            after = (root / path).read_text(encoding="utf-8")
            if path in mapping.get("shared_inputs", []):
                raise ValueError("Shared policy/installation/helper input: " + path)
            if path in {
                p for group in mapping["tests"].values() for p in group
            } and shared_fixture_change(before, after):
                raise ValueError("Changed shared fixture or test imports: " + path)
            if path in mapping["prose"] and not any(
                marker in before + after for marker in ("```", "~~~", ">>>", "    ", "\t")
            ):
                names.add("fast")
            elif path in mapping["runtime"]["release"] or path in mapping["tests"]["release"]:
                if not changed_functions_safe(before, after):
                    raise ValueError(
                        "Policy, process, initialization, or uncertain release effect: " + path
                    )
                names.add("release")
            elif path in mapping["tests"]["doctor"]:
                if not changed_functions_safe(before, after):
                    raise ValueError(
                        "Policy, process, initialization, or uncertain doctor effect: " + path
                    )
                names.add("doctor")
            elif (
                path in owned
                and path.startswith("dphtools/")
                or path in mapping["tests"]["library"]
            ):
                if not changed_functions_safe(before, after):
                    raise ValueError(
                        "Import, initialization, fixture, signature, or uncertain library effect: "
                        + path
                    )
                names.add("library")
            else:
                raise ValueError("Shared, packaging, policy, or unknown input: " + path)
            reasons.append(path)
        product_domains = names - {"fast"}
        if len(product_domains) > 1:
            raise ValueError("Changes span domains")
        name = max(names or {"full"}, key=TIERS.get)
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.CalledProcessError) as error:
        name = "full"
        reasons.append(str(error))
    return scope(root, name, details)
