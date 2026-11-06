"""Check unrounded coverage counts and the contents of source and wheel archives."""

import argparse
import json
import stat
import tarfile
import tomllib
import zipfile
from configparser import ConfigParser
from pathlib import Path, PurePosixPath, PureWindowsPath


def check_coverage(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    totals = report["totals"]
    hits, lines = totals["covered_lines"], totals["num_statements"]
    branches, taken = totals["num_branches"], totals["covered_branches"]
    if not report["meta"]["branch_coverage"] or not (0 < lines and 0 < branches):
        raise ValueError("Expected line and branch measurements")
    if 100 * hits < 90 * lines or 100 * (hits + taken) < 90 * (lines + branches):
        raise ValueError("Line and combined coverage must each reach 90 percent")
    entry = [
        row["summary"]
        for name, row in report["files"].items()
        if Path(name.replace("\\", "/")).parts[-2:] == ("modelaudit", "__main__.py")
    ]
    if len(entry) != 1 or not entry[0]["covered_lines"] or entry[0]["missing_lines"]:
        raise ValueError("Expected full coverage of the CLI subprocess entry point")
    print(f"Lines {hits}/{lines}. Combined {hits + taken}/{lines + branches}.")


def safe_name(name):
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or PureWindowsPath(name).drive
        or ":" in name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Unsafe archive path {name!r}")
    return PurePosixPath(name)


def verify_members(actual, required, optional=()):
    missing = required - actual
    unexpected = actual - required - set(optional)
    if missing or unexpected:
        raise ValueError(f"Missing {sorted(missing)}. Unexpected {sorted(unexpected)}.")


def check_archives(directory, root):
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    stem = f"{project['name']}-{project['version']}"
    wheel = directory / f"{stem}-py3-none-any.whl"
    source = directory / f"{stem}.tar.gz"
    verify_members({p.name for p in directory.iterdir()}, {wheel.name, source.name})
    modules = {
        p.relative_to(root / "src").as_posix() for p in (root / "src/modelaudit").rglob("*.py")
    }
    metadata = stem + ".dist-info/"
    wheel_required = modules | {
        metadata + n
        for n in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set()
        for member in archive.infolist():
            name = safe_name(member.filename.rstrip("/")).as_posix()
            if stat.S_ISLNK(member.external_attr >> 16) or name in names:
                raise ValueError(f"Invalid wheel member {name}")
            if not member.is_dir():
                names.add(name)
        verify_members(names, wheel_required, {metadata + "top_level.txt"})
        entry = ConfigParser()
        entry.read_string(archive.read(metadata + "entry_points.txt").decode())
        if dict(entry["console_scripts"]) != {"modelaudit": "modelaudit.cli:main"}:
            raise ValueError("Invalid wheel console entry point")
    public = {
        p.relative_to(root).as_posix()
        for pattern in ("docs/*.md", "configs/*.yaml", "scripts/*.py", "tests/*.py")
        for p in root.glob(pattern)
    }
    source_required = (
        {"src/" + m for m in modules}
        | public
        | {
            "LICENSE",
            "README.md",
            "AGENTS.md",
            "constraints-python314.txt",
            "pyproject.toml",
            "MANIFEST.in",
            "PKG-INFO",
        }
    )
    generated = {
        "src/modelaudit.egg-info/" + n
        for n in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    with tarfile.open(source) as archive:
        names = set()
        for member in archive:
            parts = safe_name(member.name.rstrip("/")).parts
            if parts[0] != stem or not (member.isdir() or member.isfile()):
                raise ValueError(f"Invalid source member {member.name}")
            if member.isfile():
                name = "/".join(parts[1:])
                if name in names:
                    raise ValueError(f"Duplicate source member {name}")
                names.add(name)
        verify_members(names, source_required, generated | {"setup.cfg"})
    print(f"Verified {wheel.name} and {source.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("coverage", "archives"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.check == "coverage":
        check_coverage(args.path)
    else:
        check_archives(args.path, Path(__file__).resolve().parents[1])
