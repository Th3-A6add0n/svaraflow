#!/usr/bin/env python3

from pathlib import Path
import argparse


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    ".cache",
    "coverage",
}


def build_tree(root: Path, ignored_dirs: set[str]) -> list[str]:
    lines = [root.name]

    def walk(directory: Path, prefix: str = ""):
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower())
            )
        except PermissionError:
            return

        entries = [
            entry
            for entry in entries
            if not (entry.is_dir() and entry.name in ignored_dirs)
        ]

        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1

            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + entry.name)

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension)

    walk(root)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate an ASCII tree structure of the current repository."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="repo-tree",
        help="Output filename prefix (default: repo-tree)",
    )

    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )

    args = parser.parse_args()

    root = Path.cwd().resolve()

    ignored_dirs = set(DEFAULT_IGNORED_DIRS)

    # By default, ignore hidden directories/files.
    lines = build_tree(root, ignored_dirs)

    if not args.include_hidden:
        # Rebuild while filtering hidden entries.
        def build_without_hidden():
            result = [root.name]

            def walk(directory: Path, prefix: str = ""):
                try:
                    entries = sorted(
                        directory.iterdir(),
                        key=lambda p: (p.is_file(), p.name.lower())
                    )
                except PermissionError:
                    return

                entries = [
                    entry
                    for entry in entries
                    if not entry.name.startswith(".")
                    and not (
                        entry.is_dir()
                        and entry.name in ignored_dirs
                    )
                ]

                for index, entry in enumerate(entries):
                    is_last = index == len(entries) - 1

                    branch = "└── " if is_last else "├── "
                    result.append(prefix + branch + entry.name)

                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        walk(entry, prefix + extension)

            walk(root)
            return result

        lines = build_without_hidden()

    tree = "\n".join(lines)

    txt_file = root / f"{args.output}.txt"
    md_file = root / f"{args.output}.md"

    txt_file.write_text(tree + "\n", encoding="utf-8")

    markdown = f"# Repository Tree\n\n```text\n{tree}\n```\n"
    md_file.write_text(markdown, encoding="utf-8")

    print(f"Repository: {root}")
    print(f"Created: {txt_file}")
    print(f"Created: {md_file}")


if __name__ == "__main__":
    main()