import difflib
import re
from dataclasses import dataclass

from rich.console import Console, ConsoleOptions, RenderResult
from rich.syntax import Syntax
from rich.text import Text

# unified diff hunk header regex
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class _LineStyle:
    prefix: str
    bg: str


_STYLE = {
    "add": _LineStyle("+ ", "on #003000"),
    "del": _LineStyle("- ", "on #300000"),
    "ctx": _LineStyle("  ", "on grey15"),
}


class DiffRenderer:
    """Renderable diff—`console.print(DiffRenderer(...))`"""

    def __init__(self, content: str, updated: str, filename: str):
        # total lines in each version
        self._old_total = len(content.splitlines())
        self._new_total = len(updated.splitlines())

        # number of digits in the largest count
        self._num_width = len(str(max(self._old_total, self._new_total))) + 2

        # get the diff
        self._diff_lines = list(
            difflib.unified_diff(
                content.splitlines(),
                updated.splitlines(),
                fromfile=f"{filename} (original)",
                tofile=f"{filename} (modified)",
                lineterm="",
            )
        )

        # get syntax style
        try:
            self._lexer_name = Syntax.guess_lexer(filename, updated)
        except Exception:
            self._lexer_name = "text"

    def __rich_console__(
        self, console: Console, opts: ConsoleOptions
    ) -> RenderResult:
        old_line = new_line = None
        width = console.width

        for raw in self._diff_lines:
            # grab line numbers from hunk header
            if m := _HUNK_RE.match(raw):
                old_line, new_line = map(int, m.groups())
                # build a marker
                n = self._num_width
                tick_col = "." * (n - 1)
                indent_ticks = f" {tick_col} {tick_col}"
                # pad to the indent width
                full_indent = indent_ticks.ljust(2 * n + 3)
                yield Text(
                    f"{full_indent}{raw}".ljust(width), style="white on grey30"
                )
                continue

            # skip header lines
            if raw.startswith(("---", "+++")):
                continue

            # split the line
            if raw.startswith("+"):
                style = _STYLE["add"]
                code = raw[1:]
            elif raw.startswith("-"):
                style = _STYLE["del"]
                code = raw[1:]
            else:
                style = _STYLE["ctx"]
                code = raw[1:] if raw.startswith(" ") else raw

            # compute line numbers
            if raw.startswith("+"):
                old_num, new_num = None, new_line
                new_line += 1
            elif raw.startswith("-"):
                old_num, new_num = old_line, None
                old_line += 1
            else:
                old_num, new_num = old_line, new_line
                old_line += 1
                new_line += 1

            old_str = str(old_num) if old_num is not None else " "
            new_str = str(new_num) if new_num is not None else " "

            # Syntax-highlight the code part
            syntax = Syntax(
                code, self._lexer_name, line_numbers=False, word_wrap=False
            )
            text_code: Text = syntax.highlight(code)
            if text_code.plain.endswith("\n"):
                text_code = text_code[:-1]
            # apply background
            text_code.stylize(style.bg)

            # line numbers + code
            nums = Text(
                f"{old_str:>{self._num_width}}{new_str:>{self._num_width}} ",
                style=f"white {style.bg}",
            )
            diff_mark = Text(style.prefix, style=f"bright_white {style.bg}")
            line_text = nums + diff_mark + text_code

            # pad to console width
            pad_len = width - line_text.cell_len
            if pad_len > 0:
                line_text.append(" " * pad_len, style=style.bg)

            yield line_text
