from rich import box, get_console
from rich.table import Table
from rich.text import Text


def render_plan_steps_rich(plan_steps, highlight_index: int | None = None):
    """Pretty table for a list of plan steps (strings or dicts), with an optional highlighted row."""

    console = get_console()

    if not plan_steps:
        return

    table = Table(
        title="Planned Steps",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold magenta",
        expand=True,
        row_styles=None,  # we'll control per-row styles manually
    )
    table.add_column("#", style="bold cyan", no_wrap=True)
    table.add_column("Name", style="bold", overflow="fold")
    table.add_column("Description", overflow="fold")
    table.add_column("Outputs", overflow="fold")
    table.add_column("Criteria", overflow="fold")
    table.add_column("Code?", justify="center", no_wrap=True)

    def bullets(items):
        if not items:
            return ""
        return "\n".join(f"• {x}" for x in items)

    def code_badge(needs_code: bool):
        return Text.from_markup(
            ":hammer_and_wrench: [bold green]Yes[/]"
            if needs_code
            else "[bold red]No[/]"
        )

    for i, step in enumerate(plan_steps, 1):
        # build cells
        if isinstance(step, dict):
            name = (
                step.get("name")
                or step.get("title")
                or step.get("id")
                or f"Step {i}"
            )
            desc = step.get("description") or ""
            outs = bullets(
                step.get("expected_outputs") or step.get("artifacts")
            )
            crit = bullets(step.get("success_criteria"))
            needs_code = bool(step.get("requires_code"))
        else:
            name, desc, outs, crit, needs_code = (
                f"Step {i}",
                str(step),
                "",
                "",
                False,
            )

        # style logic
        row_style = None
        idx0 = i - 1
        step_label = str(i)

        if highlight_index is not None:
            if idx0 < highlight_index:
                row_style = "dim"
            elif idx0 == highlight_index:
                row_style = "bold white on grey50"  # light gray
                # row_style = "bold black on bright_green"
                step_label = f"▶ {i}"  # pointer on current row

        table.add_row(
            step_label,
            str(name),
            str(desc),
            outs,
            crit,
            code_badge(needs_code),
            style=row_style,
        )

    console.print(table)
