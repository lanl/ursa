from ursa.skills.activation import (
    annotate_skill_requests,
    requested_skill_names,
)
from ursa.skills.discovery import Skill


def catalog(*names):
    return {
        name: Skill(
            name=name,
            description=f"Does {name}",
            instructions="Body",
            path=None,
            scope="project",
        )
        for name in names
    }


def test_finds_a_reference_anywhere_in_the_prompt():
    known = catalog("write-python")

    assert requested_skill_names("please $write-python this", known) == [
        "write-python"
    ]


def test_finds_multiple_references_in_order_without_duplicates():
    known = catalog("alpha", "beta")

    names = requested_skill_names("$beta then $alpha and $beta again", known)

    assert names == ["beta", "alpha"]


def test_ignores_references_to_unknown_skills():
    assert requested_skill_names("$nope", catalog("known")) == []


def test_ignores_shell_variables_and_currency():
    known = catalog("PATH")

    assert requested_skill_names("costs US$5", known) == []
    assert requested_skill_names("echo $$PATH", known) == []


def test_matches_a_shell_style_reference_when_it_names_a_skill():
    assert requested_skill_names("echo $PATH", catalog("PATH")) == ["PATH"]


def test_trailing_sentence_punctuation_is_not_part_of_the_name():
    known = catalog("write-python")

    assert requested_skill_names("use $write-python.", known) == [
        "write-python"
    ]


def test_prompt_is_unchanged_when_nothing_matches():
    prompt = "Just a normal question about $money."

    assert annotate_skill_requests(prompt, catalog("write-python")) == prompt


def test_empty_catalog_short_circuits():
    assert requested_skill_names("$anything", {}) == []


def test_directive_is_appended_after_the_original_prompt():
    annotated = annotate_skill_requests("Do it $alpha", catalog("alpha"))

    assert annotated.startswith("Do it $alpha")
    assert "explicitly requested these skills: alpha" in annotated
