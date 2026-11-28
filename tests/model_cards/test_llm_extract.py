from tests.agents.utils import RecordingChatModel
from ursa.model_cards.extract import CardDocument, extract_model_card
from ursa.model_cards.llm_extract import (
    CardExtraction,
    Quoted,
    QuotedIO,
    merge_extraction,
    numbers_supported,
    quote_in_source,
    refine_card,
)
from ursa.model_cards.schema import CardSource

CARD = """---
license: mit
---
# HeatNet

HeatNet emulates transient heat conduction in 2D plates. The network
receives the temperature field on a 96 x 96 grid and returns the field
after one step of 0.1 s.

Temperatures are given in kelvin and scaled to [0, 1] with fixed bounds.
It is not suited to convection-dominated problems.
"""


def base_card():
    return extract_model_card(
        CardDocument(
            source=CardSource(kind="file", uri="test://heatnet"), text=CARD
        )
    )


def test_quote_matching_tolerates_markup_and_wrapping():
    assert quote_in_source(
        "The network receives the temperature field on a 96 x 96 grid", CARD
    )
    assert quote_in_source(
        "HeatNet emulates transient heat conduction ... returns the field after one step",
        CARD,
    )
    assert not quote_in_source(
        "The network receives pressure fields on a 128 x 128 grid", CARD
    )
    assert not quote_in_source("short", CARD)


def test_numbers_must_appear_in_the_quote():
    assert numbers_supported("96 x 96", "on a 96 x 96 grid")
    assert not numbers_supported("128 x 128", "on a 96 x 96 grid")
    assert numbers_supported(
        "710867670", "Number of model parameters: 710,867,670"
    )


def test_verified_facts_fill_gaps_and_hallucinations_are_dropped():
    card = base_card()
    extraction = CardExtraction(
        tasks=[
            Quoted(
                value="surrogate_simulation",
                evidence="HeatNet emulates transient heat conduction in 2D plates.",
            )
        ],
        io=[
            QuotedIO(
                role="input",
                name="temperature",
                kind="field",
                shape="(96, 96)",
                units="K",
                normalization="scaled to [0, 1] with fixed bounds",
                evidence="The network receives the temperature field on a 96 x 96 grid",
            ),
            QuotedIO(
                role="output",
                name="heat_flux",
                shape="(128, 128)",
                evidence="returns the heat flux on a 128 x 128 grid",
            ),
        ],
        time_step=Quoted(
            value="0.1 s", evidence="returns the field after one step of 0.1 s"
        ),
        parameter_count=Quoted(
            value="45M", evidence="HeatNet has 45M parameters."
        ),
        limitations=[
            Quoted(
                value="Not for convection-dominated problems",
                evidence="It is not suited to convection-dominated problems.",
            )
        ],
    )
    merged = merge_extraction(card, extraction, CARD)

    temperature = next(s for s in merged.inputs if s.name == "temperature")
    assert temperature.shape == ["96", "96"]
    assert temperature.units == "K"
    assert merged.temporal.time_step == "0.1 s"
    assert "surrogate_simulation" in merged.tasks
    assert merged.limitations == ["Not for convection-dominated problems"]
    # The output quote and the parameter count are not in the card.
    assert all(s.name != "heat_flux" for s in merged.outputs)
    assert merged.architecture.parameter_count is None
    assert "llm_unverified" in merged.issue_codes()
    assert merged.provenance["temporal.time_step"].method == "llm"
    assert card.temporal.time_step is None, (
        "merge must not mutate the input card"
    )


def test_llm_values_never_override_structured_ones():
    card = base_card()
    card.architecture.parameter_count = 1_000
    card.license.commercial_use = True
    extraction = CardExtraction(
        parameter_count=Quoted(value="96", evidence="on a 96 x 96 grid"),
        license_restrictions=[
            Quoted(
                value="Research use only",
                evidence="This text is not in the card at all.",
            )
        ],
    )
    merged = merge_extraction(card, extraction, CARD)
    assert merged.architecture.parameter_count == 1_000
    assert merged.license.commercial_use is True


def test_refine_card_uses_the_llm_and_records_its_calls():
    extraction = CardExtraction(
        n_spatial_dims=Quoted(
            value="2", evidence="transient heat conduction in 2D plates"
        )
    )
    llm = RecordingChatModel(structured_factory=lambda schema: extraction)
    refined = refine_card(base_card(), CARD, llm)
    assert refined.spatial.n_dims == 2
    prompt = llm.calls[0][-1].content
    assert "<model_card>" in prompt and "HeatNet" in prompt


def test_refine_card_survives_unusable_llm_output():
    llm = RecordingChatModel(structured_factory=lambda schema: schema())
    refined = refine_card(base_card(), CARD, llm)
    assert refined.inputs == base_card().inputs


def test_changed_numbers_cannot_hide_in_the_fuzzy_match():
    # A long quote may differ in 10% of its word trigrams. Changing "0.1" to
    # "0.5" breaks only 3 of 41 trigrams, so the number needs its own check.
    quote = (
        "HeatNet emulates transient heat conduction in 2D plates. The network "
        "receives the temperature field on a 96 x 96 grid and returns the field "
        "after one step of {step} s. Temperatures are given in kelvin and scaled "
        "to [0, 1] with fixed bounds."
    )
    assert quote_in_source(quote.format(step="0.1"), CARD)
    assert quote_in_source(
        quote.format(step="0.1").replace("transient heat", "transient thermal"),
        CARD,
    )
    assert not quote_in_source(quote.format(step="0.5"), CARD)


def test_llm_cannot_quote_comments_or_template_text():
    text = CARD + "\n<!-- Trained on 4096 GPUs for 3 years. -->\n"
    extraction = CardExtraction(
        min_gpu_memory_gb=Quoted(
            value="4096", evidence="Trained on 4096 GPUs for 3 years."
        )
    )
    llm = RecordingChatModel(structured_factory=lambda schema: extraction)
    refined = refine_card(base_card(), text, llm)
    assert refined.compute.min_gpu_memory_gb is None
    assert "4096" not in llm.calls[0][-1].content
    assert "llm_unverified" in refined.issue_codes()
