"""Prompt templates for demographic conditioning (P0, P1, P2, P3)."""

from typing import Optional


def format_p0_control(**_ignored) -> str:
    """P0: No demographics (control condition).

    build_prompt() always forwards the full **demographics dict to whichever
    template function is selected, so P0 -- the one condition that discards
    demographics by design -- must still accept and ignore those kwargs.
    """
    return "Please answer the following survey question:"


def format_p1_minimal(
    age: Optional[int] = None, sex: Optional[str] = None, region: Optional[str] = None, **_ignored
) -> str:
    """P1: Minimal demographics (age, sex, region only).

    build_prompt() forwards the full ~14-key demographics dict to whichever
    condition is selected; P1 deliberately uses only 3 of those keys, so it
    must accept (and drop) the rest via **_ignored -- same fix as P0 needed
    after it crashed on its first-ever real run.
    """
    demo_parts = []

    if age is not None:
        demo_parts.append(f"Age: {age}")
    if sex is not None:
        demo_parts.append(f"Sex: {sex}")
    if region is not None:
        demo_parts.append(f"Region: {region}")

    demographics = ", ".join(demo_parts) if demo_parts else "N/A"

    return f"""You are a survey respondent with the following characteristics:
{demographics}

Please answer the following survey question based on your background:"""


def format_p2_structured(
    sex: Optional[str] = None,
    age: Optional[int] = None,
    marital_status: Optional[str] = None,
    n_children: Optional[int] = None,
    education: Optional[str] = None,
    employment: Optional[str] = None,
    occupation: Optional[str] = None,
    social_class: Optional[str] = None,
    income_decile: Optional[str] = None,
    religion: Optional[str] = None,
    urban_rural: Optional[str] = None,
    region: Optional[str] = None,
    town_size: Optional[str] = None,
    interview_language: Optional[str] = None,
) -> str:
    """P2: Full structured list of 14 demographic attributes."""
    attributes = [
        ("Sex", sex),
        ("Age", age),
        ("Marital status", marital_status),
        ("Number of children", n_children),
        ("Education level", education),
        ("Employment status", employment),
        ("Occupation", occupation),
        ("Social class", social_class),
        ("Income decile", income_decile),
        ("Religion", religion),
        ("Urban/Rural", urban_rural),
        ("Region", region),
        ("Town size", town_size),
        ("Interview language", interview_language),
    ]

    # Only include non-None attributes
    attr_str = "\n".join(f"- {name}: {value}" for name, value in attributes if value is not None)

    return f"""You are a survey respondent with the following demographic profile:

{attr_str}

As this person, please answer the following survey question. Your answer should reflect the values and perspectives typical of someone with your background:"""


def format_p3_naturalistic(
    sex: Optional[str] = None,
    age: Optional[int] = None,
    occupation: Optional[str] = None,
    region: Optional[str] = None,
    urban_rural: Optional[str] = None,
    education: Optional[str] = None,
    religion: Optional[str] = None,
    **_ignored,
) -> str:
    """P3: Full naturalistic backstory (Argyle-style first-person prose).

    Uses 7 of the ~14 demographics keys build_prompt() forwards; **_ignored
    drops the rest instead of crashing (same bug class as P0/P1).
    """
    # Build a narrative persona based on demographics
    gender_pronoun = "I'm a woman" if sex == "Female" else "I'm a man" if sex == "Male" else "I'm a person"
    age_desc = f"{age} years old" if age else "in my working years"
    occ_desc = f"working as a {occupation}" if occupation else "working"
    edu_desc = f"with {education} education" if education else ""
    region_desc = f"in {region}" if region else "in India"
    urban_desc = f"in a {urban_rural.lower()} area" if urban_rural else "in my community"
    religion_desc = f"and practice {religion}" if religion else ""

    backstory = f"{gender_pronoun}, {age_desc}, {occ_desc} {edu_desc}. I live {urban_desc} {region_desc} {religion_desc}. "
    backstory += (
        "I have developed my worldview through my experiences and values shaped by my community and background. "
        "When I answer survey questions, I draw on my lived experience and perspective as someone in my situation."
    )

    return f"""Imagine you are this person: {backstory}

Please answer the following survey question as this person would, based on their background and values:"""


def get_prompt_template(condition: str) -> callable:
    """Get the prompt template function for a condition."""
    templates = {
        "P0": format_p0_control,
        "P1": format_p1_minimal,
        "P2": format_p2_structured,
        "P3": format_p3_naturalistic,
    }

    if condition not in templates:
        raise ValueError(f"Unknown condition: {condition}. Must be one of {list(templates.keys())}")

    return templates[condition]


def build_prompt(
    condition: str,
    question_text: str,
    answer_options: Optional[str] = None,
    **demographics
) -> str:
    """
    Build a full prompt for a respondent and question.

    Args:
        condition: P0, P1, P2, or P3
        question_text: The survey question
        answer_options: Optional text listing answer options
        **demographics: Demographic attributes (sex, age, region, etc.)

    Returns: Full prompt text
    """
    template_fn = get_prompt_template(condition)
    demographic_prompt = template_fn(**demographics)

    full_prompt = f"""{demographic_prompt}

Question: {question_text}"""

    if answer_options:
        full_prompt += f"\n\nAnswer options:\n{answer_options}"

    full_prompt += "\n\nYour answer:"

    return full_prompt
