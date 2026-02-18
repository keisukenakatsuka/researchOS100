# src/values/generator.py
"""Value generation logic — seed data for 12 life value domains.

This module contains the initial seed definitions for all value domains.
Each domain includes:
- An identity-oriented value definition (who you are, not what you do)
- A behavioral translation (how the value becomes visible in action)
- Three concrete example behaviors (small, executable today)
- A misalignment description (what it looks like when off-track)
- Reflection questions (for voice + avatar sessions)
- Micro habits (smallest viable daily actions)

Domains (12):
  1. Family                       7. Leisure
  2. Marriage / Romantic           8. Spirituality
  3. Parenting                     9. Community / Social Life
  4. Friendships / Interpersonal  10. Health
  5. Career / Work                11. Environment
  6. Personal Growth              12. Creative / Arts

These are starting points for quarterly review. The user refines them
over time through reflection cycles and optional LLM enhancement.

Pure functions — no I/O, no LLM calls, no side effects.
"""

from __future__ import annotations

from src.values.schema import (
    ValueBehavior,
    ValueDomain,
    ValueRecord,
)


# ----------------------------------------------------------------
# Seed data: 12 domains
# ----------------------------------------------------------------

def _domain_family() -> ValueDomain:
    return ValueDomain(
        domain_id="family",
        domain_label="Family",
        value_definition=(
            "I am someone who shows up as a grounded, present, and "
            "emotionally available member of my family system."
        ),
        behavioral_translation=(
            "Prioritizing genuine connection over logistics, creating "
            "rituals that reinforce belonging, and being the kind of "
            "family member I needed growing up."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Have one fully present, phone-free interaction with a family member",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Initiate a check-in conversation with a family member I haven't connected with recently",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Protect and lead one shared family ritual (meal, call, outing)",
                frequency_hint="weekly",
            ),
        ),
        misalignment_description=(
            "Treating family as background obligation; being physically present "
            "but emotionally unavailable; letting logistics replace connection."
        ),
        reflection_questions=(
            "How do I want to show up as a family member?",
            "What kind of presence do I want to embody within my family system?",
            "Am I creating the family culture I want, or inheriting one by default?",
        ),
        micro_habits=(
            "Put phone away during shared family time",
            "Name one specific thing you appreciate about a family member today",
            "Ask 'how are you really doing?' and wait for the full answer",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_marriage_romantic() -> ValueDomain:
    return ValueDomain(
        domain_id="marriage_romantic",
        domain_label="Marriage / Romantic Relationship",
        value_definition=(
            "I am someone who builds deep intimacy through honest presence, "
            "intentional vulnerability, and evolving partnership."
        ),
        behavioral_translation=(
            "Prioritizing undistracted connection time with my partner, "
            "expressing needs directly, and investing in the relationship "
            "as an evolving practice rather than a static state."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Have one 15-minute phone-free conversation with partner",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Express one specific appreciation or gratitude to partner",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Plan and protect one dedicated date block together",
                frequency_hint="weekly",
            ),
        ),
        misalignment_description=(
            "Defaulting to parallel screen time instead of conversation; "
            "avoiding difficult topics to keep surface-level peace; "
            "treating the relationship as maintenance rather than growth."
        ),
        reflection_questions=(
            "What kind of partner do I want to be?",
            "If I were my ideal self, how would I relate to my partner?",
            "What am I avoiding saying that needs to be said?",
        ),
        micro_habits=(
            "Put phone face-down during meals together",
            "Say 'thank you for...' with one specific detail before sleep",
            "Ask 'what's on your mind?' and wait for the full answer",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_parenting() -> ValueDomain:
    return ValueDomain(
        domain_id="parenting",
        domain_label="Parenting",
        value_definition=(
            "I am someone who creates a stable, curious, and emotionally "
            "safe environment for my children to grow into themselves."
        ),
        behavioral_translation=(
            "Being consistently present during family time, modeling "
            "emotional regulation, and nurturing qualities in my children "
            "that I believe matter most."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Spend 20 minutes of undivided attention with each child",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Ask one open-ended question about their day and listen fully",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Lead one shared activity (meal, game, outing) focused on connection",
                frequency_hint="weekly",
            ),
        ),
        misalignment_description=(
            "Being physically present but mentally checked out; "
            "outsourcing connection to screens or structured activities; "
            "reacting to children instead of responding with intention."
        ),
        reflection_questions=(
            "What kind of parent do I want to be?",
            "What qualities do I want to nurture in my children?",
            "What emotional pattern am I modeling that I want my children to inherit?",
        ),
        micro_habits=(
            "Make eye contact and pause work when a child speaks to you",
            "Name your own emotions out loud as a modeling practice",
            "Create one 'no screens' transition moment at homecoming",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_friendships_interpersonal() -> ValueDomain:
    return ValueDomain(
        domain_id="friendships_interpersonal",
        domain_label="Friendships / Interpersonal Relationships",
        value_definition=(
            "I am someone who cultivates a small circle of reciprocal, "
            "honest relationships where I use my strengths to show up "
            "meaningfully."
        ),
        behavioral_translation=(
            "Initiating contact without waiting for the other person, "
            "showing up during hard times, and being honest rather than "
            "performatively agreeable."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Send one genuine check-in message to a friend",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Schedule one in-person or voice catch-up with a close friend",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Offer specific help to someone going through difficulty",
                frequency_hint="as-needed",
            ),
        ),
        misalignment_description=(
            "Letting months pass without reaching out; only connecting "
            "when you need something; staying in shallow group chats "
            "instead of deepening individual bonds."
        ),
        reflection_questions=(
            "What kind of friend do I want to be?",
            "How do I want to use my strengths in relationships?",
            "Am I showing up as the friend I want to have?",
        ),
        micro_habits=(
            "When you think of someone, message them immediately instead of later",
            "Replace one group-chat reply with a direct personal message",
            "Ask one question you actually want the answer to",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_career_work() -> ValueDomain:
    return ValueDomain(
        domain_id="career_work",
        domain_label="Career / Work",
        value_definition=(
            "I am someone who builds meaningful work that compounds "
            "skill, reputation, and economic independence over time."
        ),
        behavioral_translation=(
            "Choosing deep-work blocks over reactive busywork, saying no "
            "to misaligned opportunities, and shipping tangible outputs "
            "that move long-term contribution forward."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Complete one 90-minute deep-work block on highest-leverage task",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Review weekly priorities and decline one low-value request",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Document one learning or decision for future reference",
                frequency_hint="daily",
            ),
        ),
        misalignment_description=(
            "Spending the day in meetings and messages with no tangible "
            "output; saying yes to everything; confusing motion with progress."
        ),
        reflection_questions=(
            "What role does work play in my life?",
            "How do I want to contribute through my work?",
            "Am I building skills that compound, or just completing tasks?",
        ),
        micro_habits=(
            "Write tomorrow's single most important task before closing work",
            "Block the first 90 minutes of work before opening email",
            "End each work day with a 2-sentence log of what shipped",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_personal_growth() -> ValueDomain:
    return ValueDomain(
        domain_id="personal_growth",
        domain_label="Personal Growth",
        value_definition=(
            "I am someone who stays intellectually hungry, seeks "
            "disconfirming evidence, and converts learning into "
            "changed behavior."
        ),
        behavioral_translation=(
            "Reading and reflecting consistently, engaging with ideas "
            "outside my comfort zone, and applying insights to real "
            "decisions rather than collecting knowledge passively."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Read for 30 minutes from a book or long-form source",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Write one reflection note connecting reading to current work or life",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Seek out one perspective that challenges a current belief",
                frequency_hint="weekly",
            ),
        ),
        misalignment_description=(
            "Consuming content without reflection; reading only within "
            "existing worldview; treating learning as entertainment "
            "rather than growth."
        ),
        reflection_questions=(
            "What do I want to keep learning?",
            "What qualities do I want to cultivate?",
            "What did I learn this week that actually changed how I think or act?",
        ),
        micro_habits=(
            "Replace 10 minutes of scrolling with 10 minutes of reading",
            "Write one sentence summarizing the most important thing you learned today",
            "Ask 'what would change my mind?' before defending a position",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_leisure() -> ValueDomain:
    return ValueDomain(
        domain_id="leisure",
        domain_label="Leisure",
        value_definition=(
            "I am someone who protects play, novelty, and spontaneity "
            "as essential — not optional — parts of a well-lived life."
        ),
        behavioral_translation=(
            "Scheduling fun with the same seriousness as work, trying "
            "new experiences regularly, and resisting the urge to "
            "optimize every moment."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Do one thing purely for enjoyment with no productivity goal",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Try one new experience, place, or activity",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Plan one larger adventure or trip for the upcoming month",
                frequency_hint="monthly",
            ),
        ),
        misalignment_description=(
            "Treating all leisure as 'wasted time'; never trying "
            "anything new; defaulting to passive consumption "
            "(social media, TV) instead of active play."
        ),
        reflection_questions=(
            "What activities bring me relaxation and joy?",
            "How do I recharge?",
            "Am I scheduling rest or just collapsing into it?",
        ),
        micro_habits=(
            "Block one 30-minute 'no agenda' window per day",
            "Say yes to the next spontaneous invitation that comes your way",
            "Keep a running 'things to try' list and pick one per week",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_spirituality() -> ValueDomain:
    return ValueDomain(
        domain_id="spirituality",
        domain_label="Spirituality",
        value_definition=(
            "I am someone who cultivates inner stillness, perspective, "
            "and a relationship with meaning that transcends daily "
            "circumstances."
        ),
        behavioral_translation=(
            "Practicing regular reflection or meditation, noticing "
            "thought patterns without being controlled by them, and "
            "returning to core principles when under pressure."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Sit in silence or meditate for 10 minutes",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Journal one observation about a recurring thought pattern",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Re-read and reflect on personal principles or philosophy notes",
                frequency_hint="monthly",
            ),
        ),
        misalignment_description=(
            "Reacting to every emotion on autopilot; no reflective "
            "practice; mistaking busyness for meaning."
        ),
        reflection_questions=(
            "What is my relationship to religion, philosophy, or deeper meaning?",
            "Am I responding to life, or just reacting?",
            "What would I do differently if I weren't afraid?",
        ),
        micro_habits=(
            "Take three conscious breaths before picking up your phone in the morning",
            "Notice one emotional reaction without acting on it immediately",
            "End each day by naming one thing you're grateful for without repeating",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_community_social() -> ValueDomain:
    return ValueDomain(
        domain_id="community_social",
        domain_label="Community / Social Life",
        value_definition=(
            "I am someone who contributes to communities beyond my "
            "immediate circle, creating value that outlasts individual "
            "transactions."
        ),
        behavioral_translation=(
            "Sharing knowledge and resources generously, participating "
            "in communities I care about, and building things that "
            "help others succeed."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Share one useful insight, resource, or introduction publicly",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Actively participate in one community event or discussion",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Mentor or advise someone earlier in their journey",
                frequency_hint="monthly",
            ),
        ),
        misalignment_description=(
            "Only consuming from communities without contributing; "
            "hoarding knowledge; isolating from broader networks."
        ),
        reflection_questions=(
            "What kind of community member do I want to be?",
            "How do I want to contribute locally?",
            "What knowledge do I have that someone else needs right now?",
        ),
        micro_habits=(
            "Reply to one post or question in a community you value",
            "Forward one useful resource to someone who would benefit",
            "Introduce two people who should know each other",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_health() -> ValueDomain:
    return ValueDomain(
        domain_id="health",
        domain_label="Health",
        value_definition=(
            "I am someone who treats physical and mental health as "
            "non-negotiable infrastructure for everything else."
        ),
        behavioral_translation=(
            "Moving my body daily, protecting sleep, eating with "
            "intention, and processing stress rather than numbing it."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Complete at least 30 minutes of intentional physical movement",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Begin wind-down routine by target bedtime",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Do one stress-processing practice (journal, walk, breathwork)",
                frequency_hint="daily",
            ),
        ),
        misalignment_description=(
            "Skipping exercise for 'urgent' work; staying up late "
            "doom-scrolling; eating reactively out of stress."
        ),
        reflection_questions=(
            "How do I want to care for my physical body?",
            "What lifestyle supports my long-term vitality?",
            "Is my sleep protecting my capacity or undermining it?",
        ),
        micro_habits=(
            "Set a phone alarm for wind-down time, not just wake-up time",
            "Do 5 minutes of movement before the first work task",
            "Drink a full glass of water before reaching for coffee",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_environment() -> ValueDomain:
    return ValueDomain(
        domain_id="environment",
        domain_label="Environment",
        value_definition=(
            "I am someone who designs my physical surroundings to "
            "support focus, calm, and the kind of life I want to live."
        ),
        behavioral_translation=(
            "Keeping spaces intentionally organized, removing friction "
            "from daily routines through environment design, and "
            "relating to my surroundings and the planet with care."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Do a 10-minute reset of primary workspace before deep work",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Remove or donate one item that no longer serves a purpose",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Review one area of the home for friction points and fix one",
                frequency_hint="monthly",
            ),
        ),
        misalignment_description=(
            "Living in accumulated clutter; workspace that creates "
            "distraction rather than focus; ignoring the impact of "
            "surroundings on daily well-being."
        ),
        reflection_questions=(
            "How do I want to relate to my physical surroundings and the planet?",
            "What daily friction could I eliminate by changing my environment?",
            "Does my workspace make me want to do great work, or escape?",
        ),
        micro_habits=(
            "Clear desk surface completely before starting work",
            "Put one thing back in its place every time you stand up",
            "Do a 2-minute scan of your space before leaving a room",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


def _domain_creative_arts() -> ValueDomain:
    return ValueDomain(
        domain_id="creative_arts",
        domain_label="Creative / Arts",
        value_definition=(
            "I am someone who creates and ships original work as a "
            "practice of self-expression and meaning-making."
        ),
        behavioral_translation=(
            "Making time for creative output separate from 'productive' "
            "work, finishing and sharing creative projects, and engaging "
            "with artistic expression as essential rather than indulgent."
        ),
        example_behaviors=(
            ValueBehavior(
                description="Spend 20 minutes on a creative project (writing, design, music, code)",
                frequency_hint="daily",
            ),
            ValueBehavior(
                description="Share or publish one piece of creative work",
                frequency_hint="weekly",
            ),
            ValueBehavior(
                description="Complete and ship one creative project to a defined 'done' state",
                frequency_hint="monthly",
            ),
        ),
        misalignment_description=(
            "Always 'planning to start' creative work but never "
            "beginning; perfectionism preventing shipping; consuming "
            "others' creative output without producing your own."
        ),
        reflection_questions=(
            "How do I engage with artistic expression?",
            "What role does creativity play in my life?",
            "Am I consuming more than I'm creating?",
        ),
        micro_habits=(
            "Open your creative project before opening email each morning",
            "Set a 20-minute timer and create without editing",
            "Share one work-in-progress with someone you trust",
        ),
        source="Manual",
        version=1,
        revision=0,
    )


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

_DOMAIN_GENERATORS = (
    _domain_family,
    _domain_marriage_romantic,
    _domain_parenting,
    _domain_friendships_interpersonal,
    _domain_career_work,
    _domain_personal_growth,
    _domain_leisure,
    _domain_spirituality,
    _domain_community_social,
    _domain_health,
    _domain_environment,
    _domain_creative_arts,
)


def generate_all_domains() -> tuple[ValueDomain, ...]:
    """Generate seed data for all 12 value domains.

    Returns a tuple of ValueDomain dataclasses.
    Pure function — no I/O, no side effects.
    """
    return tuple(gen() for gen in _DOMAIN_GENERATORS)


def generate_value_record(
    *,
    review_quarter: str,
    version: str = "2.0",
    notes: str = "",
) -> ValueRecord:
    """Generate a complete ValueRecord with all 12 seed domains.

    Parameters
    ----------
    review_quarter : str
        Quarter identifier, e.g. "2026-Q1".
    version : str
        Schema version for forward compatibility.
    notes : str
        Free-form notes about this review cycle.

    Returns
    -------
    ValueRecord
        Complete record ready for serialization.
    """
    return ValueRecord(
        version=version,
        review_quarter=review_quarter,
        domains=generate_all_domains(),
        notes=notes,
    )
