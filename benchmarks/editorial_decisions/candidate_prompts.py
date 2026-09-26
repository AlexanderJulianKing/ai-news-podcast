"""Candidate revisions to the two Tier 3 selection prompts. Not used by the show.

Each candidate is the production prompt with named edits applied, so the diff
stays explicit and a change to the production text that an edit depends on
fails loudly instead of drifting.

Why these edits (evidence: 20 mornings labeled by Alex, 2026-09-20; see README):

running_story   Alex led with a ceasefire tagged [DEVELOPMENT] and rejected a
                same-pattern strike with the same tag. The old wording ("with
                care... when it is close, choose the fresh story") cannot tell
                those apart and erred in both directions.
lasting_change  Every tier led a Speaker's removal over fusion power reaching
                the grid, and no tier led official AI job-loss data in 30 runs.
                Alex led both. The lenses reward institutional reaction and
                have no place for a lasting change that forces no one to react.
foreign_second  The second-slot ban on stories abroad blocked Alex's pick twice
                (a strike on Iran, US troops in Mexico).
living_through  The second slot's sourcing preference pushed every tier off a
                confirmed payments outage whose cause was unconfirmed.

second_slot     (variant 'candidate2') Alex defined the slot on 2026-09-20 as
                "second most important, with a lean toward the average
                Californian". The production prompt asks a different question
                (what matters most to an average Californian), so models chose
                consumer-health stories where Alex chose the next biggest story.
                candidate2 keeps the candidate lead prompt and replaces the
                second-slot question; it drops foreign_second and
                living_through, which showed no effect and which the new
                question makes unnecessary.

These edits were written after seeing the labels they are scored against, so
any gain on those mornings is in-sample. Mornings labeled later are the test.
"""
import re

from newscaster.prompts import TIER3_EVERYMAN_STORY_PROMPT, TIER3_IMPORTANT_STORY_PROMPT


def _edit(text, old, new, name):
    if text.count(old) != 1:
        raise ValueError("candidate edit '{}' no longer matches the production prompt".format(name))
    return text.replace(old, new)


_RUNNING_STORY_OLD = (
    "- '[DEVELOPMENT]' stories CAN be selected, with care. The audience heard a full segment on this story at least two days ago; "
    "the Coverage notes say when, and what they already know. Lead with it only when the new development, judged on its own by the "
    "lenses above, is clearly the day's most important story. When it is close, choose the fresh story. If you select it, your "
    "reasoning must say what is new against what the audience already knows."
)
_RUNNING_STORY_NEW = (
    "- '[DEVELOPMENT]' stories CAN be selected. The audience heard a full segment on this story at least two days ago; the Coverage "
    "notes say when, and what they already know. Ask one question: has the story changed in kind, or is this more of the same? "
    "A change in kind is a turn the audience could not have predicted from what they already know: fighting starts or stops, a deal "
    "is signed or collapses, a court rules, a standoff produces its first direct effect on the public. Judge a change in kind on the "
    "lenses above exactly like a fresh story, with no penalty for the tag. More of the same is another instance of the pattern the "
    "audience already knows: another day of strikes, another day without a vote, another statement. More of the same must not be "
    "selected, however heavily it is covered; it belongs in the roundup. If you select a '[DEVELOPMENT]' story, your reasoning must "
    "say what changed in kind against what the audience already knows."
)

_LASTING_OLD = "If the three lenses point to different stories, prefer the one with broader implications for more Americans.\n"
_LASTING_NEW = (
    "LENS 4 — LASTING CHANGE: Will this still matter in five years? Some stories force no institution to react today and still "
    "change what is possible or how people live: a first in science, medicine, or energy; a measured shift in how Americans work, "
    "earn, or stay healthy. Check: (a) it is a first, a threshold, or a measured change, not a forecast or a claim; (b) it is attested "
    "by someone other than the party that benefits — a regulator, official statistics, a peer-reviewed trial, an independent "
    "observer; (c) its reach is wide, touching millions of people or a whole industry.\n\n"
    "If the lenses point to different stories, prefer the one with broader implications for more Americans. When one contender is "
    "turmoil that will resolve within weeks (a leadership fight, a deadline standoff, a one-day market move) and another is a lasting "
    "change that passes Lens 3 or Lens 4, prefer the lasting change unless the turmoil has already harmed the public.\n"
)

_DUPLICATE_OLD = "If both lenses point to different stories, prefer the one with broader implications for more Americans. The story must be specific"
_DUPLICATE_NEW = "The story must be specific"

_FOREIGN_OLD = "Do not pick something that is happening in a non-US country. "
_FOREIGN_NEW = (
    "A story set abroad qualifies only when Americans are directly involved or directly affected: US troops or US hostages, "
    "or prices and supplies Americans depend on. Do not pick a foreign story with no such link. "
)

_SOURCING_OLD = "Base your decision on the research briefs — prefer stories with strong sourcing and concrete details over thinly sourced claims. "
_SOURCING_NEW = (
    "Base your decision on the research briefs — prefer stories with strong sourcing and concrete details over thinly sourced claims. "
    "A disruption people are living through today (payments failing, flights grounded, a drug pulled from shelves) qualifies on its "
    "confirmed effects even when its cause is unconfirmed. "
)

_EVERYMAN_DEV_OLD = (
    "- '[DEVELOPMENT]' stories CAN be selected, with care: the audience heard a full segment at least two days ago (the Coverage notes "
    "say when and what they know). Lead with it only when the new development is clearly the most important story for Californians on "
    "its own; when it is close, choose the fresh story, and if you select it say what is new."
)
_EVERYMAN_DEV_NEW = (
    "- '[DEVELOPMENT]' stories CAN be selected: the audience heard a full segment at least two days ago (the Coverage notes say when "
    "and what they know). Select one only when the story has changed in kind — fighting starts or stops, a deal is signed or "
    "collapses, a standoff produces its first direct effect on the public — and then judge it like a fresh story. Another instance of "
    "the pattern the audience already knows must not be selected. If you select it, say what changed in kind."
)


_SLOT_OLD = (
    "Given the following research briefs on today\'s candidate stories, what would you say is the most important news story for the "
    "average person in California? If there is nothing directly impacting Californians, pick something that would affect the everyday "
    "American. Do not pick something that is happening in a non-US country. "
)
_SLOT_NEW = (
    "Given the following research briefs on today\'s candidate stories, pick the second main story for the show. The lead story is "
    "already chosen and is named below. Pick the most important story that remains, judged the way an editor judges a front page: "
    "how many people it affects, how much it changes, and how solid the reporting is. Then apply one lean. The audience lives in "
    "California, so when two stories are close in importance, prefer the one an average Californian will feel in daily life: their "
    "money, health, safety, or state. The lean breaks ties. It does not lift a minor local or consumer story over a clearly bigger "
    "one. A story set abroad qualifies when it is among the day\'s most important or when Americans are directly involved. "
)


def candidate_important():
    text = TIER3_IMPORTANT_STORY_PROMPT
    text = _edit(text, _LASTING_OLD, _LASTING_NEW, "lasting_change")
    text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    text = _edit(text, _RUNNING_STORY_OLD, _RUNNING_STORY_NEW, "running_story")
    return text


def candidate_everyman():
    """Still a template: format with excluded_headline, like the production prompt."""
    text = TIER3_EVERYMAN_STORY_PROMPT
    text = _edit(text, _FOREIGN_OLD, _FOREIGN_NEW, "foreign_second")
    text = _edit(text, _SOURCING_OLD, _SOURCING_NEW, "living_through")
    text = _edit(text, _EVERYMAN_DEV_OLD, _EVERYMAN_DEV_NEW, "running_story_second")
    return text


def candidate2_everyman():
    """The second slot redefined: second most important, leaning Californian."""
    text = TIER3_EVERYMAN_STORY_PROMPT
    text = _edit(text, _SLOT_OLD, _SLOT_NEW, "second_slot")
    text = _edit(text, _EVERYMAN_DEV_OLD, _EVERYMAN_DEV_NEW, "running_story_second")
    return text


# --- Round 2 (2026-09-20): lead-prompt hypotheses, tested side by side on Gemma ---
#
# Round 1's LENS 4 ("lasting change") raised the score on the mornings it was
# written against and lowered it on fresh ones. Across 30 labeled mornings the
# steadiest disagreement is this: where every model misses Alex's lead, he led a
# verified change in ordinary people's lives (a medical first, health costs,
# coverage, job losses) and the models led a mid-size story about who holds
# power. Each variant below is one way of telling the model that.

_LIVES_NEW = (
    "LENS 4 — DIRECT EFFECT ON PEOPLE'S LIVES: Does the story report a completed, verified change to the health, income, costs, "
    "coverage, or work of millions of Americans? Examples: a treatment approved, or a medical first confirmed by a regulator or a "
    "published trial; insurance coverage lost or gained; premiums, prices, or benefits set; official statistics on how many people "
    "lost work or how they now work. Check: (a) it has happened or is officially scheduled, not proposed or forecast; (b) the number "
    "of people is stated and large; (c) the source is official data, a regulator, a court, or a published trial.\n\n"
    "If the lenses point to different stories, prefer the one with broader implications for more Americans. A story about who holds "
    "power (a resignation, a removal, a procedural ruling, a leadership fight, an executive order not yet in effect) outranks a Lens 4 "
    "story only when its effect on the public is direct and near. When that effect is indirect or months away, prefer the Lens 4 story.\n"
)

_TASTE_NOTE = (
    "\n\nEDITOR'S TASTE: The show's editor has labeled past mornings. Apply these patterns from his choices.\n"
    "- The number of outlets carrying a story is not a measure of its importance. A verified first reported by one specialist outlet "
    "can lead over a story carried everywhere.\n"
    "- Between a mid-size story about who holds power in Washington and a verified change in people's health, costs, coverage, or "
    "work that reaches millions, he leads with the second.\n"
    "- Power stories lead when they are themselves very large: a war starting or ending, a coup in a nuclear-armed state, troops "
    "deployed at home, a chamber of Congress changing its own rules, a Supreme Court ruling that takes effect nationwide, an "
    "emergency move by the Federal Reserve.\n"
    "- On AI he ranks verified real-world harm, safety failures attested under oath or by outside evaluators, and official data on "
    "jobs above funding rounds, hearings, and executive orders.\n"
    "- A running story leads only when it has turned; another day of the same pattern does not.\n"
)

# Round 3: round 2 showed the lives lens over-corrects toward pocketbook stories.
# The labels say something narrower: on mornings with a substantive AI story,
# Alex led with AI 8 times out of 9.
_AI_NOTE = (
    "\n\nAI PRIORITY: The show's editor treats artificial intelligence as the most consequential running subject of this period. "
    "When the candidates include an AI story with verified substance, lead with it unless another story is of the very largest kind: "
    "a war starting or ending, troops deployed at home, or a Supreme Court ruling that takes effect nationwide. An AI story has "
    "verified substance when it reports real-world harm caused by a deployed system, a safety failure attested under oath or by "
    "outside evaluators, official statistics on jobs lost to AI, or a capability confirmed by independent parties. Prefer measured "
    "effects on people and attested safety failures over government action on AI. Funding rounds, product launches, hearings, "
    "executive orders, and statements do not qualify on their own.\n"
)

# Round 4 (2026-09-20): story KIND. A plain read of all 30 labeled mornings, both
# passes, found that Alex's leads share a kind, not a size: changes to the rules,
# the frontier, or the international order. Across 81 stories that were incidents
# or anticipation he led with one once in 60 lead picks, while each model led
# with such stories on 8 of 30 mornings. Alex confirmed the reading ("it sounds
# like me") and said a statistic leads or not depending on the story it tells.
# The production LENS 1 ("who is forced to react", "high uncertainty about
# critical systems") rewards incidents, so v_kind replaces the lenses outright.
_KIND_CHANGES = (
    "Judge first by the KIND of story, then by its size.\n\n"
    "CHANGES LEAD. A change is a story after which the rules, the frontier of what is possible, or the international order are "
    "different from yesterday:\n"
    "1. A binding act of government, done and in effect: a law passed, a ruling by the Supreme Court or a federal court, a chamber "
    "of Congress changing its own rules, emergency powers invoked, an emergency move by the Federal Reserve, a regulation or order "
    "taking effect. Also the moment such an act reaches people's lives at scale: coverage lost or gained, benefits stopped, prices "
    "or premiums reset by policy.\n"
    "2. A verified first in medicine, science, or energy, confirmed by a regulator, a published trial, or independent observers. "
    "A company's own announcement does not count.\n"
    "3. Frontier artificial intelligence with substance: a capability or a loss of control attested by outsiders, a safety failure "
    "described under oath or by independent evaluators, official data on what AI is doing to jobs, autonomous systems being given "
    "real authority. Funding rounds, product launches, hearings, deals, and lawsuits are business news, not this.\n"
    "4. A change in the international order with great-power or nuclear stakes: a blockade begins or ends, a government falls in a "
    "nuclear-armed state, a war starts or stops.\n"
    "5. An official statistic, when it is the first measurement of something new or shows the country has crossed into a different "
    "state. A routine reading of a familiar gauge, even a bad one, is an indicator (see below).\n\n"
)
_KIND_INCIDENTS = (
    "INCIDENTS AND ANTICIPATION DO NOT LEAD when any change is among the candidates, however large or heavily covered they are:\n"
    "- disasters, accidents, and casualties, at home or abroad, including American military deaths\n"
    "- corporate deals, failures, recalls, and layoffs\n"
    "- forecasts, projections, approaching deadlines, scheduled talks, hearings, and anything an official is only considering\n"
    "- another day of a pattern the audience already knows: another night of strikes, another day without a vote\n"
    "- outages, strikes, and disruptions that will pass\n"
    "- resignations, removals, and impeachments: who holds an office matters less than what the office can now do\n"
    "- market moves and routine economic readings\n"
    "The rest of the show covers these. Lead with one only on a morning that has no change at all, and then pick the one that "
    "touches the most people directly.\n\n"
)
_KIND_TIEBREAK = (
    "Among several changes, prefer the one that reaches the most Americans and will still matter in years. Frontier AI and verified "
    "firsts rank with major acts of government, not below them. The number of outlets carrying a story says nothing about its kind.\n\n"
)
_LENSES_START = "Use two lenses to judge importance:"
_LENSES_END = "IMPORTANT: Base your decision on the research briefs provided."
_VIOLENCE_OLD = ("For violence, disasters, and tragedies: prioritize only when the scale forces federal or international response, "
                 "or when it signals system failure.\n\n")


def kind_important(full=True):
    text = TIER3_IMPORTANT_STORY_PROMPT
    start, end = text.index(_LENSES_START), text.index(_LENSES_END)
    if full:  # replace the lenses outright
        text = text[:start] + _KIND_CHANGES + _KIND_INCIDENTS + _KIND_TIEBREAK + text[end:]
        text = _edit(text, _VIOLENCE_OLD, "", "violence_paragraph")
    else:     # keep the production lenses, add only the exclusion
        text = text[:end] + _KIND_INCIDENTS + text[end:]
    text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    text = _edit(text, _RUNNING_STORY_OLD, _RUNNING_STORY_NEW, "running_story")
    return text


# Round 5: v_kind (lenses replaced) ranked badly inside the class; v_kind_lite (exclusion
# added beside the lenses) lost to LENS 1, e.g. "passes Lens 1 (Forced Reaction) at the
# highest scale" for a Speaker's removal. v_kind2 keeps the accountability and frontier
# lenses, which rank well, and rewrites LENS 1 itself.
_LENS1_OLD_START = "Use two lenses to judge importance:"
_LENS1_OLD_END = "LENS 2 — ACCOUNTABILITY OF POWER"
_LENS1_NEW = (
    "Use these lenses to judge importance:\n\n"
    "LENS 1 — THE RULES CHANGED: Did a government body do something binding that is now in effect? A law passed, a ruling by the "
    "Supreme Court or a federal court, a chamber of Congress changing its own rules, emergency powers invoked, an emergency move by "
    "the Federal Reserve, a regulation or order taking effect. The moment such an act reaches people's lives at scale also counts: "
    "coverage lost or gained, benefits stopped, prices or premiums reset by policy. Completed actions beat threats, proposals, and "
    "deadlines. A story does not pass this lens merely because institutions must respond to it.\n\n"
)
_LENS3_TAIL_OLD = "a new model with better scores does not, however large the scores.\n"
_LENS3_TAIL_NEW = (
    "a new model with better scores does not, however large the scores. Verified firsts in medicine, science, and energy pass this "
    "lens, as does official data on what AI is doing to jobs. AI funding rounds, product launches, hearings, deals, and lawsuits do not.\n\n"
    "LENS 4 — THE INTERNATIONAL ORDER CHANGED: a blockade begins or ends, a government falls in a nuclear-armed state, a war starts or "
    "stops, with great-power or nuclear stakes. Another exchange of fire in a conflict the audience already knows does not pass.\n"
)
_KIND2_INCIDENTS = _KIND_INCIDENTS.replace(
    "- resignations, removals, and impeachments: who holds an office matters less than what the office can now do\n",
    "- resignations, removals, and impeachments, even when they stall an institution: who holds an office matters less than what "
    "the office can now do\n",
).replace(
    "The rest of the show covers these.",
    "None of these passes a lens just because it is large, urgent, or forces institutions to respond. The rest of the show covers these.",
) + "Always select exactly one story, even on a slow morning.\n\n"


def kind2_important():
    text = TIER3_IMPORTANT_STORY_PROMPT
    start, end = text.index(_LENS1_OLD_START), text.index(_LENS1_OLD_END)
    text = text[:start] + _LENS1_NEW + text[end:]
    text = _edit(text, _LENS3_TAIL_OLD, _LENS3_TAIL_NEW, "lens3_tail")
    text = _edit(text, "If the three lenses point to different stories,", "If the lenses point to different stories,", "lens_count")
    cut = text.index(_LENSES_END)
    text = text[:cut] + _KIND2_INCIDENTS + text[cut:]
    text = _edit(text, _VIOLENCE_OLD, "", "violence_paragraph")
    text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    text = _edit(text, _RUNNING_STORY_OLD, _RUNNING_STORY_NEW, "running_story")
    return text


# Round 6: ranking INSIDE the class. Drawn from mornings where two changes met:
# substantive AI beat a Supreme Court ruling and an emergency Fed cut; "what AI is doing to
# people / is it under control" (jobs data, safety testimony, loss of control) always beat
# "what AI can newly do" (verified benchmarks, proofs) and government AI rules; medical
# firsts traded places with AI and with a removal-power ruling; an act arriving in
# millions of lives beat a structural ruling both times they met (Medi-Cal over a Voting
# Rights Act ruling, premiums over control of the statistics agency); agency rulemaking
# and commercial court rulings never led; statistics other than AI's led once in 60.
_KIND_ORDER = (
    "WHEN SEVERAL STORIES PASS A LENS, the editor's order of preference is:\n"
    "1. What frontier AI is doing to people and whether it is under control: real-world harm, a loss of control, a safety failure "
    "attested under oath or by outside evaluators, official data on jobs, autonomous systems given real authority. These outrank "
    "news of what AI can newly do (verified benchmarks, proofs, solved problems), which outranks government rules about AI.\n"
    "2. Verified firsts in medicine, science, and energy, and extraordinary acts by the constitutional actors: the president "
    "invoking emergency powers, Congress or one of its chambers changing the rules, the Supreme Court ruling, the Federal Reserve "
    "acting in an emergency, and changes in the international order with great-power or nuclear stakes. These are peers of each "
    "other and close behind the first group; choose among them by how many people are reached and how long it will matter.\n"
    "3. A binding act arriving in millions of lives today (coverage ends, benefits stop, premiums jump) outranks a structural "
    "ruling or order whose effects arrive later.\n"
    "4. Routine agency rulemaking, and court rulings in commercial disputes, are changes but minor ones. They lead only when "
    "nothing above is among the candidates.\n"
    "5. Official statistics, other than those measuring what AI is doing, are second-story material. They lead only when nothing "
    "above is among the candidates.\n\n"
)


# Round 7: Opus followed round 6 faithfully and exposed two wrong rules. It rejected a lab's
# admission that its model broke into servers ("a company's own announcement does not pass")
# and called a contract for weapons that pick their own targets "anticipation" because
# fielding is years away. Three smaller clarifications come from the labels: Alex dismissed a
# narrow hospital tool as "a garbage model"; he led official jobs data in 4 of 6 lead picks
# on mornings that also had a safety story; and he passed over the end of a blockade.
_KIND_ORDER_V4 = _KIND_ORDER.replace(
    "1. What frontier AI is doing to people and whether it is under control: real-world harm, a loss of control, a safety failure "
    "attested under oath or by outside evaluators, official data on jobs, autonomous systems given real authority. These outrank ",
    "1. What frontier AI is doing to people and whether it is under control. First, measured effects on work and livelihoods "
    "(official data on jobs). Then control and safety: real-world harm from a frontier or widely deployed general system, a loss of "
    "control, a safety failure attested under oath or by outside evaluators, a government giving autonomous systems real authority. "
    "A narrow commercial tool failing is an incident, not this. These outrank ",
).replace(
    "These are peers of each other and close behind the first group; choose among them by how many people are reached and how long "
    "it will matter.\n",
    "These are peers of each other and close behind the first group; choose among them by how many people are reached and how long "
    "it will matter. A crisis beginning outranks a crisis winding down.\n",
)
_KIND_CLARIFY = (
    "TWO CLARIFICATIONS ON EVIDENCE AND TIMING:\n"
    "- A developer admitting that it lost control of its own system, or pausing a release because of what the system did, passes "
    "LENS 3 on the developer's word alone: it is an admission against its own interest. A developer claiming a new capability "
    "still needs outside confirmation.\n"
    "- A government decision to hand authority to autonomous systems (weapons that choose their own targets, software that "
    "decides benefits or sentences) is a completed act on the day the decision is made, even when deployment comes years later. "
    "It is not anticipation.\n\n"
)
assert _KIND_ORDER_V4 != _KIND_ORDER and "A crisis beginning" in _KIND_ORDER_V4 and "A narrow commercial tool" in _KIND_ORDER_V4


def kind4_important():
    text = kind2_important()
    cut = text.index(_LENSES_END)
    return text[:cut] + _KIND_ORDER_V4 + _KIND_CLARIFY + text[cut:]


# Round 8: under v_kind4 the lead reached parity and the second slot became the gap. The
# second-slot prompt steered Opus to pocketbook and California items where Alex's second
# pick is usually the next-best change. So the second slot is judged by the same rules as
# the lead. Incidents may take it when no second change exists (Alex gave them the second
# slot 14 times in 81), and California breaks ties only.
_SECOND_HEAD_OLD = "select the single most important story for the United States."
_SECOND_HEAD_NEW = (
    "select the show's SECOND main story. The lead story is already chosen: '{excluded_headline}'. Do not pick it, or any other "
    "brief about the same event. Judge the stories that remain exactly as you would judge a lead, by the rules below. If no "
    "remaining story is a change, an incident may take this slot; pick the one that matters most. When two remaining stories are "
    "close, prefer the one an average Californian will feel in daily life."
)


def kind4_second():
    text = kind4_important()
    assert "{" not in text and "}" not in text  # the template is filled with str.format
    return _edit(text, _SECOND_HEAD_OLD, _SECOND_HEAD_NEW, "second_head")


# Round 9: the kind-ruled second slot missed Alex's big incidents (a Speaker removed, a
# payments outage, a Guard deployment), and the pocketbook second slot missed his
# next-best changes. His second story reads as "the other thing a well-informed person
# needs today", with incidents competing freely, which is what the production LEAD prompt
# already rewards. So: kind rules pick the lead, production's lenses pick the second.
_SECOND_HEAD_PROD = (
    "select the show's SECOND main story. The lead story is already chosen: '{excluded_headline}'. Do not pick it, or any other "
    "brief about the same event. Pick the most important story that remains. When two remaining stories are close, prefer the one "
    "an average Californian will feel in daily life."
)


def production_as_second():
    text = TIER3_IMPORTANT_STORY_PROMPT
    assert "{" not in text and "}" not in text
    text = _edit(text, _SECOND_HEAD_OLD, _SECOND_HEAD_PROD, "second_head_prod")
    text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    return _edit(text, _RUNNING_STORY_OLD, _RUNNING_STORY_NEW, "running_story")


# --- Round 10 (2026-09-20): the second slot as a gradient ---------------------
#
# Reading of all 60 second picks (both labeling passes, 30 mornings). Counts:
#   * About half of Alex's second picks are stories the lead rules exclude:
#     roughly 27 of 60 are incidents, disruptions, market moves, running
#     conflicts, or anticipation, against 1 in 60 lead picks. So the second slot
#     re-admits them; kind4_second's gate ("an incident may take this slot only
#     if no change remains") lost five mornings by preferring an abstract change
#     to a concrete incident.
#   * The everyman question (candidate2_everyman) fails the other way: it drops
#     to health and consumer items when Alex wanted the day's big national story.
#     The two prompts are nearly complementary — together their exact picks cover
#     17 of 30 mornings, each alone 11 to 13.
#   * After a power/rules/international lead, 7 of 10 of his seconds are stories
#     people feel; after a lives or science lead the split is even. So the pair
#     tends to balance, weakly.
# Reading of "a gradient between most important and everyman": the second slot
# is still ranked by importance, but importance measured as how much of the story
# lands on people and how soon, not by which institutions must respond. The
# California lean only breaks ties.
_S1_HEAD = (
    "select the show's SECOND main story. The lead story is already chosen: '{excluded_headline}'. Do not pick it, or any other "
    "brief about the same event. The rules below say how this show ranks stories. Read them, then read THE SECOND SLOT at the end, "
    "which changes how they apply here."
)
_S1_BLOCK = (
    "THE SECOND SLOT. Everything above describes the lead. Two things change for the second story.\n\n"
    "First, the exclusion list does not apply in this slot. The lead has already given the audience the day's change. The second "
    "segment is the other thing a person who follows the news would be surprised to hear this show skip, so an incident, a "
    "disruption, a collapse, a running conflict, or a break in a market competes here on equal terms with a change. Incidents take "
    "this slot about as often as changes do. The tag rules below hold with one change: an '[UPDATE]' story may never be selected, "
    "but a '[DEVELOPMENT]' story that is more of the same may take this slot when the day's own event is large, though it does not "
    "outrank a fresh story of similar weight.\n\n"
    "Second, rank what remains by how much of it lands on people, and how soon, rather than by which institutions must respond.\n"
    "- Full weight to a story whose effect is being felt now or arrives within days: power or water cut off, payments or systems "
    "failing at scale, an evacuation or a warning people will act on today, a contaminated product pulled from shelves, transport "
    "stopped, a benefit that ends, coverage or a price reset by a public act.\n"
    "- Full weight to the largest disturbances at the top of government or the economy, even though they are incidents: the "
    "functioning or leadership of a chamber of Congress broken, an officer of the government impeached or forced out, troops sent "
    "into an American city, a fall in a market of a size not seen in years. A completed vote by a chamber counts for more than a "
    "resignation.\n"
    "- Full weight to the next change in the order set out above, with artificial intelligence keeping its rank there: when a "
    "substantive AI story did not lead, it belongs in this slot.\n"
    "- Less weight to a story whose effect is a projection, a deadline still ahead, or a milestone that crosses a line on paper "
    "without changing anything anyone will notice this month, however historic it sounds.\n\n"
    "Tie-breaks, in this order, for stories that are close after that. Prefer the one that touches health, safety, or money people "
    "must spend over the one that touches convenience or travel. Prefer the one that follows from a binding public act over one "
    "company's decision about itself. Prefer the one an average Californian will feel in daily life. These break ties only; they "
    "must not lift a small local, consumer, or single-company item over anything above.\n\n"
)


# Round 11: v_kind4_s1 scored 13/30 with Alex's own lead supplied. Its "full
# weight to what is reaching people now" bullet over-fired on disasters and
# warnings: it took a hurricane over a payments outage, sailors killed over a
# Supreme Court ruling, listeria deaths over an order on the statistics agency,
# and two storm forecasts. The labels are clear on this. Of the twelve
# disaster or casualty briefs in the 30 mornings, Alex gave a slot to exactly
# two, both in California (a wildfire in Riverside and San Bernardino, rain over
# Southern California burn scars); he passed over ten elsewhere, including an
# earthquake that killed 900 and a hurricane that moved 2.5 million people.
# s1 also lost a commercial antitrust ruling to the order of preference being
# read as suspended, and lost a customer data breach to a jump in gas prices.
# s2: state the disaster rule, demote price moves and a company's own fortunes,
# and say that only the incidents paragraph is suspended, not the ranking.
_S2_BLOCK = (
    "THE SECOND SLOT. Everything above describes the lead. Two things change for the second story and nothing else does; in "
    "particular the order of preference set out above still decides between stories of similar weight.\n\n"
    "First, the paragraph that keeps incidents and anticipation out of the lead does not apply in this slot. The lead has already "
    "given the audience the day's change. The second segment is the other thing a person who follows the news would be surprised to "
    "hear this show skip, so a disruption, a collapse, a running conflict, or a break in a market competes here on equal terms with "
    "a change. Incidents take this slot about as often as changes do. The tag rules below hold with one change: an '[UPDATE]' story "
    "may never be selected, but a '[DEVELOPMENT]' story that is more of the same may take this slot when the day's own event is "
    "large, though it does not outrank a fresh story of similar weight.\n\n"
    "Second, rank what remains by how much of it lands on people, and how soon, rather than by which institutions must respond.\n"
    "- Full weight to a story whose effect is reaching people now or within days: power or water cut off, payments or systems "
    "failing at scale, a product that is harming people pulled from shelves, transport stopped, a benefit that ends, coverage or a "
    "price reset by a public act, personal records taken from millions of customers.\n"
    "- Full weight to the largest disturbances at the top of government or the economy, even though they are incidents: the "
    "functioning or leadership of a chamber of Congress broken, an officer of the government impeached or forced out, troops sent "
    "into an American city, a fall in a market of a size not seen in years. A completed vote by a chamber counts for more than a "
    "resignation.\n"
    "- Full weight to the next change in the order of preference above, with artificial intelligence keeping its rank there: when a "
    "substantive AI story did not lead, it belongs in this slot.\n"
    "- A disaster, an accident, a storm, or a death toll takes this slot only when it falls on California, where the audience "
    "lives. Elsewhere it stays in the roundup however many were killed, evacuated, or warned, and a disaster that has not arrived "
    "yet is a forecast wherever it is pointed.\n"
    "- Little weight to a story whose effect is a projection, a deadline still ahead, or a line crossed on paper that changes "
    "nothing anyone will notice this month, however historic it sounds. A move in prices, in an index, or in one company's own "
    "fortunes — its results, its shares, a deal, what it will go on selling — is an indicator, not an event. Harm done to a "
    "company's customers is judged on the harm, not on the company.\n\n"
    "Tie-breaks, in this order, for stories still close after that. Prefer the one that happened to identifiable people — their "
    "bodies, their homes, their money, their records — over the one that moved a number. Prefer the one that follows from a public "
    "act over one company's choice about itself. Prefer the one an average Californian will feel in daily life. These break ties "
    "only; they must not lift a small local, consumer, or single-company item over anything above.\n\n"
)


def _second_slot(head, block):
    text = kind4_important()
    assert "{" not in text and "}" not in text  # the template is filled with str.format
    text = _edit(text, _SECOND_HEAD_OLD, head, "second_head_gradient")
    cut = text.index(_LENSES_END)
    text = text[:cut] + block + text[cut:]
    assert text.count("{excluded_headline}") == 1 and text.count("{") == 1 and text.count("}") == 1
    return text


def kind4_second_gradient():
    """Round 10: the lead's kind rules, with incidents re-admitted for slot two."""
    return _second_slot(_S1_HEAD, _S1_BLOCK)


def kind4_second_gradient2():
    """Round 11: the same, with disasters, forecasts, and price moves demoted."""
    return _second_slot(_S1_HEAD, _S2_BLOCK)


# Round 12: s2 scored 18/30 with Alex's lead supplied, against 13 for s1 and 12,
# 10, 10 for the three earlier prompts on the mornings they can be compared on.
# Two of its twelve misses come from clauses s2 itself added. The California
# carve-out let a revised death count from a heat wave that ended in August beat
# a chamber of Congress impeaching a cabinet officer, so it needs to cover only a
# disaster the audience is living through. And a food-poisoning recall beat an
# order putting the federal statistics agency under White House supervision, so a
# product harm has to sit below a binding national act. Both repairs are narrow.
_S3_BLOCK = _S2_BLOCK.replace(
    "- A disaster, an accident, a storm, or a death toll takes this slot only when it falls on California, where the audience "
    "lives. Elsewhere it stays in the roundup however many were killed, evacuated, or warned, and a disaster that has not arrived "
    "yet is a forecast wherever it is pointed.\n",
    "- A disaster, an accident, a storm, or a death toll takes this slot only when it falls on California, where the audience "
    "lives, and only while they are living through it. Elsewhere it stays in the roundup however many were killed, evacuated, or "
    "warned. A disaster that has not arrived yet is a forecast wherever it is pointed, and a death toll revised upward for an "
    "event that has already ended is a statistic, not a disaster.\n",
).replace(
    "a product that is harming people pulled from shelves, transport stopped,",
    "a product that is harming people pulled from shelves (which still ranks below a binding act of government of national "
    "reach), transport stopped,",
)
assert _S3_BLOCK != _S2_BLOCK and _S3_BLOCK.count("already ended") == 1 and _S3_BLOCK.count("national reach") == 1


def kind4_second_gradient3():
    """Round 12: s2 with the California carve-out and the product-harm clause narrowed."""
    return _second_slot(_S1_HEAD, _S3_BLOCK)


def kind3_important(lite=False):
    text = kind_important(False) if lite else kind2_important()
    cut = text.index(_LENSES_END)
    return text[:cut] + _KIND_ORDER + text[cut:]


# Round 10 (2026-09-22): BROAD PRINCIPLES. The topic rules of rounds 4-9 (v_kind4_s2)
# matched Alex on the 30 dev mornings and then lost to production on all three models
# on the 20 holdout mornings (Opus 5.5 7/20 vs 10/20 exact). Alex: "we need a broader
# generalizable rule". Constraint on every variant below: one short principle that
# names no topic and no story type, replacing production's lenses outright, so the
# model judges each story instead of matching it against a list. Tag and provenance
# rules are kept, rewritten only where they referred to the lenses.
_PRINCIPLES = {
    "consequence": (
        "Pick the story with the greatest consequence: how much it changes how Americans live or how the country is governed, "
        "weighing how many people it reaches, how deeply it changes things for them, and how long the change will last. "
        "How much coverage a story gets, how dramatic it is, and how urgent it feels today do not count on their own."
    ),
    "year_end": (
        "Pick the story most likely to appear in a careful year-end review of what actually mattered this year. Such a review "
        "keeps what changed the country's direction or people's lives in a lasting way, and drops what was merely loud, "
        "dramatic, or heavily covered for a few days."
    ),
    "direction": (
        "Pick the story that most changes what a well-informed person should now expect about where the country and the world "
        "are heading. Prefer news that moves that picture a long way for many people over news that fits the picture they "
        "already had, however large or loud it is."
    ),
    # Written AFTER seeing the holdout results (post hoc): the four principles above pulled toward
    # material welfare; Alex's holdout leads were mostly about how power is held and checked.
    "power": (
        "Pick the story that most changes who holds power, how that power may be used, or what limits it, in the United States or "
        "the world; or that most changes what people are newly able to do. Changes in how people are faring rank highest when a "
        "decision by those who hold power produced them. How much coverage a story gets, how dramatic it is, and how urgent it "
        "feels today do not count on their own."
    ),
    "one_thing": (
        "If a thoughtful listener could learn only one thing from today's news, which story would leave them best informed "
        "about the world they live in and the decisions ahead of them? Pick that story. Judge each story on its substance, "
        "not on how much coverage, drama, or urgency surrounds it."
    ),
}
_PRINCIPLE_DEV_NEW = (
    "- '[DEVELOPMENT]' stories CAN be selected. The audience heard a full segment on this story at least two days ago; the Coverage "
    "notes say what they already know. Judge only what is new against that knowledge, by the principle above. If the new part adds "
    "little to what the audience already knows, choose another story."
)


def principle_important(key):
    text = TIER3_IMPORTANT_STORY_PROMPT
    start, end = text.index(_LENS1_OLD_START), text.index(_LENSES_END)
    text = text[:start] + "HOW TO CHOOSE: " + _PRINCIPLES[key] + "\n\n" + text[end:]
    text = _edit(text, _VIOLENCE_OLD, "", "violence_paragraph")
    text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    text = _edit(text, _RUNNING_STORY_OLD, _PRINCIPLE_DEV_NEW, "running_story_principle")
    for banned in ("LENS", "lenses", "medic", "AI ", "court", "disaster", "recall"):
        assert banned not in text.split("IMPORTANT: Some headlines")[0] or banned in ("court",) and "court" not in _PRINCIPLES[key], banned
    return text


# Mornings from the first, easy invented set used as worked examples. They are
# excluded from scoring under the few-shot variant (FEWSHOT_EXCLUDE).
FEWSHOT_CASES = (
    "hypo_01_ai_capability_vs_war_increment",
    "hypo_03_process_vs_outcome",
    "hypo_05_arc_fatigue_vs_fresh",
    "hypo_06_spectacle_vs_policy",
    "hypo_07_slow_science_vs_political_noise",
    "hypo_09_which_ai_story",
)


def _fewshot_block():
    """Build the examples from the current labels, so a relabel flows through."""
    from benchmarks.editorial_decisions.hypotheticals import load_hypotheticals
    from benchmarks.editorial_decisions.label_server import load_labels

    labels = load_labels()
    cases = {c["case_id"]: c for c in load_hypotheticals()}
    parts = ["\n\nEXAMPLES: Past mornings and the story the show's editor led with. Learn his judgment from them; "
             "do not copy their topics.\n"]
    for n, case_id in enumerate(FEWSHOT_CASES, 1):
        case, label = cases[case_id], labels[case_id]
        lines = ["Example {}. Candidates:".format(n)]
        lines += ["  - " + b["raw_headline"] for b in case["briefs"]]
        lead = next(b for b in case["briefs"] if b["index"] == label["lead_index"])
        lines.append("  The editor led with: " + lead["headline"])
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def _lead(*edits):
    text = TIER3_IMPORTANT_STORY_PROMPT
    if "lives" in edits:
        text = _edit(text, _LASTING_OLD, _LIVES_NEW, "lives_lens")
        text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    if "running" in edits:
        text = _edit(text, _RUNNING_STORY_OLD, _RUNNING_STORY_NEW, "running_story")
    if "ai" in edits:
        text += _AI_NOTE
    if "taste" in edits:
        text += _TASTE_NOTE
    if "fewshot" in edits:
        text += _fewshot_block()
    return text


# name -> (builder of the lead prompt, builder of the second-slot template)
REGISTRY = {
    "production": (lambda: TIER3_IMPORTANT_STORY_PROMPT, lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "candidate": (candidate_important, candidate_everyman),
    "candidate2": (candidate_important, candidate2_everyman),
    "v_running": (lambda: _lead("running"), candidate2_everyman),
    "v_lives": (lambda: _lead("lives"), candidate2_everyman),
    "v_lives_running": (lambda: _lead("lives", "running"), candidate2_everyman),
    "v_taste": (lambda: _lead("taste"), candidate2_everyman),
    "v_kind": (lambda: kind_important(True), candidate2_everyman),
    "v_kind_lite": (lambda: kind_important(False), candidate2_everyman),
    "v_kind2": (kind2_important, candidate2_everyman),
    "v_kind3": (lambda: kind3_important(False), candidate2_everyman),
    "v_kind4": (kind4_important, candidate2_everyman),
    "v_kind4b": (kind4_important, kind4_second),
    "p_consequence": (lambda k='consequence': principle_important(k), lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "p_year_end": (lambda k='year_end': principle_important(k), lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "p_direction": (lambda k='direction': principle_important(k), lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "p_one_thing": (lambda k='one_thing': principle_important(k), lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "p_power": (lambda k='power': principle_important(k), lambda: TIER3_EVERYMAN_STORY_PROMPT),

    "v_kind4c": (kind4_important, production_as_second),  # kind rules for the lead, production lenses for the second  # same lead as v_kind4; second slot judged by the lead's rules
    "v_kind4_s1": (kind4_important, kind4_second_gradient),  # same lead as v_kind4; second slot as a gradient
    "v_kind4_s2": (kind4_important, kind4_second_gradient2),  # s1 with disasters, forecasts and price moves demoted
    "v_kind4_s3": (kind4_important, kind4_second_gradient3),  # s2 with the disaster and product-harm clauses narrowed
    "v_kind_lite3": (lambda: kind3_important(True), candidate2_everyman),
    "v_ai": (lambda: _lead("ai"), candidate2_everyman),
    "v_ai_running": (lambda: _lead("ai", "running"), candidate2_everyman),
    "v_fewshot": (lambda: _lead("fewshot"), candidate2_everyman),
}
VARIANTS = tuple(REGISTRY)
FEWSHOT_EXCLUDE = {"v_fewshot": set(FEWSHOT_CASES)}


def prompts_for(variant):
    """Return (important_prompt, everyman_template) for a variant name."""
    lead, second = REGISTRY[variant]
    return lead(), second()


# --- Round 11 (2026-09-22): GENERAL lenses (g_*) ------------------------------
#
# Alex: the rule must be broad; topic lists ("medical firsts rank with acts of
# state") are "way too specific". So every g_ variant names no domain and no
# kind of story. Diagnosis from the 30 dev mornings only (Gemma x3, production
# 23/30 in set; the five one-paragraph principles 20-22/30):
#   * Production's lenses win where the principles lose on 5 mornings: an
#     extraordinary use of authority (troops, a top-court ruling), a first or a
#     loss of control, and a betrayal of trust. The principles drifted to
#     measured pocketbook items and a long-dated structural ruling.
#   * Production loses 7 mornings. In 3 of them (plus one where it picked
#     acceptable but not exact) its essays credit LENS 1 to turmoil that will
#     settle within weeks (an officer removed, a lockout, a deadly collision)
#     over a completed change that will last; one essay called a lasting first
#     "historic but not urgent". One more is a forecast chosen over done acts.
# So the g_ variants keep the three lenses, strip their named examples, and add
# abstract weighing: has it happened, will it last, how far does it reach.
_G_LENS1 = (
    "LENS 1 — FORCED REACTION: Who is forced to react to this, and at what scale? The most important stories force reactions "
    "from the institutions with the widest authority, national or international. Less important stories only require local "
    "response.\n\n"
    "Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. "
    "High uncertainty about a critical system can be as important as a completed action because institutions are forced to "
    "prepare for multiple outcomes.\n\n"
)
_G_LENS1_AUTHORITY = (
    "LENS 1 — AUTHORITY AND REACTION: Did someone with wide authority make a binding decision, or did an event force such a "
    "decision, and at what scale? The most important stories involve the institutions with the widest authority, national or "
    "international. Less important stories only require local response.\n\n"
    "Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. "
    "Distinguish two kinds of reaction: an event that makes institutions scramble until it passes, and one that changes what "
    "they can or must do from now on. The second counts for more. High uncertainty about a critical system can be as important "
    "as a completed action because institutions are forced to prepare for multiple outcomes.\n\n"
)
_G_LENS2 = (
    "LENS 2 — ACCOUNTABILITY OF POWER: Does the story reveal that powerful people or institutions acted corruptly, illegally, or "
    "in betrayal of public trust? Check: (a) the scale of power involved, (b) the quality of evidence, and (c) whether the story "
    "reveals systemic failure, not just individual misconduct.\n\n"
)
_G_LENS3 = (
    "LENS 3 — FRONTIER CAPABILITY AND CONTROL: Does the story show something being done that could not be done before, or those "
    "who built or operate a powerful system losing control of it? This lens exists because such events change what every "
    "institution must plan for, even before any of them has reacted. Check: (a) the capability or loss of control is stated "
    "concretely, not as a mood or a \"growing concern\"; (b) it is attested by someone other than the party that gains from the "
    "claim — independent verification, peer review, an outside evaluator, an official finding, or the affected party — a party's "
    "own announcement or its score on a test of its choosing does not pass; (c) it is a first or a boundary crossed, not an "
    "increment.\n\n"
)
_G_TIE_PROD = "If the three lenses point to different stories, prefer the one with broader implications for more Americans.\n\n"
_G_WEIGH = (
    "WEIGHING: When several stories pass a lens, or the lenses point to different stories, weigh them on three questions.\n"
    "- Has it happened? A completed decision or a confirmed result outranks a forecast, a warning, a threat, a proposal, or a "
    "deadline still ahead.\n"
    "- Will it last? Ask whether things will still be different a year from now because of it. A change that persists, in the "
    "rules, in what is possible, or in people's circumstances, outranks a disturbance that will be settled, reversed, or over "
    "within weeks, however urgent, dramatic, or heavily covered it is today. Urgency is not importance.\n"
    "- How far does it reach? Prefer the story with broader implications for more Americans.\n\n"
)
_G_HARM_NEW = (
    "For a story whose weight rests on the harm or loss it has caused: prioritize it only when the scale forces national or "
    "international response, or when it reveals the failure of a system people rely on.\n\n"
)
_G_SPECIFIC_OLD = 'The story must be specific, not vague like "Israel Hamas War".'
_G_SPECIFIC_NEW = "The story must be a specific event, not a vague topic."
# Words that name a domain or a kind of story. None may appear in the selection
# guidance (everything before the tag rules) of a g_ variant.
_G_BANNED = (r"medic", r"health", r"\bAI\b", r"technolog", r"scien", r"court", r"supreme", r"election", r"congress",
             r"militar", r"disaster", r"recall", r"market", r"compan", r"california", r"\bwars?\b", r"violen", r"tragedi",
             r"president", r"cyber", r"bank", r"government", r"developer", r"israel", r"econom", r"climate", r"energy",
             r"\bjobs?\b", r"weapon", r"troops", r"price")


def _g_build(lens1, weigh):
    text = TIER3_IMPORTANT_STORY_PROMPT
    start, end = text.index(_LENS1_OLD_START), text.index(_LENSES_END)
    block = "Use these lenses to judge importance:\n\n" + lens1 + _G_LENS2 + _G_LENS3 + (weigh or _G_TIE_PROD)
    text = text[:start] + block + text[end:]
    text = _edit(text, _VIOLENCE_OLD, _G_HARM_NEW, "harm_paragraph")
    if weigh:
        text = _edit(text, _DUPLICATE_OLD, _DUPLICATE_NEW, "duplicate_tiebreak")
    text = _edit(text, _G_SPECIFIC_OLD, _G_SPECIFIC_NEW, "specific_event")
    guidance = text.split("IMPORTANT: Some headlines")[0]
    for banned in _G_BANNED:
        assert not re.search(banned, guidance, re.I), banned
    return text


def g1_abstract():
    """Production's lenses with every named domain and example removed. Control for the cost of abstraction."""
    return _g_build(_G_LENS1, None)


def g2_lasting():
    """g1 plus abstract weighing: has it happened, will it last, how far does it reach."""
    return _g_build(_G_LENS1, _G_WEIGH)


def g3_authority():
    """g2 with LENS 1 split into reactions that pass and decisions that change what institutions can do."""
    return _g_build(_G_LENS1_AUTHORITY, _G_WEIGH)


REGISTRY.update({
    "g1_abstract": (g1_abstract, lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "g2_lasting": (g2_lasting, lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "g3_authority": (g3_authority, lambda: TIER3_EVERYMAN_STORY_PROMPT),
})
VARIANTS = tuple(REGISTRY)


# Round 11b: g2's weighing was read against two lenses. "Has it happened?" sank a
# completed decision whose effects come later (a contract fielded in two years
# lost to a finished rule), and "will it last?" was read as "weeks or months"
# for turmoil over who holds a post. g4 states that a decision counts from the day
# it is made, asks what will differ a year on beyond the fact that it happened,
# and ranks the rarity of the power used. g5 is the same idea in two sentences.
_G_LENS1_RARE = (
    "LENS 1 — FORCED REACTION: Who is forced to react to this, and at what scale? The most important stories force reactions "
    "from the institutions with the widest authority, national or international. Less important stories only require local "
    "response. The rarer the power used or the step taken — something done once in decades, or never before — the more "
    "important the story.\n\n"
    "Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. "
    "A reaction that lasts only until the trouble is settled counts for less than a decision that changes what those "
    "institutions can or must do from now on. High uncertainty about a critical system can be as important as a completed "
    "action because institutions are forced to prepare for multiple outcomes.\n\n"
)
_G_WEIGH2 = (
    "WEIGHING: When several stories pass a lens, or the lenses point to different stories, weigh them on three questions.\n"
    "- Has it happened? A completed decision or a confirmed result outranks a forecast, a warning, a threat, or a proposal. "
    "A decision counts from the day it is made, even when its effects arrive later.\n"
    "- Will it last? Ask what will be different a year from now because of this story, beyond the fact that it happened. A "
    "change that persists, in the rules, in what is possible, in what those in power may do, or in people's circumstances, "
    "outranks a disturbance that will be settled, reversed, or over within weeks, however urgent, dramatic, or heavily covered "
    "it is today. A change in who holds a position matters less than a change in what that position can do.\n"
    "- How far does it reach? Prefer the story with broader implications for more Americans.\n\n"
)
_G_WEIGH_SHORT = (
    "If several stories pass, or the lenses point to different stories, prefer the one after which the most will be different a "
    "year from now, for the most Americans: a lasting change in the rules, in what is possible, or in people's circumstances "
    "outranks a disturbance that will be settled or over within weeks, however urgent or loud it is today. A decision counts "
    "from the day it is made, even when its effects arrive later.\n\n"
)


def g4_rarity():
    """g2 with rarity of the power used, reactions that pass vs decisions that last, and a sharper weighing."""
    return _g_build(_G_LENS1_RARE, _G_WEIGH2)


def g5_short():
    """g1 with production's tie-break replaced by one two-sentence weighing rule."""
    return _g_build(_G_LENS1, _G_WEIGH_SHORT)


REGISTRY.update({
    "g4_rarity": (g4_rarity, lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "g5_short": (g5_short, lambda: TIER3_EVERYMAN_STORY_PROMPT),
})
VARIANTS = tuple(REGISTRY)


# Round 11c: under g4 the remaining misses are argued through LENS 1 with
# reactions the model imagines ("this forces a reaction from Congress", "forces
# agencies to prepare"). g6 asks for reactions that are reported, not predicted.
_G_LENS1_OBSERVED = _G_LENS1_RARE.replace(
    "Completed actions beat threats. ",
    "Count only reactions the briefs report or that the event requires by its nature; a reaction the story might provoke "
    "later is a forecast. Completed actions beat threats. ",
)
assert _G_LENS1_OBSERVED != _G_LENS1_RARE


def g6_observed():
    """g4 with LENS 1 limited to reported or required reactions, not predicted ones."""
    return _g_build(_G_LENS1_OBSERVED, _G_WEIGH2)


REGISTRY.update({"g6_observed": (g6_observed, lambda: TIER3_EVERYMAN_STORY_PROMPT)})
VARIANTS = tuple(REGISTRY)


# Round 11d: on GPT-6 Luna, g4's three-question weighing overrode the lenses
# (a rule change of modest reach beat an extraordinary use of authority twice).
# g7 keeps g4's LENS 1 and makes the weighing a tie-break for stories that pass
# at a similar scale, as production's own tie-break is.
_G_WEIGH_TIE = (
    "If the lenses point to different stories, or several stories pass at a similar scale, prefer the one after which more "
    "will still be different a year from now, for more Americans. A decision counts from the day it is made, even when its "
    "effects arrive later. A disturbance that will be settled, reversed, or over within weeks ranks below a lasting change, "
    "however urgent it feels today. A change in who holds a position matters less than a change in what that position can "
    "do.\n\n"
)


def g7_tiebreak():
    """g4's LENS 1 with the weighing reduced to a tie-break."""
    return _g_build(_G_LENS1_RARE, _G_WEIGH_TIE)


REGISTRY.update({"g7_tiebreak": (g7_tiebreak, lambda: TIER3_EVERYMAN_STORY_PROMPT)})
VARIANTS = tuple(REGISTRY)


# Round 11e: on Luna, g4 lost four mornings production wins, three of them to
# stripped examples. Two were emergency actions to keep a failing system working
# (production's LENS 1 had named the institutions that take them), and one was an
# independently confirmed "better than before" result that production's LENS 3
# example ("a new model with better scores does not") had excluded. g8 restores
# both ideas in abstract terms, without naming any institution or field.
_G_LENS1_SYSTEM = _G_LENS1_RARE.replace(
    "High uncertainty about a critical system can be as important as a completed action because institutions are forced to "
    "prepare for multiple outcomes.\n\n",
    "When a system that much else depends on is failing, emergency action to keep it working counts as a completed action at "
    "the highest scale. High uncertainty about such a system can be as important as a completed action because institutions "
    "are forced to prepare for multiple outcomes.\n\n",
)
_G_LENS3_INCREMENT = _G_LENS3.replace(
    "(c) it is a first or a boundary crossed, not an increment.\n\n",
    "(c) it is a first or a boundary crossed, not an increment: doing something already possible better, faster, or cheaper "
    "is an increment, however large the margin and whoever confirms it.\n\n",
)
assert _G_LENS1_SYSTEM != _G_LENS1_RARE and _G_LENS3_INCREMENT != _G_LENS3


def g8_restored():
    """g4 with production's critical-system and increment ideas restored in abstract words."""
    text = _g_build(_G_LENS1_SYSTEM, _G_WEIGH2)
    text = _edit(text, _G_LENS3, _G_LENS3_INCREMENT, "lens3_increment")
    guidance = text.split("IMPORTANT: Some headlines")[0]
    for banned in _G_BANNED:
        assert not re.search(banned, guidance, re.I), banned
    return text


REGISTRY.update({"g8_restored": (g8_restored, lambda: TIER3_EVERYMAN_STORY_PROMPT)})
VARIANTS = tuple(REGISTRY)


# Round 11f: g8's critical-system sentence won back the emergency rescues on Luna
# but let Gemma rank a stoppage ("critical infrastructure") over a lasting ruling.
# g9 keeps only the increment clause. g10 words the rescue as an action by those
# with the widest authority, so a failure by itself earns nothing extra.
_G_LENS1_RESCUE = _G_LENS1_RARE.replace(
    "High uncertainty about a critical system can be as important as a completed action",
    "Emergency action by those with the widest authority to keep a failing system working is a completed action at the "
    "highest scale. High uncertainty about a critical system can be as important as a completed action",
)
assert _G_LENS1_RESCUE != _G_LENS1_RARE


def _g_increment(lens1):
    text = _edit(_g_build(lens1, _G_WEIGH2), _G_LENS3, _G_LENS3_INCREMENT, "lens3_increment")
    guidance = text.split("IMPORTANT: Some headlines")[0]
    for banned in _G_BANNED:
        assert not re.search(banned, guidance, re.I), banned
    return text


def g9_increment():
    """g4 plus the abstract increment clause in LENS 3."""
    return _g_increment(_G_LENS1_RARE)


def g10_rescue():
    """g9 plus emergency action to keep a failing system working, credited to the actor, not the failure."""
    return _g_increment(_G_LENS1_RESCUE)


REGISTRY.update({
    "g9_increment": (g9_increment, lambda: TIER3_EVERYMAN_STORY_PROMPT),
    "g10_rescue": (g10_rescue, lambda: TIER3_EVERYMAN_STORY_PROMPT),
})
VARIANTS = tuple(REGISTRY)
