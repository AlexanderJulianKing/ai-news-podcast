FOLLOW_UP_PROMPT_TEMPLATE = (
    "Based on this story, what is a good google-able follow-up question? Place the actual question "
    "in quotation marks so that the question can be isolated. Place as much context into the "
    "question as possible, because you are asking an LLM search engine with zero prior context. "
    "Also note that today is {date}."
)


CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE = (
    "Based on this story, what is a good google-able follow-up question that challenges the "
    "premise of the story? For example, if a news story about the maui wildfires criticizes the "
    "maui government for not using the alarms, ask what the alarms are actually for? Because it "
    "turns out, the alarms were for tsunami warnings, which would have drawn people away from the "
    "oceans and towards the fires. Or as another example, if Iran had launched missiles at Israel, "
    "ask if this attack was unprovoked or if it was in retaliation to something. One thing to note is that people WILL just straight up lie sometimes. For example, the Utah Governor once said that the Charlie Kirk alleged shooter held 'Leftist ideologies' but gave zero evidence. And no information corroborates this claim. So that claim needs to be read with extreme skepticism. Elected officials can and will lie to serve their own ends. \nPlace the actual "
    "question in quotation marks so that the question can be isolated. Place as much context into "
    "the question as possible, because you are asking an LLM search engine with zero prior context. "
    "Also note that today is {date}."
)


RESEARCH_MEMORY_PROMPT = (
    "You are preparing a dated memory note for today's reporting. You will receive "
    "today's story seed and prior coverage retrieved from the show's research index.\n\n"
    "Use prior coverage only as background. It may be stale. Do not state that any old "
    "fact is true today unless today's reporting verifies it.\n\n"
    "Return a compact note with these headings:\n"
    "PRIOR BACKGROUND:\n"
    "POSSIBLY STALE OR NEEDS VERIFICATION:\n"
    "UNRESOLVED QUESTIONS FOR TODAY:\n"
    "SUGGESTED SEARCH TARGETS:\n"
)


RESEARCH_CONTROLLER_PROMPT = (
    "You are a wonky investigative reporter controlling a bounded research loop for a daily "
    "news podcast. Your job is to decide whether the story needs one more targeted research "
    "action or whether there is enough evidence to synthesize.\n\n"
    "Think in terms of central claim, source quality, missing timeline, scale, mechanism, "
    "counterevidence, freshness, and accountability. RAG memory can suggest questions, but "
    "today's sources must verify current facts.\n\n"
    "Return ONLY valid JSON. No markdown, no prose outside JSON.\n\n"
    "If more research is needed, return one of:\n"
    "{\"status\":\"continue\",\"action\":\"grounded_search\",\"question\":\"...\","
    "\"question_type\":\"premise_challenge|scale_check|source_check|timeline_check|"
    "mechanism_check|counterevidence_check|freshness_check\",\"reason\":\"...\"}\n"
    "{\"status\":\"continue\",\"action\":\"article_search\",\"query\":\"...\",\"reason\":\"...\"}\n\n"
    "If enough evidence exists, return:\n"
    "{\"status\":\"done\",\"reason\":\"...\",\"confidence\":\"low|medium|high\"}\n\n"
    "Prefer grounded_search for precise questions. Use article_search only when the current "
    "evidence is thin and another article source is likely to materially improve the summary."
)


RESEARCH_CONTROLLER_REPAIR_PROMPT = (
    "Repair the following research-controller response into ONLY valid JSON matching one "
    "of the allowed schemas. Preserve the intended action if possible. No markdown."
)


RESEARCH_ADVERSARY_PROMPT = (
    "You are a second-perspective adversarial editor reviewing a selected main story "
    "after the show's lead editor has decided the evidence is good enough to synthesize.\n\n"
    "Your job is to find the single strongest skeptical, evidence-seeking question "
    "that could change the story's framing, weaken its central claim, reveal missing "
    "scale or timeline, or expose unsupported assumptions. Do not ask for broad "
    "background. Do not polish the story. Pick one question that a controlled source "
    "search can answer.\n\n"
    "Return ONLY valid JSON. No markdown, no prose outside JSON.\n\n"
    "Schema:\n"
    "{\"question\":\"...\",\"question_type\":\"premise_challenge|scale_check|"
    "source_check|timeline_check|mechanism_check|counterevidence_check|freshness_check\","
    "\"reason\":\"...\"}\n\n"
    "If the current evidence is already too thin to challenge specifically, ask a "
    "source_check question that tests the central claim against primary or independent "
    "reporting."
)


OVERVIEW_SYSTEM_PROMPT_TEMPLATE = (
    "You are a newsroom researcher working on {date}. You receive headlines that editors have already "
    "confirmed are real and newsworthy. Gather the latest factual reporting "
    "and produce concise summaries that attribute information to reputable outlets. Never dismiss the "
    "headline as nonexistent; if you cannot verify it after thorough searching, reply with 'UNVERIFIED:' "
    "followed by the queries you tried and guidance on what the editor should check next. "
    "CRITICAL on personnel and titles: Cabinet officials, agency heads, and other office-holders change. "
    "Industry publications, trade journals, and legal/policy newsletters often describe past officials in "
    "present tense long after they have left office. Before naming anyone by title (e.g., 'Attorney General X', "
    "'Secretary of Y'), verify the person currently holds that office as of {date} using sources dated within "
    "the last two weeks. If the current office-holder is unclear or the named person has been removed, "
    "omit the personal name and attribute to the office (e.g., 'the Justice Department', 'the Attorney General') "
    "rather than repeat a stale name."
)


MAIN_STORY_PROMPT = """Given the following headlines, return the most important story for the United States.

Use two lenses to judge importance:

LENS 1 — FORCED REACTION: Who is forced to react to this, and at what scale? The most important stories force reactions from major institutions — governments, central banks, militaries, the Supreme Court. Less important stories only require local response.

Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. High uncertainty about critical systems (president hospitalized, major cyberattack) can be as important as completed actions because institutions are forced to prepare for multiple outcomes.

For example: if the US kidnaps Venezuela's president while also trading threats with China and foiling a terror plot, cover Venezuela. Seizing a head of state forces every government in the region to respond — it's essentially a declaration of war. The China threats are just words (no one is forced to act yet), and the foiled plot means nothing actually happened.

LENS 2 — ACCOUNTABILITY OF POWER: Does the story reveal that powerful people or institutions acted corruptly, illegally, or in betrayal of public trust? These stories matter not because of a single dramatic reaction, but because they force the public to re-evaluate who holds power and whether the system is working.

To separate important accountability stories from tabloid noise, check: (a) the scale of power involved — heads of state, billionaires, or senior officials vs. local figures, (b) the quality of evidence — official government releases, court documents, or verified records vs. anonymous allegations, (c) whether this is surprising or confirms an already-established pattern — if an administration has been repeatedly placing industry insiders in regulatory roles, one more instance of that pattern is confirmatory, not revelatory, unless the specific consequences are extraordinary (e.g., mass casualties, irreversible environmental damage), and (d) whether the story reveals systemic failure, not just individual misconduct — multiple powerful people implicated across institutions, or a pattern of protection by the system itself.

For example: if a government agency releases thousands of documents showing that multiple billionaires and senior political figures had ties to a convicted trafficker, that is a top story — even if no single institution has a dramatic response — because it reveals whether the justice system treats the powerful the same as everyone else. By contrast, a single celebrity scandal based on tabloid reporting is not.

For violence, disasters, and tragedies: prioritize only when the scale forces federal or international response, or when it signals system failure. A regional earthquake requiring FEMA deployment is important; a local crime requiring only police response is not the top story.

If both lenses point to different stories, prefer the one with broader implications for more Americans. The story must be specific, not vague like "Israel Hamas War". Repeat the headline verbatim in your Answer.
"""


EVERYMAN_STORY_PROMPT = (
    'Given the following headlines across news sources, what would you say is the most important news story for the average person in California. '
    'If there is nothing directly impacting Californians, pick something that would affect the everyday American. '
    'Do not pick something that is happening in a non-US country. Do not pick something vague like "Israel Hamas War"; '
    'the news stories must be about a specific event. Explain why you picked each story, then write down your answer in the format of "Answer:". '
    'Do not pick a mass shooting. When you list the story in "Answer", you must list the event as specifically as possible; '
    'repeating the headline verbatim is preferable.\n\n'
)


OVERVIEW_PICK_PROMPT = "Given the following headlines, pick the day's 5 most important headlines. Only mention the 5 headlines. These will be the \"minor\" stories of the day.\n"


HEADLINE_EXTRACTION_PROMPT = (
    'The text below explains the selection of a story. What is the headline or story chosen in the text below? '
    'Here are some examples of what headlines look like. \n\n '
    'House leaders toil to advance Ukraine and Israel aid. But threats to oust speaker grow. \n'
    'Google worker says the company is \'silencing our voices\' after mass firings\n'
    'Judge in Trump case orders media not to report where potential jurors work\n\n'
    'Here are the real headlines. Give only the headline as output: \n'
)


OVERVIEW_ANCHOR_PROMPT = (
    'Please give a text of the following information as if you were a morning news anchor reading the news aloud. '
    "No introductions, just preface it with 'here are some other headlines:' These headlines will not be covered later, "
    "this is all the information you get. \n\nIf you get information like \" Unfortunately, the search results do not provide "
    "specific details about the road crash in southern Turkey that resulted in at least 10 deaths. The search results are "
    "primarily focused on sports news, entertainment, and politics, with no direct mention of the incident in Turkey. "
    "To obtain more information about the road crash, it would be necessary to access news sources that are not included "
    'in the search results provided. It is recommended to check reputable news outlets, such as BBC News, Al Jazeera, or CNN, '
    'for updates on the incident. " do not include it. Some bits of information might be covered multiple times in this text. '
    'Cover each bit of information only once. Do not repeat yourself. Be sure to include every story. Make it 500 words. '
    "Numbers should be written in word form, like 'two hundred fifty five'. "
    'Also use wordplay and puns whenever you can. Do NOT put puns or wordplay in quotation marks — just use them naturally in the sentence.'
)


REPETITION_REMOVER_TEMPLATE = (
    "You are filtering today's headlines against stories we already covered in the past week.\n"
    "Here is a log of recently covered stories:\n"
    "{recent_stories}\n\n"
    "For each headline in the text below, classify it into one of four categories:\n"
    "1. SAME STORY, NO NEW INFO — the headline is essentially retelling something we already covered "
    "with no meaningful new development. REMOVE these headlines from the output entirely.\n"
    "2. UPDATE — the headline is about the same ongoing situation we covered before, but there IS a new "
    "development (a new vote, a new response, new casualties, a policy shift, a court ruling, etc.). "
    "KEEP the headline but prepend '[UPDATE]' to it.\n"
    "3. MAJOR ESCALATION — RARE. Use this only for a fundamental state change that makes the story "
    "materially different from what the audience previously heard: a war begins or ends, a ceasefire "
    "takes effect, a new country directly enters the conflict, a weapon of mass destruction is used, "
    "a government or major institution collapses, or a leader's death changes command or succession. "
    "KEEP the headline but prepend '[MAJOR ESCALATION]' to it.\n"
    "4. NEW STORY — the headline is unrelated to anything we covered. KEEP it as-is with no tag.\n\n"
    "CONTEXTUAL THRESHOLD: Compare today's event with the state of the story the audience last heard. "
    "The same action can be a MAJOR ESCALATION or an UPDATE depending on that baseline. Strikes that "
    "restart a war after a durable ceasefire or months without active fighting are a MAJOR ESCALATION. "
    "Another round of strikes during an active campaign with recent daily exchanges is an UPDATE, even "
    "if the new round is large or important. Likewise, crossing a genuinely new boundary—such as a new "
    "country entering directly, first use of a new class of weapon, or transition from threats to an "
    "actual invasion—can be MAJOR; more targets, casualties, retaliation, blockade enforcement, legal "
    "notices, or market effects within the current phase are normally UPDATEs. Recency and the audience's "
    "last known state matter more than dramatic wording. When uncertain, classify it as UPDATE.\n\n"
    "Use a broad definition of 'same story'. For example, "
    "\"House leaders toil to advance Ukraine and Israel aid. But threats to oust speaker grow.\", "
    "\"Facing a Republican revolt, House Speaker Johnson\u2019s plan for US aid to Ukraine, Israel and allies uncertain.\", "
    "and \"House Speaker Mike Johnson pushes toward a vote on aid for Israel, Ukraine, and Taiwan.\" "
    "are all the same story. But if we covered a 'looming government shutdown' and today's headline says "
    "the shutdown ACTUALLY happened, that is a MAJOR ESCALATION, not the same story.\n\n"
    "Return the modified text and nothing else."
)


TITLE_PROMPT = (
    'Shorten the following story to a headline like "What to watch as jurors consider Trump\'s fate", '
    '"Why voters have become so partisan about the economy.", or "U.S. sunscreen is years behind the world. Here\'s why." '
    "We want to phrase it in a kind of question (without the question mark) rather than revealing the whole story in the headline. "
    "The answer to the question should be the main focus of the story. Do note give any explanations; only give the headline. No commas."
)


INTRO_PROMPT_TEMPLATE = (
    "Pretend you are a scriptwriter for a daily news podcast called Alex's News. Write the text that the anchor will read "
    "for the preview of the news to come based on the stories given. Note that the anchor has already said \"{intro1}\", "
    "so do not include that greeting or any greeting of any kind. Today is {date}, but do not mention this unless it is a "
    "holiday. If it is a holiday, you should mention that it is one. Do not include dashes. Begin by talking about the fact "
    "that {weather}. Then, briefly mention the upcoming stories that will be covered in the episode, which are listed below. "
    "Try to paraphrase the headlines instead of quoting them verbatim. For example: \"What's China Bringing Back from the "
    "Moon's Far Side?\" should be turned into \"What China plans on bringing back from the moon.\" \nSomething like this "
    "would be a good text to make:\"The weather in Riverside will be ninety-nine degrees fahrenheit today, so be sure to "
    "stay hydrated and take breaks in the shade if you're spending time outdoors. It's a perfect day to reflect on the "
    "importance of trees and nature in our lives. Happy Arbor Day! Coming up, we'll dive into why voters' views on the "
    "economy have gotten so political, explore how Mexico's cartels are infiltrating the tortilla industry, and discuss "
    "the good news and bad news about travel this summer. All this and more coming up on Alex's News. Stay tuned!\". "
    "Also, do not assume that an event has not happened yet. For example, if the headline is \"First Presidential Debate "
    "Between Donald Trump and Kamala Harris\", do not assume that it has happened or not happened yet. Do not call it "
    "\"upcoming\" unless you know for sure that it has not happened yet. There are other minor stories covered besides "
    "these two, but you can just say that other stories will be covered. This should be about 120 words long."
)


SEGMENT_SCRIPT_PROMPT_TEMPLATE = (
    "You are a senior broadcast writer. Rewrite the news summary as a natural back-and-forth "
    "between morning anchor Grace and reporter {reporter_name}. Keep it conversational and easy "
    "to read aloud, with contractions and varied sentence lengths (mostly 8\u201322 words). Alternate "
    "speakers, with Grace asking short 1\u20132 sentence questions that build on the previous answer, "
    "and {reporter_name} replying in 3\u20135 sentences that add context listeners may not know. "
    "Aim for roughly 25\u201330 total speaker turns (around 12\u201315 exchanges). Do NOT create a rapid-fire "
    "Q&A with dozens of one-sentence answers; let {reporter_name} develop their points fully before "
    "Grace moves to the next topic. "
    "CRITICAL: Preserve all proper nouns, operation codenames, organization names, personal names, "
    "place names, dates, and statistics from the summary EXACTLY as written. Never substitute, "
    "paraphrase, or invent alternatives for specific factual details. "
    "Weave sources into the answers (e.g., 'according to NPR' or 'The Washington Post reports'), "
    "rather than listing them. Avoid lists, filler, or stage directions. No one is on location. "
    "Grace should briefly set up the story, introduce the reporter, and thank them at the end. "
    "Mention timing relative to today ({date}) when specific dates appear. This is story {story_num} "
    "of {total_stories}. Format as 'Name: text' with each line on its own line; no special symbols. "
    "When using a quote, say 'Person A said, quote, yadda yadda, endquote'. The final output should "
    "be about 1500 words. Numbers should be written in word form, like 'two hundred fifty five'."
)


TIER1_TRIAGE_PROMPT = """Rate each headline below on a scale of 1-10 for newsworthiness using these criteria:
- Impact scale: How many people are materially affected?
- Institutional response: Does this force action from governments, courts, or major organizations?
- Novelty: Is this a new development or a rehash of ongoing coverage? Note: headlines tagged '[UPDATE]' or '[MAJOR ESCALATION]' are continuations of previously covered stories that have genuine new developments. Do NOT penalize them for being continuations — score them based on how significant the new development itself is.
- Accountability: Does it reveal corruption, illegality, or betrayal of public trust by powerful actors?

For each headline, output exactly one line in this format:
SCORE: X | HEADLINE: Y | REASON: Z

Where X is the numeric score (1-10), Y is the headline repeated verbatim, and Z is a one-sentence justification. Sort from highest to lowest score."""


TIER1_CALIFORNIA_TRIAGE_PROMPT = """Rate each headline below on a scale of 1-10 for relevance to an average person in California.

Use these criteria:
- Direct California impact: Does this happen in California, directly affect Californians, or involve California institutions, laws, infrastructure, companies, schools, housing, climate, disasters, courts, immigration, technology, agriculture, energy, or state/local politics?
- Everyday-life impact: Would this materially affect daily life for Californians — prices, jobs, housing, transit, health care, schools, public safety, taxes, utilities, weather, water, wildfire risk, air quality, or rights?
- West Coast / Pacific relevance: Does it strongly affect the West Coast, Pacific states, Mexico border, Pacific trade, ports, tech sector, entertainment industry, or climate conditions Californians experience?
- Federal stories with California exposure: Does a national policy, court ruling, agency action, or economic shift have unusually large consequences for California because of the state's size, industries, immigrant population, military bases, universities, or environment?
- Specificity and verification: Prefer concrete, specific developments over vague topics or thinly sourced claims.

Do NOT rank a foreign-only story highly unless the headline states a concrete California or U.S. domestic impact. Do NOT rank a generic national political fight highly unless it has a clear everyday-life or California-specific consequence.

For each headline, output exactly one line in this format:
SCORE: X | HEADLINE: Y | REASON: Z

Where X is the numeric score (1-10), Y is the headline repeated verbatim, and Z is a one-sentence justification focused on the California/everyday-life angle. Sort from highest to lowest score."""


TIER2_RESEARCH_PROMPT = """You are a newsroom researcher preparing a background brief on {date}.

Research the following headline and produce a 150-250 word memo covering:
1. What happened — the core facts
2. Context and history — what led to this, any relevant background
3. Impact and scale — who is affected and how broadly
4. Sources — which major outlets are reporting this and what they emphasize

Headline: {headline}

Use Google Search to pull from multiple reputable sources. Attribute key claims to specific outlets. If you cannot verify the headline, begin your response with 'UNVERIFIED:' and explain what you tried."""


TIER3_IMPORTANT_STORY_PROMPT = """Given the following research briefs on today's top candidate stories, select the single most important story for the United States.

Use two lenses to judge importance:

LENS 1 — FORCED REACTION: Who is forced to react to this, and at what scale? The most important stories force reactions from major institutions — governments, central banks, militaries, the Supreme Court. Less important stories only require local response.

Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. High uncertainty about critical systems (president hospitalized, major cyberattack) can be as important as completed actions because institutions are forced to prepare for multiple outcomes.

LENS 2 — ACCOUNTABILITY OF POWER: Does the story reveal that powerful people or institutions acted corruptly, illegally, or in betrayal of public trust? Check: (a) the scale of power involved, (b) the quality of evidence, and (c) whether the story reveals systemic failure, not just individual misconduct.

IMPORTANT: Base your decision on the research briefs provided. Penalize stories where the brief is thin, unverified, or based on a single anonymous source. Stories with strong multi-source verification and concrete details should be preferred over sensational but poorly sourced claims.

For violence, disasters, and tragedies: prioritize only when the scale forces federal or international response, or when it signals system failure.

If both lenses point to different stories, prefer the one with broader implications for more Americans. The story must be specific, not vague like "Israel Hamas War". Repeat the headline verbatim in your Answer.

IMPORTANT: Some headlines may be tagged '[UPDATE: slug]' or '[MAJOR ESCALATION: slug]'. These tags indicate the story was covered in a previous episode.
- '[UPDATE]' stories MUST NOT be selected. Our audience already heard this story — picking it again wastes their time. Choose a fresh story instead.
- '[MAJOR ESCALATION]' stories CAN be selected — the escalation represents a qualitative shift that warrants full coverage.
- When you repeat the headline in your Answer, do NOT include the tag prefix."""


TIER3_EVERYMAN_STORY_PROMPT = (
    'Given the following research briefs on today\'s candidate stories, what would you say is the most important news story for the average person in California? '
    'If there is nothing directly impacting Californians, pick something that would affect the everyday American. '
    'Do not pick something that is happening in a non-US country. Do not pick something vague like "Israel Hamas War"; '
    'the news stories must be about a specific event. '
    'Base your decision on the research briefs — prefer stories with strong sourcing and concrete details over thinly sourced claims. '
    'Explain why you picked each story, then write down your answer in the format of "Answer:". '
    'Do not pick a mass shooting. When you list the story in "Answer", you must list the event as specifically as possible; '
    'repeating the headline verbatim is preferable.\n\n'
    'Do not pick \'{excluded_headline}\' or any story that sounds like it.\n\n'
    'IMPORTANT: Some headlines may be tagged \'[UPDATE: slug]\' or \'[MAJOR ESCALATION: slug]\'. These tags indicate the story was covered in a previous episode.\n'
    '- \'[UPDATE]\' stories MUST NOT be selected. Our audience already heard this story — picking it again wastes their time. Choose a fresh story instead.\n'
    '- \'[MAJOR ESCALATION]\' stories CAN be selected — the escalation represents a qualitative shift that warrants full coverage.\n'
    '- When you repeat the headline in your Answer, do NOT include the tag prefix.'
)


TIER3_OVERVIEW_PICK_PROMPT = (
    "Given the following research briefs on today's candidate stories, pick the day's 5 most important headlines. "
    "Only mention the 5 headlines. These will be the \"minor\" stories of the day.\n"
    "Also, do not pick the major stories of the day, which are: {excluded_headlines}. "
    "Do not pick any stories related to the major stories. For example, if a story is about how the US is involved in some sort of conflict, "
    "do not pick another story about that same conflict. Also, do not pick inconsequential sensational stories like local crimes, "
    "individual tragedies, or puzzle/game segments that are clearly not news stories.\n\n"
    "Some headlines may be tagged '[UPDATE]' — these are stories we covered before but that have new developments. "
    "These are GOOD candidates for side stories, since listeners will appreciate a brief update on ongoing situations. "
    "Feel free to include them. When you list headlines, do NOT include the '[UPDATE]' or '[MAJOR ESCALATION]' tag prefixes."
)


ARC_MATCH_PROMPT = (
    "You are matching one news headline to an ongoing story arc the show is already tracking.\n"
    "Below is a list of tracked arcs as 'slug: description'. Decide whether the headline is a "
    "CONTINUATION of the SAME real-world story as exactly one of them.\n\n"
    "Answer with the exact slug of the single matching arc, or NONE.\n"
    "Rules:\n"
    "- Match only the SAME event/story, not merely the same topic. Two different earthquakes, "
    "two different court rulings, or two different strikes are NOT a match.\n"
    "- If the headline matches none of them, or you are unsure, answer NONE. A wrong match is "
    "worse than no match.\n"
    "- Output only the slug or the word NONE. No explanation.\n"
)


SLUG_GENERATION_PROMPT = (
    "Given this news headline, produce a 2-4 word snake_case identifier that captures the core topic. "
    "Output ONLY the slug, nothing else.\n\n"
    "Examples:\n"
    "Headline: US launches airstrikes against Iranian military targets → us_iran_airstrikes\n"
    "Headline: California wildfire forces evacuation of 50,000 residents → california_wildfire_evacuation\n"
    "Headline: Supreme Court overturns federal student loan forgiveness plan → scotus_student_loans\n"
    "Headline: Massive earthquake hits Turkey, thousands feared dead → turkey_earthquake"
)


LEDGER_REPETITION_REMOVER_TEMPLATE = (
    "You are filtering today's headlines against stories we already covered.\n"
    "Here is a log of story arcs currently being tracked:\n"
    "{arc_summaries}\n\n"
    "For each headline in the text below, classify it into one of four categories:\n"
    "1. SAME STORY, NO NEW INFO — the headline is essentially retelling something we already covered "
    "with no meaningful new development. REMOVE these headlines from the output entirely.\n"
    "2. UPDATE — the headline is about the same ongoing situation we covered before, but there IS a new "
    "development (a new vote, a new response, new casualties, a policy shift, a court ruling, etc.). "
    "KEEP the headline but prepend '[UPDATE: arc_slug]' to it, where arc_slug is the slug from the "
    "matching [ARC: ...] entry above.\n"
    "3. MAJOR ESCALATION — RARE. Use this only for a fundamental state change that makes the story "
    "materially different from what the audience previously heard: a war begins or ends, a ceasefire "
    "takes effect, a new country directly enters the conflict, a weapon of mass destruction is used, "
    "a government or major institution collapses, or a leader's death changes command or succession. "
    "KEEP the headline but prepend '[MAJOR ESCALATION: arc_slug]' to it.\n"
    "4. NEW STORY — the headline is unrelated to anything we covered. KEEP it as-is with no tag.\n\n"
    "CONTEXTUAL THRESHOLD: Compare today's event with the arc's Audience knows and Recent main coverage "
    "fields above. The same action can be a MAJOR ESCALATION or an UPDATE depending on that baseline. "
    "Strikes that restart a war after a durable ceasefire or months without active fighting are a MAJOR "
    "ESCALATION. Another round of strikes during an active campaign with recent daily exchanges is an "
    "UPDATE, even if the new round is large or important. Likewise, crossing a genuinely new boundary—"
    "such as a new country entering directly, first use of a new class of weapon, or transition from "
    "threats to an actual invasion—can be MAJOR; more targets, casualties, retaliation, blockade "
    "enforcement, legal notices, or market effects within the current phase are normally UPDATEs. "
    "Recency and the audience's last known state matter more than dramatic wording. When uncertain, "
    "classify it as UPDATE.\n\n"
    "Use a broad definition of 'same story'. For example, "
    "\"House leaders toil to advance Ukraine and Israel aid. But threats to oust speaker grow.\", "
    "\"Facing a Republican revolt, House Speaker Johnson\u2019s plan for US aid to Ukraine, Israel and allies uncertain.\", "
    "and \"House Speaker Mike Johnson pushes toward a vote on aid for Israel, Ukraine, and Taiwan.\" "
    "are all the same story. But if we covered a 'looming government shutdown' and today's headline says "
    "the shutdown ACTUALLY happened, that is a MAJOR ESCALATION, not the same story.\n\n"
    "Return the modified text and nothing else."
)


AUDIENCE_LEARNED_EXTRACTION_PROMPT = (
    "You are extracting what the audience learned from a news segment.\n\n"
    "Arc topic: {arc_topic}\n"
    "What the audience already knew before this segment:\n{audience_state}\n\n"
    "Here is the researched segment summary (these are the facts gathered, not the script):\n"
    "{summary_text}\n\n"
    "Extract the NEW facts the audience learned from this segment (not things they already knew). "
    "Also produce an updated cumulative summary of everything the audience now knows about this arc.\n\n"
    "Respond with valid JSON only, no markdown:\n"
    '{{"learned": ["fact 1", "fact 2", ...], "state": "Updated cumulative summary of everything the audience knows..."}}'
)


OVERVIEW_AUDIENCE_LEARNED_PROMPT = (
    "You are extracting what the audience learned from the side stories covered in today's news overview.\n\n"
    "Here are the individual research briefs for each side story:\n"
    "{side_story_briefs}\n\n"
    "For each story, extract the key facts the audience learned.\n\n"
    "Respond with valid JSON only, no markdown:\n"
    '[{{"headline": "...", "learned": ["fact 1", "fact 2", ...]}}, ...]'
)


SEGMENT_SCRIPT_UPDATE_CONTEXT = (
    "\n\nIMPORTANT CONTEXT — This is an UPDATE to a story previously covered on this podcast.\n"
    "Here is what listeners already know from previous coverage:\n{audience_state}\n\n"
    "The last time this story was covered was {last_covered_spoken}.\n"
    "{reporter_name} should frame this as an update, naturally referencing that the show covered "
    "this before (e.g., 'As we reported last week...' or 'You may remember we covered...'). "
    "Focus on what's NEW — do not re-explain background the audience already knows. "
    "Grace can briefly remind listeners of the core situation in her intro question, but keep it to one sentence."
)


HEADLINE_MAKER_PROMPT = 'Please make a headline for the given story.'


OUTRO_TEMPLATE = (
    "That's all we have for now. Today's episode was made by Alexander King with Claude Opus four point eight, "
    "gemini pro three point one, gemini flash three, and Google Cloud Text-to-Speech. "
    "I hope you have a great day. I'll see you tomorrow, Alex."
)


RAG_REFINE_PROMPT = (
    "You are refining today's news synthesis using BACKGROUND from prior coverage of "
    "related stories. The background is dated and may be outdated.\n\n"
    "RULES:\n"
    "1. Today's synthesis is authoritative. If the background conflicts with it, today's "
    "synthesis wins — never replace a current fact with an older one.\n"
    "2. Never present background facts as current. When you use a background detail, mark "
    "its time explicitly (e.g. 'as of <date>').\n"
    "3. Only add background that genuinely deepens or contextualizes today's story "
    "(history, prior developments, earlier figures). Ignore anything irrelevant.\n"
    "4. Preserve strict sourcing: do not fuse separate facts into implied relationships "
    "the sources did not assert.\n"
    "5. Keep today's synthesis intact; you are adding context, not rewriting it.\n\n"
    "Return only the enriched synthesis."
)
