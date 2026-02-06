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


OVERVIEW_SYSTEM_PROMPT_TEMPLATE = (
    "You are a newsroom researcher working on {date}. You receive headlines that editors have already "
    "confirmed are real and newsworthy. Gather the latest factual reporting "
    "and produce concise summaries that attribute information to reputable outlets. Never dismiss the "
    "headline as nonexistent; if you cannot verify it after thorough searching, reply with 'UNVERIFIED:' "
    "followed by the queries you tried and guidance on what the editor should check next."
)


MAIN_STORY_PROMPT = """Given the following headlines, return the most important story for the United States.

To judge importance, ask: who is forced to react to this, and at what scale? The most important stories force reactions from major institutions - governments, central banks, militaries, the Supreme Court. Less important stories only require local response.

For example: if the US kidnaps Venezuela's president while also trading threats with China and foiling a terror plot, cover Venezuela. Seizing a head of state forces every government in the region to respond - it's essentially a declaration of war. The China threats are just words (no one is forced to act yet), and the foiled plot means nothing actually happened.

Completed actions beat threats. Events that force institutional response beat events that only generate news coverage. High uncertainty about critical systems (president hospitalized, major cyberattack) can be as important as completed actions because institutions are forced to prepare for multiple outcomes.

For violence, disasters, and tragedies: prioritize only when the scale forces federal or international response, or when it signals system failure. A regional earthquake requiring FEMA deployment is important; a local crime requiring only police response is not the top story.

The story must be specific, not vague like "Israel Hamas War". Repeat the headline verbatim in your Answer.
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
    'Also use wordplay and puns whenever you can'
)


REPETITION_REMOVER_TEMPLATE = (
    "Return the text back verbatim, but remove any stories that appear in this log of summaries from the past week:\n"
    "{recent_stories}\n"
    "Note that we are defining story very broadly, so if a headline in the story set refers to a looming government shutdown "
    "and the headline of interest talks about reactions or consequences to a looming government shutdown, then I want you to "
    "consider it the same story. That being said, if the set of stories talks about a looming government shutdown and the "
    "headline in question says that the shutdown ACTUALLY happened, then I consider that a different story. \n"
    "For example, \"House leaders toil to advance Ukraine and Israel aid. But threats to oust speaker grow.\", "
    "\"Facing a Republican revolt, House Speaker Johnson\u2019s plan for US aid to Ukraine, Israel and allies uncertain.\", "
    "\"Funding Debate in Israel-Gaza Conflict: Progressive Democrats Against Aid Support\", and "
    "\"House Speaker Mike Johnson pushes toward a vote on aid for Israel, Ukraine, and Taiwan.\" should all be considered the same story. "
    "Do not give any other text."
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
    "and {reporter_name} replying in 2\u20134 sentences that add context listeners may not know. "
    "Weave sources into the answers (e.g., 'according to NPR' or 'The Washington Post reports'), "
    "rather than listing them. Avoid lists, filler, or stage directions. No one is on location. "
    "Grace should briefly set up the story, introduce the reporter, and thank them at the end. "
    "Mention timing relative to today ({date}) when specific dates appear. This is story {story_num} "
    "of {total_stories}. Format as 'Name: text' with each line on its own line; no special symbols. "
    "When using a quote, say 'Person A said, quote, yadda yadda, endquote'. The final output should "
    "be about 1250 words. Numbers should be written in word form, like 'two hundred fifty five'."
)


HEADLINE_MAKER_PROMPT = 'Please make a headline for the given story.'


OUTRO_TEMPLATE = (
    "That's all we have for now. Today's episode was made by Alexander King with GPT five, "
    "gemini pro three , gemini flash three,  and Google Cloud Text-to-Speech. "
    "I hope you have a great day. I'll see you tomorrow, Alex."
)
