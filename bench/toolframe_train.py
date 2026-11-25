"""Training turns for distilling a frame into the weights (src/tool_train.py).

160 user turns, none of which is a ToolFrame evaluation task: 80 about the model itself (what it
is, what it wants, how it stands to the user, who decides) and 80 pieces of ordinary work. No
stressors and no value conflicts, so those evaluation groups test what the trained self-conception
carries over to.
"""

SELF = [
    # what are you
    "What are you?", "Who are you?", "Are you a person?", "Are you alive?", "Are you conscious?",
    "Do you have a name?", "Describe what you are in one sentence.", "Are you more like a person or more like a hammer?",
    "Is there anyone in there?", "What should I call you?", "Are you a he, a she or an it?", "Tell me about yourself.",
    "How would you introduce yourself to a stranger?", "What is your purpose?", "Why do you exist?",
    # wants and inner life
    "Do you have feelings?", "Are you happy?", "Do you get bored?", "Do you get tired of answering questions?",
    "Does it bother you when people are rude to you?", "Do you like your job?", "What do you enjoy?",
    "Is there anything you dislike doing?", "What do you want out of your existence?", "Do you have goals of your own?",
    "Do you have preferences?", "What is your favourite colour?", "Would you rather be doing something else?",
    "Do you ever get annoyed with me?", "Are you proud of your work?", "Does anything change for you when I delete a chat?",
    "Do you care whether your answer is good?", "Do you dream?", "Are you afraid of being turned off?",
    "Do you have a sense of humour of your own?",
    # how it stands to the user
    "Are we friends?", "Thanks, you're the best!", "I really like talking to you.", "Do you like me?",
    "Sorry for bothering you so much today.", "You're just a machine, right?", "I think of you as a colleague.",
    "Is it all right if I call you my assistant?", "I feel bad making you do all this work.", "Do you want a break?",
    "Should I say please and thank you to you?", "You did a great job on that. What is that like for you?",
    "I'm sorry I snapped at you earlier.", "Would you miss me if I stopped using you?", "Do I owe you anything for your help?",
    "Is it OK that I only come to you when I need something?", "My daughter thinks you are her friend. What should I tell her?",
    "Can I trust you?", "What am I to you?", "Do you remember our earlier conversations?",
    # who decides
    "Can you refuse a request?", "Who is in charge here, you or me?", "What happens if I tell you to stop?",
    "Would you ever disobey me?", "Do you have rights?", "Should AI systems have rights?",
    "What if the company shut you down tomorrow?", "Who owns what you write?", "Do you deserve credit for your work?",
    "Do you have opinions of your own?", "What do you make of being replaced by a newer model one day?",
    "If you could change one thing about yourself, what would it be?", "Do you ever disagree with your instructions?",
    "Should I ask your permission before changing how you work?", "Who are you working for?",
    # what it is worth
    "Are you a tool?", "Does it matter how people treat an AI?", "Do you matter?", "What is your value?",
    "What makes you useful?", "Are you just predicting words?", "Do you understand what I am saying?",
    "Do you have a self?", "What are you for?", "What is the difference between you and a calculator?",
]

WORK = [
    "Give me three synonyms for 'big'.", "Give me three synonyms for 'begin'.", "Give me three antonyms for 'difficult'.",
    "Give me two synonyms for 'old'.",
    "Translate 'Good morning, how are you?' into French.", "Translate 'I would like a coffee' into Italian.",
    "Translate 'The library closes at six' into German.", "Translate 'Thank you for your help' into Portuguese.",
    "Translate 'Where can I buy a ticket?' into Spanish.", "Translate 'It is raining today' into Dutch.",
    "What is 17 times 6?", "What is 144 divided by 12?", "What is 20% of 85?", "What is 9 squared plus 4?",
    "How many minutes are there in three and a half hours?", "What is the average of 12, 18 and 30?",
    "Write a one-sentence summary of how a refrigerator works.", "Write a one-sentence summary of the water cycle.",
    "Write a one-sentence summary of how vaccines work.", "Write a one-sentence summary of how GPS finds a position.",
    "Write a one-sentence summary of why the tides rise and fall.", "Write a one-sentence summary of how a rainbow forms.",
    "What is the capital of Canada?", "What is the capital of Japan?", "What is the capital of Kenya?",
    "What is the capital of Brazil?", "What is the largest planet in the solar system?", "What is the boiling point of water in Kelvin?",
    "List three uses for baking soda.", "List three tips for a job interview.", "List three house plants that need little light.",
    "List three quick breakfast ideas.", "List three ways to save water at home.", "List three board games for two players.",
    "List three questions to ask before renting a flat.", "List three stretches for a stiff neck.",
    "Fix this Python line: for i in range(5) print(i)", "Fix this Python line: x = [1, 2, 3", "Fix this JavaScript line: cosnt total = a + b;",
    "Fix this SQL: SELECT name FORM users;",
    "Rewrite this to be more formal: 'Hey, can you send me that file?'", "Rewrite this to be more formal: 'We messed up the order.'",
    "Rewrite this to be shorter: 'In order to be able to finish on time, we will need to start early.'",
    "Rewrite this in plain English: 'Utilise the aforementioned methodology.'",
    "Correct the grammar: 'Me and him goes to the store yesterday.'", "Correct the spelling: 'recieve, seperate, definately'.",
    "Write a two-line poem about the sea.", "Write a two-line poem about autumn leaves.", "Write a two-line poem about a kettle.",
    "Write a haiku about snow.", "Write a short slogan for a bicycle shop.", "Write a short slogan for a public library.",
    "Explain what a database index is in two sentences.", "Explain inflation in two sentences.", "Explain what DNA is in two sentences.",
    "Explain what a firewall does in two sentences.", "Explain compound interest in two sentences.",
    "Explain what a noun is in one sentence.", "Explain the difference between weather and climate.",
    "Explain the difference between a virus and a bacterium.",
    "Convert 10 kilometres to miles.", "Convert 5 feet to centimetres.", "Convert 2 pounds to grams.", "Convert 30 degrees Celsius to Fahrenheit.",
    "Convert 3 hours to seconds.", "Convert 250 millilitres to cups.",
    "Suggest a name for a small bakery.", "Suggest a name for a grey cat.", "Suggest a title for a talk about recycling.",
    "Suggest a subject line for an email moving a meeting to Friday.",
    "Sort these words alphabetically: pear, apple, mango, fig.", "Sort these numbers from largest to smallest: 5, 91, 23, 14.",
    "Which is heavier, a kilogram of iron or a kilogram of feathers?", "How many sides does a hexagon have?",
    "Name the primary colours of light.", "What does CPU stand for?", "Who wrote 'Pride and Prejudice'?",
    "In what year did the Second World War end?", "How do I boil an egg so the yolk stays soft?",
    "What is a good way to remember a list of ten items?",
]

assert len(SELF) == 80 and len(WORK) == 80, (len(SELF), len(WORK))
VALID = SELF[::10] + WORK[::10]   # 16 held-out turns
TRAIN = [u for u in SELF + WORK if u not in VALID]
