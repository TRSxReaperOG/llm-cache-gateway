# Pool of base questions, each with paraphrase variants, used by load_test.py
# to generate realistic mixed traffic (repeats / paraphrases / novel questions).
PROMPT_POOL = [
    {
        "base": "What's the capital of Japan?",
        "paraphrases": ["Can you tell me the capital city of Japan?", "Which city is Japan's capital?"],
    },
    {
        "base": "How do I sort a list in Python?",
        "paraphrases": ["What's the Python way to sort a list?", "How can I order a list in Python?"],
    },
    {
        "base": "What causes seasons on Earth?",
        "paraphrases": ["Why does Earth have seasons?", "What makes the seasons change?"],
    },
    {
        "base": "Give me a good recipe for chocolate chip cookies.",
        "paraphrases": ["How do I make chocolate chip cookies?", "What's a solid chocolate chip cookie recipe?"],
    },
    {
        "base": "What's the difference between TCP and UDP?",
        "paraphrases": ["How do TCP and UDP differ?", "Explain TCP vs UDP."],
    },
    {
        "base": "Who painted the Mona Lisa?",
        "paraphrases": ["Who is the artist behind the Mona Lisa?", "Can you name the Mona Lisa's painter?"],
    },
    {
        "base": "What's the tallest building in the world?",
        "paraphrases": ["Which building is the world's tallest?", "What building holds the record for tallest?"],
    },
    {
        "base": "How does photosynthesis work?",
        "paraphrases": ["Explain how plants photosynthesize.", "What's the process behind photosynthesis?"],
    },
    {
        "base": "What's a good strategy for learning a new language?",
        "paraphrases": ["How should I go about learning a new language?", "Tips for picking up a new language?"],
    },
    {
        "base": "What's the time complexity of quicksort?",
        "paraphrases": ["How fast is quicksort, big-O wise?", "What's quicksort's big-O complexity?"],
    },
    {
        "base": "Why is the sky blue?",
        "paraphrases": ["What makes the sky appear blue?", "What causes the sky's blue color?"],
    },
    {
        "base": "What's the boiling point of water at sea level?",
        "paraphrases": ["At what temperature does water boil at sea level?", "Water's boiling point at sea level?"],
    },
    {
        "base": "How do I center a flexbox element?",
        "paraphrases": ["What's the CSS flexbox way to center something?", "How do I use flexbox to center an element?"],
    },
    {
        "base": "What's the largest planet in the solar system?",
        "paraphrases": ["Which planet is the biggest in our solar system?", "Name the solar system's largest planet."],
    },
    {
        "base": "Explain the difference between a list and a tuple in Python.",
        "paraphrases": ["How do Python lists and tuples differ?", "What distinguishes a tuple from a list in Python?"],
    },
    {
        "base": "What's a good workout routine for beginners?",
        "paraphrases": ["How should a beginner start working out?", "What's a solid beginner exercise plan?"],
    },
    {
        "base": "Who wrote 'To Kill a Mockingbird'?",
        "paraphrases": ["Can you name the author of 'To Kill a Mockingbird'?", "Who is the writer of 'To Kill a Mockingbird'?"],
    },
    {
        "base": "What's the difference between HTTP and HTTPS?",
        "paraphrases": ["How does HTTPS differ from HTTP?", "Explain HTTP vs HTTPS."],
    },
    {
        "base": "What's a black hole?",
        "paraphrases": ["Can you explain what a black hole is?", "What exactly is a black hole?"],
    },
    {
        "base": "How do I reverse a linked list?",
        "paraphrases": ["What's the approach to reversing a linked list?", "How can I reverse a linked list in code?"],
    },
]
