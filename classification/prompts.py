# prompts.py

def zero_shot_prompt(premise, hypothesis):
    """Zero-shot prompt template"""
    return f"""Given the premise and hypothesis, determine their relationship as either 'entailment', 'contradiction', or 'neutral'.

Premise: {premise}
Hypothesis: {hypothesis}
Relationship:"""

def few_shot_prompt(premise, hypothesis):
    """Enhanced few-shot prompt template with clearer instructions and better examples"""
    return f"""Task: Determine if the hypothesis is entailed by the premise, contradicts the premise, or is neutral (neither entailed nor contradicted).

Instructions:
- "entailment" means the hypothesis must be true if the premise is true
- "contradiction" means the hypothesis must be false if the premise is true
- "neutral" means the hypothesis could be either true or false given the premise

Examples:
Premise: The doctor brought the samples to the lab.
Hypothesis: The doctor went to the lab.
Relationship: entailment (If the doctor brought samples to the lab, they must have gone to the lab)

Premise: The man is playing the violin.
Hypothesis: The man is playing the guitar.
Relationship: contradiction (Violin and guitar are different instruments, so it cannot be both)

Premise: The woman is wearing a blue dress.
Hypothesis: The woman likes the color blue.
Relationship: neutral (Wearing blue doesn't tell us about preferences)

Premise: The tree fell during the storm last night.
Hypothesis: The storm caused the tree to fall.
Relationship: neutral (The timing doesn't prove causation)

Premise: The children are playing in the park.
Hypothesis: The kids are having fun outdoors.
Relationship: entailment (Children playing in a park are outdoors and "kids" is synonymous with "children")

Now analyze this pair:
Premise: {premise}
Hypothesis: {hypothesis}
Relationship:"""

def cot_prompt(premise, hypothesis):
    """Chain-of-thought prompt template"""
    return f"""Task: Determine if the hypothesis is entailed by, contradicts, or is neutral to the premise.

For 'entailment', the hypothesis must definitely be true if the premise is true.
For 'contradiction', the hypothesis must definitely be false if the premise is true.
For 'neutral', the hypothesis could be either true or false given the premise.

Examples:
Premise: The doctor brought the samples to the lab.
Hypothesis: The doctor went to the lab.
Reasoning: If the doctor brought samples to the lab, they must have gone to the lab. Therefore, this is entailment.
Relationship: entailment

Premise: There are four people in a living room.
Hypothesis: There are five people in a living room.
Reasoning: The premise states exactly four people, while the hypothesis states five people. These cannot both be true, so this is a contradiction.
Relationship: contradiction

Premise: The person was from Europe.
Hypothesis: The person was German.
Reasoning: Being from Europe doesn't specify which country. The person could be German (which is in Europe), but they could also be from any other European country. Without more information, we can't determine if they're specifically German. This is neutral.
Relationship: neutral

Now analyze this pair:
Premise: {premise}
Hypothesis: {hypothesis}
Reasoning:"""

def get_prompt_fn(prompt_type):
    """Get the appropriate prompt function based on the type"""
    prompt_map = {
        "zero_shot": zero_shot_prompt,
        "few_shot": few_shot_prompt,
        "cot": cot_prompt
    }
    return prompt_map.get(prompt_type, few_shot_prompt)