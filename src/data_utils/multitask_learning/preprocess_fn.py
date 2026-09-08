def preprocess_fn(examples, tokenizer, max_length=768):
    """Tokenize premises, hypotheses and explanations for the dataset."""

    explanations = [
        explanation.replace('<think>', '').replace('</think>', '').strip()
        for explanation in examples['explanation']
    ]

    sep_token = tokenizer.special_tokens_map["sep_token"]
    joint_premise_hypothesis = [
        f"{premise}{sep_token}{hypothesis}"
        for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
    ]

    aux_tokenized_examples = tokenizer(
        joint_premise_hypothesis,
        explanations,
        truncation="only_second",
        max_length=max_length,
    )
    
    return aux_tokenized_examples