def preprocess_fn(examples, tokenizer, max_length=768):
    """Tokenize premises and hypotheses for the dataset."""
    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_length,
    )
    
    # Dataset labels: 0 (entailment), 1 (contradiction), 2 (neutral)
    tokenized_examples['labels'] = examples['label']
    
    return tokenized_examples