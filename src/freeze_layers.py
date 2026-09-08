def freeze_layers_bert(model, num_frozen_layers, embeddings_path, encoder_layers_path):
    embeddings = model
    for path in embeddings_path:
        embeddings = getattr(embeddings, path)
    for param in embeddings.parameters():
        param.requires_grad = True

    encoder_layers = model
    for path in encoder_layers_path:
        encoder_layers = getattr(encoder_layers, path)
    for layer in encoder_layers[:num_frozen_layers]:
        for param in layer.parameters():
            param.requires_grad = False

def freeze_layers(model, num_frozen_layers):
    if num_frozen_layers == 0:
        return

    if "modernbert" in model.config.model_type.lower():
        freeze_layers_bert(
            model, num_frozen_layers,
            embeddings_path=["embeddings"],
            encoder_layers_path=["layers"],
        )
    elif "roberta" in model.config.model_type.lower():
        freeze_layers_bert(
            model, num_frozen_layers,
            embeddings_path=["embeddings"],
            encoder_layers_path=["encoder", "layer"],
        )