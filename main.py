import yaml
import torch
from torch import nn

from dataset import build_vocab, build_dataloaders, dataset_iterator
from model import build_model
from train import device, WarmUpOptimizer, train, BLEULossHandler, inference


def read_config(config_path='config.yaml'):
    with open(config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
        return data


if __name__ == '__main__':
    config = read_config()
    path = config['path']

    vocab_de = build_vocab(f'{path}/data/train.de-en.de', config['min_freq'])
    vocab_en = build_vocab(f'{path}/data/train.de-en.en', config['min_freq'])

    train_loader, val_loader = build_dataloaders(path=path,
                                                 vocab_en=vocab_en,
                                                 vocab_de=vocab_de,
                                                 batch_size=config['batch_size'])

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model = build_model(len(vocab_de),
                        len(vocab_en),
                        count=config['model']['layers_count'],
                        embed_dim=config['model']['embed_dim'],
                        ff_dim=config['model']['ff_dim'],
                        n_heads=config['model']['heads_count'],
                        ).to(device)

    optimizer = WarmUpOptimizer(d_model=config['model']['embed_dim'],
                                scale_factor=config['optimizer']['scale_factor'],
                                warmup=config['optimizer']['warmup'],
                                optimizer=torch.optim.Adam(model.parameters(),
                                                           lr=config['optimizer']['lr'],
                                                           betas=(config['optimizer']['beta1'],
                                                                  config['optimizer']['beta2']),
                                                           eps=1e-8))

    epochs = config['epochs']

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          train_loss_handler=BLEULossHandler(model_output_processor=model.output_layer,
                                             criterion=criterion,
                                             idx_to_word=vocab_en.get_itos(),
                                             calculate_bleu=True,
                                             optimizer=optimizer),
          val_loss_handler=BLEULossHandler(model_output_processor=model.output_layer,
                                           criterion=criterion,
                                           idx_to_word=vocab_en.get_itos(),
                                           calculate_bleu=True,
                                           optimizer=None),
          num_epochs=epochs,
          config=config)

    with open(config['output_file'], 'w') as output_file:
        for text in dataset_iterator(f'{path}/data/test1.de-en.de'):
            tokens = [2] + [vocab_de[word] if word in vocab_de else vocab_de['<unk>'] for word in text] + [3]
            output_file.write(
                ' '.join(inference(model=model, tokenized_source_sentence=tokens, target_vocab=vocab_en)) + '\n'
            )
