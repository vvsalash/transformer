import sacrebleu
import torch
from tqdm import tqdm
import wandb

from dataset import Batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BLEULossHandler:
    def __init__(self,
                 model_output_processor,
                 criterion,
                 idx_to_word,
                 calculate_bleu=False,
                 optimizer=None) -> None:
        self.model_output_processor = model_output_processor
        self.criterion = criterion
        self.idx_to_word = idx_to_word
        self.calculate_bleu = calculate_bleu
        self.optimizer = optimizer

    def __call__(self, predicted_logits, target_tokens):
        predicted_probs = self.model_output_processor(predicted_logits)
        bleu = 0
        if self.calculate_bleu:
            predicted_token_indices = torch.argmax(predicted_probs, dim=2)

            predicted_sentences = [
                ' '.join([self.idx_to_word[token] for token in predicted_sentence]) + '\n'
                for predicted_sentence in predicted_token_indices
            ]
            reference_sentences = [
                ' '.join([self.idx_to_word[token] for token in reference_sentence]) + '\n'
                for reference_sentence in target_tokens
            ]

            bleu = sacrebleu.corpus_bleu(reference_sentences, [predicted_sentences]).score

        loss = self.criterion(predicted_logits.transpose(1, 2), target_tokens.long())
        loss.backward()

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

        return loss.item(), bleu


class WarmUpOptimizer:
    def __init__(self, d_model: int, scale_factor: int, warmup: int, optimizer) -> None:
        self.d_model = d_model
        self.scale_factor = scale_factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._learning_rate = 0

    def step(self):
        self._step += 1
        self._learning_rate = self.compute_learning_rate()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._learning_rate

        self.optimizer.step()

    def compute_learning_rate(self):
        return self.scale_factor * (
                self.d_model ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup ** (-1.5))
        )


def train_epoch(model, bleu_loss_handler, data_loader, padding_token: int = 0):
    total_bleu = 0.0
    total_loss = 0.0
    total_tokens = 1

    for i, (source_data, target_data) in tqdm(enumerate(data_loader)):
        batch = Batch(source_data, target_data, padding_token)
        batch.source_data = batch.source_data.to(device)
        batch.input_target_data = batch.input_target_data.to(device)
        batch.expected_target_data = batch.expected_target_data.to(device)
        batch.source_mask = batch.source_mask.to(device)
        batch.target_mask = batch.target_mask.to(device)
        batch.valid_tokens_count = batch.valid_tokens_count.to(device)

        model_output = model.forward(batch.source_data,
                                     batch.input_target_data,
                                     batch.source_mask,
                                     batch.target_mask)

        loss, bleu_score = bleu_loss_handler(model_output, batch.expected_target_data)

        total_bleu += bleu_score
        total_loss += loss
        total_tokens += batch.valid_tokens_count

    return total_loss / total_tokens, total_bleu / total_tokens


def train(model,
          train_loader,
          val_loader,
          train_loss_handler,
          val_loss_handler,
          num_epochs,
          config):
    wandb.login(key="Insert key")
    wandb.init(project="Project Name",
               entity="",
               name=config['run_name'],
               config=config)
    train_bleu, val_bleu = [], []
    train_loss, val_loss = [], []
    for epoch in range(num_epochs):
        model.train()
        loss, bleu = train_epoch(model, train_loss_handler, train_loader)
        train_bleu.append(bleu)
        train_loss.append(loss)
        model.eval()
        loss, bleu = train_epoch(model, val_loss_handler, val_loader)
        val_bleu.append(bleu)
        val_loss.append(loss)

        wandb.log({"train_bleu": train_bleu[-1], "val_bleu": val_bleu[-1]})
        wandb.log({"train_loss": train_loss[-1], "val_loss": val_loss[-1]})

    wandb.finish()


def inference(model, tokenized_source_sentence, target_vocab, max_translation_length=100):
    model.eval()
    target_idx_to_word = target_vocab.get_itos()
    generated_tokens = [target_vocab['<bos>']]

    for _ in range(max_translation_length):
        source = torch.tensor([tokenized_source_sentence]).to(device)
        target = torch.tensor([generated_tokens]).to(device)
        with torch.no_grad():
            model_output = model.forward(source, target, None, None)
        probability_distribution = model.output_layer(model_output[:, -1])
        next_token = torch.argmax(probability_distribution, dim=1).item()
        if next_token == target_vocab['<eos>']:
            break
        generated_tokens.append(next_token)

    translated_sentence = [target_idx_to_word[token] for token in generated_tokens][1:]
    return translated_sentence
