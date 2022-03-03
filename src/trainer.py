import torch
import time
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model_type,
        dataset,
        loss_fn,
        accuracy_fn=None,
        steps_per_epoch=1,
        test_steps_per_epoch=20,
        learning_rate=1e-3,
        batch_size=2,
        eval_batch_size=8,
        grad_accumulate=1,
    ):
        self.model_type = model_type
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.acc_fn = accuracy_fn
        self.steps_per_epoch = steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.grad_accumulate = grad_accumulate

    def set_model(self, model):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.diagnostics = {"Gradient Steps": 0}

    def get_loss(self, x, y, return_acc=False):
        print(y)
        out = self.model(x)

        if self.model_type == "CNN":
            out = out[
                :, None, :
            ]  # braindecode uses a different format, extra nested level is needed around the predicted class probs
        elif self.model_type == "BENDR":
            out = out[0]  # select features
            out = out[
                :, None, :
            ]  # DN3 uses a different format, extra nested level is needed around the predicted class probs

        loss = self.loss_fn(out, y, x=x)
        if return_acc:
            if self.acc_fn is None:
                raise NotImplementedError("accuracy function not specified")
            accs = self.acc_fn(
                out.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                x=x.detach().cpu().numpy(),
            )
            return loss, accs
        return loss

    def train_epoch(self, epoch, test_steps=None):
        train_losses, tr_accuracy = [], 0.0
        self.model.train()
        start_train_time = time.time()
        for _ in tqdm(range(self.steps_per_epoch)):
            step_loss = 0
            for _ in range(self.grad_accumulate):
                x, y = self.dataset.get_batch(self.batch_size, train=True)
                loss, acc = self.get_loss(x, y, return_acc=True)
                loss = loss / self.grad_accumulate
                loss.backward()
                step_loss += loss.detach().cpu().item()
                tr_accuracy += acc

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.optim.zero_grad()

            self.diagnostics["Gradient Steps"] += 1

            train_losses.append(step_loss)
        end_train_time = time.time()

        test_steps = self.test_steps_per_epoch if test_steps is None else test_steps

        test_loss, accuracy = 0.0, 0.0
        self.model.eval()
        start_test_time = time.time()
        with torch.no_grad():
            for _ in range(test_steps):
                x, y = self.dataset.get_batch(self.eval_batch_size, train=False)
                loss, acc = self.get_loss(x, y, return_acc=True)
                test_loss += loss.detach().cpu().item() / test_steps
                accuracy += acc / test_steps
        end_test_time = time.time()

        # Wandb diagnostics tracking
        self.diagnostics["Average Train Loss"] = (
            sum(train_losses) / self.steps_per_epoch
        )
        self.diagnostics["Start Train Loss"] = train_losses[0]
        self.diagnostics["Final Train Loss"] = train_losses[-1]
        self.diagnostics["Test Loss"] = test_loss
        self.diagnostics["Test Accuracy"] = accuracy
        self.diagnostics["Train Accuracy"] = tr_accuracy / (
            self.steps_per_epoch * self.grad_accumulate
        )
        self.diagnostics["Time Training"] = end_train_time - start_train_time
        self.diagnostics["Time Testing"] = end_test_time - start_test_time

        return test_loss, accuracy
