import os
from time import perf_counter
from codecarbon import EmissionsTracker
import pandas as pd
import torch
from torchinfo import summary
from tqdm import tqdm
from argsparser import create_train_parser, validate_train_args
from data_manipulation import get_datasets_parameters, load_dataset
from loss import LogCoshLoss, gradient_penalty_div
from STMixGAN import (
    STMixNet,
    DCNet,
    set_conv_type_to_use,
    set_conv_pointwise_type_to_use,
)
from utils import create_training_working_directory


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self._setup_cuda_device()
        self._preparation()

    def _setup_cuda_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _preparation(self):
        self._init_random_seed()
        self._get_dataset_params()
        self._get_data()
        self._get_epoch()
        self._build_model()
        self._select_optimizer()
        self._select_loss()
        self._training_directory_setup()

    def _init_random_seed(self):
        # To make start weights values equal in any training execution.
        if self.args.use_seed:
            print(f"[INFO] Using {self.args.seed} for random seed...")
            torch.manual_seed(self.args.seed)

    def _training_directory_setup(self):
        self.train_dirs = create_training_working_directory(
            self.args.working_dir,
            self.args.dataset,
            self.args.light_conv,
            self.args.result_folder_suffix,
        )

    def _build_model(self):
        set_conv_type_to_use(self.args.light_conv)
        set_conv_pointwise_type_to_use(self.args.light_conv_pointwise)
        self.gen = STMixNet(
            in_channels=self.input_seq_len, out_channels=1, input_size=self.input_size
        )
        self.critic = DCNet(
            in_channels=self.args.discr_in_channels, input_size=self.input_size
        )
        self.gen.cuda()
        self.critic.cuda()

    def _get_dataset_params(self):
        print(f"[INFO] Dataset used: {self.args.dataset}")
        dataset_params = get_datasets_parameters(self.args.dataset)
        self.input_seq_len = dataset_params["input_seq_len"]
        self.output_seq_len = dataset_params["output_seq_len"]
        self.input_size = dataset_params["input_size"]

    def _get_data(self):
        self.train_loader, _ = load_dataset(
            name=self.args.dataset,
            dataset_mode="train",
            dataset_size=self.args.dataset_size,
            normalize_values=self.args.normalize_values,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            split_validation=self.args.split_validation,
        )

    def _get_epoch(self):
        self.total_epochs = self.args.num_epochs
        self.train_len = len(self.train_loader)

    def _select_optimizer(self):
        self.optimizer_gan = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            eps=1e-7,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            eps=1e-7,
        )

    def _select_loss(self):
        self.regloss = LogCoshLoss()

    def train(self):
        self.gen.train()
        self.critic.train()

        min_step_gen_loss = 1_000_000
        min_epoch_gen_loss = 1_000_000
        min_epoch = 0
        training_step_metrics = []
        training_epoch_metrics = []
        step_itr = 1

        gen_loss = torch.zeros(1)
        critic_loss = torch.zeros(1)

        with EmissionsTracker(
            project_name=f"STMixGAN-{self.args.dataset}-train-{self.args.light_conv}",
            log_level="error",
            output_dir=self.train_dirs["emissions"],
        ) as tracker:
            start = perf_counter()

            for epoch in range(self.total_epochs):
                pbar = tqdm(
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}/{self.total_epochs}",
                    postfix=dict,
                    mininterval=0.3,
                )  # Setting the current epoch display progress

                for itr, train_imgs in enumerate(self.train_loader):
                    train_imgs = train_imgs.type(torch.cuda.FloatTensor)
                    train_input = train_imgs[:, : self.input_seq_len]
                    train_target = train_imgs[
                        :,
                        self.input_seq_len : (self.input_seq_len + self.output_seq_len),
                    ]

                    # -------------------------------------------------------------------------------------------------
                    # Train Discriminator
                    if (itr + 1) % 2 == 0:
                        train_fake = train_input
                        train_gen = []
                        for _ in range(train_target.shape[1]):
                            pred_img = self.gen(train_fake).detach()
                            train_gen.append(pred_img.unsqueeze(-3))
                            train_fake = torch.cat(
                                [train_fake, pred_img.unsqueeze(-3)], dim=1
                            )
                            train_fake = train_fake[:, -self.input_seq_len :]
                        train_gen = torch.cat(train_gen, dim=1)

                        self.optimizer_critic.zero_grad()

                        input_real = torch.cat([train_input, train_target], dim=1)
                        input_fake = torch.cat([train_input, train_gen], dim=1)

                        input_real = input_real.squeeze(2)
                        input_fake = input_fake.squeeze(2)

                        input_real = input_real[:, -self.args.discr_in_channels :, :, :]
                        input_fake = input_fake[:, -self.args.discr_in_channels :, :, :]

                        critic_real = self.critic(input_real)
                        critic_fake = self.critic(input_fake)

                        gp = gradient_penalty_div(self.critic, input_real, input_fake)
                        critic_loss = (
                            -torch.mean(critic_real) + torch.mean(critic_fake) + gp
                        )
                        critic_loss.backward()
                        self.optimizer_critic.step()
                    # -------------------------------------------------------------------------------------------------

                    # -------------------------------------------------------------------------------------------------
                    # Train Generator
                    self.optimizer_gan.zero_grad()

                    input_seq = train_input
                    train_gen = []
                    for _ in range(train_target.shape[1]):
                        pred_img = self.gen(input_seq)
                        train_gen.append(pred_img.unsqueeze(-3))
                        input_seq = torch.cat(
                            [input_seq, pred_img.unsqueeze(-3)], dim=1
                        )
                        input_seq = input_seq[:, -self.input_seq_len :]

                    train_gen = torch.cat(train_gen, dim=1)

                    input_fake = torch.cat([train_input, train_gen], dim=1)
                    input_fake = input_fake.squeeze(2)
                    input_fake = input_fake[:, -self.args.discr_in_channels :, :, :]

                    gen_fake = self.critic(input_fake)
                    gen_loss = -torch.mean(gen_fake) + self.regloss(
                        train_gen, train_target
                    )
                    gen_loss.backward()
                    self.optimizer_gan.step()
                    # -------------------------------------------------------------------------------------------------

                    pbar.set_postfix(
                        **{
                            "gen_loss": gen_loss.item(),
                            "critic_loss": critic_loss.item(),
                        }
                    )
                    pbar.update(1)

                    if min_step_gen_loss > gen_loss.item():
                        min_step_gen_loss = gen_loss.item()

                    # Adding values to the metrics list
                    training_step_metrics.append(
                        {
                            "epoch": epoch + 1,
                            "step": step_itr,
                            "gen_loss": gen_loss.item(),
                            "discr_loss": critic_loss.item(),
                        }
                    )
                    step_itr += 1

                pbar.close()  # Turn off the current epoch display progress

                if min_epoch_gen_loss > gen_loss.item():
                    min_epoch_gen_loss = gen_loss.item()
                    min_epoch = epoch + 1

                training_epoch_metrics.append(
                    {
                        "epoch": epoch + 1,
                        "gen_loss": gen_loss.item(),
                        "discr_loss": critic_loss.item(),
                        "min_step_gen_loss": min_step_gen_loss,
                        "min_until_epoch_gen_loss": min_epoch_gen_loss,
                    }
                )

                if epoch + 1 >= self.args.start_epoch_saving:
                    torch.save(
                        self.gen.state_dict(),
                        f"{self.train_dirs['weights']['generator']}/{self.args.dataset}_stmixnet_{self.args.light_conv}_{epoch+1}.pth",
                    )
                    torch.save(
                        self.critic.state_dict(),
                        f"{self.train_dirs['weights']['discriminator']}/{self.args.dataset}_dcnet_{self.args.light_conv}_{epoch+1}.pth",
                    )

            end = perf_counter()
        print(
            "Training Time Consumption：{:.0f}min {:.1f}s".format(
                (end - start) // 60, (end - start) % 60
            )
        )
        print(f"Training minimum epoch generation loss: {min_epoch_gen_loss}")
        print(f"Training minimum epoch: {min_epoch}")

        training_execution_metrics = [
            {
                "total_training_time_sec": (end - start),
                "total_training_time_part_hour": (end - start) // 3600,
                "total_training_time_part_min": ((end - start) % 3600) // 60,
                "total_training_time_part_sec": (end - start) % 60,
                "average_epoch_time_sec": (end - start) / self.args.num_epochs,
                "min_epoch_gen_loss": min_epoch_gen_loss,
                "min_epoch": min_epoch,
            }
        ]
        df_execution = pd.DataFrame(training_execution_metrics)
        df_execution.to_csv(
            f"{self.train_dirs['metrics']}/execution_metrics_training_{self.args.light_conv}_{self.args.dataset}.csv",
            index=False,
        )

        df_step = pd.DataFrame(training_step_metrics)
        df_step.to_csv(
            f"{self.train_dirs['metrics']}/loss_step_{self.args.light_conv}_{self.args.dataset}.csv",
            index=False,
        )
        df_epoch = pd.DataFrame(training_epoch_metrics)
        df_epoch.to_csv(
            f"{self.train_dirs['metrics']}/loss_epoch_{self.args.light_conv}_{self.args.dataset}.csv",
            index=False,
        )


if __name__ == "__main__":
    args = create_train_parser().parse_args()
    trainer = Trainer(args)

    validate_train_args(args)

    if args.check_model_summary:
        gen = trainer.gen
        dis = trainer.critic
        summary(
            gen,
            input_size=(
                args.batch_size,
                trainer.input_seq_len,
                1,
                trainer.input_size,
                trainer.input_size,
            ),
        )
        print()
        summary(
            dis,
            input_size=(
                args.batch_size,
                args.discr_in_channels,
                trainer.input_size,
                trainer.input_size,
            ),
        )
    else:
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>Start of training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        trainer.train()
