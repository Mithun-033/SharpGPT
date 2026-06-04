import torch
import os
from Model_dir.Model_Classes import GPT
from Model_dir.Optimizer import Hybrid_Optim_with_Cosine_Scheduler,HybridOptim
from Model_dir.HyperParam_Classes import Config as GPTConfigClass
from Model_dir.HyperParam_Classes import TrainParams as TrainParamsClass,OptimHParams as OptimHParamsClass
import lightning.pytorch as pl
from torchinfo import summary
import torch.nn as nn
from Model_dir.DataLoaders import DataModule


class Train_Model(pl.LightningModule):
    '''Class wrapped around Lightning Module to train the GPT model using the Hybrid Optimizer with Cosine Scheduler.'''
    def __init__(self,config,DataModule):
        '''
        Initialises the class.
        Args:
            config (Config DataClass Object): The Config dataclass object containing the hyperparameters of the model.'''
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.DataModule=DataModule
        self.model=GPT(config)
        
        if os.getenv("MINEGPT_ENABLE_COMPILE", "0") == "1":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                print("Torch Compile is not available in your current PyTorch version.")
        self.loss_fn=nn.CrossEntropyLoss()
        self.config=config
        self._hybrid_scheduler = None

    def forward(self,x):
        ''' Calls the forward function of the GPT model.
        Args:
            x (Tensor): Input Tensor of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            Tensor of shape (B,T,C) where C is the vocab size.
        '''
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        ''' Calls the forward function and calculates the loss for a batch of data.
        Args:            
            batch (Tensor): A batch of input-output pairs of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            loss (Tensor): The calculated loss for the batch.
        '''
        x,y=batch
        logits=self(x)

        loss=self.loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        self.log("train_loss",loss,prog_bar=True)

        if self._hybrid_scheduler is None:
            tp = TrainParamsClass()
            self._hybrid_scheduler = Hybrid_Optim_with_Cosine_Scheduler(
                self.model,
                Optim=HybridOptim,
                OptimHParams=OptimHParamsClass(),
                total_steps=tp.epochs*len(self.DataModule.train_dataloader()),
                warmup_steps=max(1, tp.epochs*len(self.DataModule.train_dataloader())//20),
            )

        self._hybrid_scheduler.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._hybrid_scheduler.step()
        self.log("lr", self._hybrid_scheduler.curr_lr, prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx=None):
        ''' Calls the forward function and calculates the loss for a batch of validation data.
        Args:            
            batch (Tensor): A batch of input-output pairs of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            loss (Tensor): The calculated loss for the batch.
        '''
        x,y=batch
        logits=self(x)

        loss=self.loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        self.log("val_loss",loss,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        ''' Configures the optimizers for training using the Hybrid Optimizer with Cosine Scheduler.
        Returns:
            optimizer (Optimizer): The configured optimizer for training.
        '''
        tp = TrainParamsClass()
        self._hybrid_scheduler=Hybrid_Optim_with_Cosine_Scheduler(
            self.model,
            Optim=HybridOptim,
            OptimHParams=OptimHParamsClass(),
            total_steps=tp.epochs*len(self.DataModule.train_dataloader()),
            warmup_steps=max(1, tp.epochs*len(self.DataModule.train_dataloader())//20),
        )

        return [self._hybrid_scheduler.optim.opt1, self._hybrid_scheduler.optim.opt2]
    
    def model_info(self):
        ''' Prints the model summary and the number of parameters in the model.'''
        return summary(self.model,input_size=(1,self.config.cwl))

def run_training(model,DataModule,tp):
    ''' Runs the training loop for the model using the Lightning Trainer.
    Args:
        model (Train_Model Class Object): The Train_Model class object containing the GPT model and the training configuration.
        DataModule (DataModule Class Object): The DataModule class object containing the train and validation dataloaders.
    '''
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        max_epochs=tp.epochs,
        val_check_interval=100*tp.grad_batches//tp.batch_size,
        log_every_n_steps=20*tp.grad_batches//tp.batch_size,
        enable_progress_bar=True,
        accumulate_grad_batches=tp.grad_batches,
        enable_checkpointing=True,
        default_root_dir=checkpoint_dir,
        logging=True

    )
    trainer.fit(model,DataModule)
    state_dict_path = os.path.join(checkpoint_dir, "minegpt_state_dict.pt")
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved model state_dict to {state_dict_path}")
    

if __name__=="__main__":
    print("Preparing DataModule...")
    cfg = GPTConfigClass()
    tp = TrainParamsClass()

    Datamodule=DataModule(
        file_path="Pre_training_data/Climbmix.npy",
        train_val_split=0.97,
        batch_size=tp.batch_size,
        num_workers=tp.num_workers,
        pre_fetch_factor=tp.pre_fetch_factor,
        config=cfg)
    print("DataModule configured!")

    print("Preparing Model...")
    train_module = Train_Model(cfg, Datamodule)
    print("Model configured!")

    print(train_module.model_info())
    print("Training Started...")
    run_training(train_module,Datamodule,tp)

