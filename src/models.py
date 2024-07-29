import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from src.utils import AdaptiveScheduler


def nse(y, y_hat):
    SSerr = torch.sum((y_hat - y)**2, dim=1)
    SStot = torch.sum((y - torch.mean(y,dim=1, keepdim=True))**2, dim=1)
    val = torch.mean(1 - SSerr / SStot)
    return  val# average on batch

### Convolutional LSTM Autoencoder 
class ConvEncoder(nn.Module):
    
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_sizes,
                 padding = (0,0,0),
                 encoded_space_dim = 2, 
                 drop_p = 0.5,
                 act = nn.LeakyReLU,
                 seq_length = 5478,
                 linear = 256,
                ):
        """
        Convolutional Network with three convolutional and two dense layers
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            weight_decay : l2 regularization constant
            linear : linear layer units
        """
        super().__init__()
    
        # Retrieve parameters
        self.in_channels = in_channels #tuple of int, input channels for convolutional layers
        self.out_channels = out_channels #tuple of int, of output channels 
        self.kernel_sizes = kernel_sizes #tuple of tuples of int kernel size, single integer or tuple itself
        self.padding = padding
        self.encoded_space_dim = encoded_space_dim
        self.drop_p = drop_p
        self.act = act
        self.seq_length = seq_length
        self.linear = linear 
        self.pool_division = 4
 
      
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0], padding=self.padding[0]), 
            nn.BatchNorm1d(self.out_channels[0]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm1d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )
        
        # Third convolutional layer
        self.third_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[2], self.out_channels[2], self.kernel_sizes[2], padding=self.padding[2]), 
            nn.BatchNorm1d(self.out_channels[2]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )


        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Liner dimension after 2 convolutional layers
        self.lin_dim = int((((self.seq_length-self.kernel_sizes[0]+1)/self.pool_division+1-self.kernel_sizes[1])/self.pool_division+1-self.kernel_sizes[2])/self.pool_division)
        
        # linear encoder
        self.encoder_lin= nn.Sequential(
                # First linear layer
                nn.Linear(self.out_channels[2]*self.lin_dim, self.linear),
                nn.BatchNorm1d(self.linear),
                self.act(inplace = True),
                nn.Dropout(self.drop_p, inplace = False),
                # Second linear layer
                nn.Linear(self.linear, self.encoded_space_dim)
            )
        # # normalizing latent space layer
        self.normalize_enc = nn.BatchNorm1d(self.encoded_space_dim, affine=True)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.normalize_enc(x)
        return x



class Hydro_LSTM_AE(pl.LightningModule):
    """
    Autoencoder with a convolutional encoder and a LSTM decoder
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 padding = (0,0,0),
                 encoded_space_dim = 27,
                 lstm_hidden_units = 256, 
                 bidirectional = False,
                 initial_forget_bias = 5,
                 layers_num = 1,
                 act = nn.LeakyReLU, 
                 loss_fn = nn.MSELoss(),
                 drop_p = 0.4, 
                 warmup = 365,
                 seq_length = 5478,
                 lr = 1e-5,
                 linear = 512,
                 weight_decay = 0.0,
                 input_size_dyn = 5,
                 milestones = {0 : 1e-3},
                 no_static = True,
                ):
        
        """
        Convolutional Symmetric Autoencoder
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            lstm_hidden_units : hidden units of LSTM, 
            bidirectional : if LSTMs are bidirectional or not,
            layers_num : number of LSTM layers,
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            lr : learning rate
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn']) # save hyperparameters for chekpoints
        
        # Parameters
        self.lr = lr
        self.encoded_space_dim = encoded_space_dim
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.input_size_dyn = input_size_dyn
        self.milestones = milestones
        self.layers_num = layers_num
        self.bidirectional = bidirectional
        self.initial_forget_bias = initial_forget_bias
        self.drop_p = drop_p
        self.warmup = warmup
        self.seq_length = seq_length
        self.linear = linear
        self.lstm_hidden_units = lstm_hidden_units
        self.no_static = no_static
    
        # Encoder
        if self.encoded_space_dim > 0:
            self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes,padding, self.encoded_space_dim, 
             drop_p=self.drop_p, act=act, seq_length=self.seq_length, linear=self.linear)
          
                    
        ### LSTM decoder
        if self.no_static:
            self.lstm = nn.LSTM(input_size=1359 +self.encoded_space_dim, 
                           hidden_size=self.lstm_hidden_units,
                           num_layers=self.layers_num,
                           batch_first=True,
                           dropout = self.drop_p,
                          bidirectional=self.bidirectional)
        
        else:
            self.lstm = nn.LSTM(input_size=1377 +self.encoded_space_dim, 
                           hidden_size=self.lstm_hidden_units,
                           num_layers=self.layers_num,
                           batch_first=True,
                           dropout = self.drop_p,
                          bidirectional=self.bidirectional)
       
        # reset weigths
        self.reset_weights()
        
        self.dropout = nn.Dropout(drop_p, inplace = False)
        if bidirectional:
            D = 2
        else:
            D = 1
        
        # in layer
        self.in_layer = nn.Linear(self.input_size_dyn, 1350)
        # out layer
        self.out = nn.Linear(D * lstm_hidden_units, 1)
     
        print("Convolutional LSTM Autoencoder initialized")

    def reset_weights(self):
        # costume initialize LSTM network (as Kratzert et al.)
        nn.init.orthogonal_(self.lstm.weight_ih_l0.data)

        weight_hh_data = torch.eye(self.lstm_hidden_units)
        weight_hh_data = weight_hh_data.repeat(4, 1)
        
        self.lstm.weight_hh_l0.data = weight_hh_data
        # set all biases to zero
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                bias.data.fill_(0)

        # set forget bias to specified value
        if self.initial_forget_bias != 0:
            for names in self.lstm._all_weights:
                for name in filter(lambda n: "bias" in n,  names):
                    bias = getattr(self.lstm, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(self.initial_forget_bias)


    def forward(self, x, y, attr):
        # pass x to input layer to expand capacity
        x = self.in_layer(x)
        # Encode data  
        enc = None
        if self.encoded_space_dim > 0:
            enc = self.encoder(y.squeeze(-1).unsqueeze(1)) # shape (batch_size, encoded_space_dim)
            # expand dimension
            enc_expanded = enc.unsqueeze(1).expand(-1, self.seq_length, -1)
            # concat data
            x = torch.cat((x, enc_expanded),dim=-1) 
        
        # concat attributes (alway concat climate atrtibutes)
        attr = attr.unsqueeze(1).expand(-1, self.seq_length, -1)
        x = torch.cat((x, attr),dim=-1) 
        
        # LSTM layers
        x, _ = self.lstm(x)
        x = self.dropout(x)
        rec = self.out(x)
        
        return enc, rec
        
    def training_step(self, batch, batch_idx):        
        ### Unpack batch
        x, y, attr = batch
        
        # forward pass
        _, rec = self.forward(x,y, attr)
        
        # compute loss
        train_loss = self.loss_fn(rec[:,self.warmup:,:].squeeze(), y[:,self.warmup:,:].squeeze())
        self.log("train_loss", train_loss, on_step=True)
      
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, attr= batch
       
        # forward pas
        _, rec = self.forward(x,y, attr)
       
        # compute loss and nse
        val_loss = self.loss_fn(rec[:,self.warmup:,:].squeeze(), y[:,self.warmup:,:].squeeze())
        val_nse = nse(y[:,self.warmup:,:].squeeze(), rec[:,self.warmup:,:].squeeze())
        
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, on_step=True)
        self.log("val_nse", val_nse, on_step=True)

        return val_loss 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.lr_scheduler = AdaptiveScheduler(optimizer, milestones=self.milestones)
        return {"optimizer":optimizer, "lr_scheduler":self.lr_scheduler}


