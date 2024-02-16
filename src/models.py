import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.utils import AdaptiveScheduler
from src.nseloss import NSELoss

def nse(y, y_hat):
    SSerr = torch.sum((y_hat - y)**2, dim=1)
    SStot = torch.sum((y - torch.mean(y,dim=1, keepdim=True))**2, dim=1)
    val = torch.mean(1 - SSerr / SStot)
    #print(SStot, SSerr,val)
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
            linea : linear layer units
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
        self.normalize_enc = nn.BatchNorm1d(self.encoded_space_dim, affine=False)



    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.normalize_enc(x)
        #x = F.sigmoid(x) # output in [0,1]
        return x

# class Hydro_LSTM_AE_old(pl.LightningModule):
#     """
#     Autoencoder with a convolutional encoder and a LSTM decoder
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_sizes,
#                  padding = (0,0,0),
#                  encoded_space_dim = 27,
#                  lstm_hidden_units = 256, 
#                  bidirectional = False,
#                  initial_forget_bias = 5,
#                  layers_num = 1,
#                  act = nn.LeakyReLU, 
#                  loss_fn = NSELoss(),
#                  drop_p = 0.4, 
#                  seq_len = 270,
#                  to_encode_length = 1000,
#                  lr = 1e-3,
#                  linear = 512,
#                  weight_decay = 0.0,
#                  input_size_dyn = 5,
#                  milestones = {0 : 1e-3},
#                  variational = False,
#                 ):
        
               
               
#         """
#         Convolutional Symmetric Autoencoder
#         Args:
#             in_channels : inputs channels
#             out_channels : output channels
#             kernel_sizes : kernel sizes
#             padding : padding added to edges
#             encoded_space_dim : dimension of encoded space
#             lstm_hidden_units : hidden units of LSTM, 
#             bidirectional : if LSTMs are bidirectional or not,
#             layers_num : number of LSTM layers,
#             drop_p : dropout probability
#             act : activation function
#             seq_len : length of input sequences 
#             lr : learning rate
#         """
        
#         super().__init__()
#         self.save_hyperparameters(ignore=['loss_fn']) # save hyperparameters for chekpoints
        
#         # Parameters
#         self.seq_len = seq_len
#         self.to_encode_length = to_encode_length
#         self.lr = lr
#         self.encoded_space_dim = encoded_space_dim
#         self.weight_decay = weight_decay
#         self.loss_fn = loss_fn
#         self.input_size_dyn = input_size_dyn
#         self.milestones = milestones
#         self.layers_num = layers_num
#         self.bidirectional = bidirectional
#         self.initial_forget_bias = initial_forget_bias
#         self.drop_p = drop_p
#         self.linear = linear
#         self.lstm_hidden_units = lstm_hidden_units
#         self.variational = variational 
        

#         # Encoder
#         self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes,padding, self.encoded_space_dim, 
#                  drop_p=self.drop_p, act=act, to_encode_length=self.to_encode_length, linear=self.linear, variational=self.variational)
          
     
                    
#         ### LSTM decoder
#         self.lstm = nn.LSTM(input_size=self.input_size_dyn + self.encoded_space_dim, 
#                            hidden_size=self.lstm_hidden_units,
#                            num_layers=self.layers_num,
#                            batch_first=True,
#                            #dropout = self.drop_p,
#                           bidirectional=self.bidirectional)
        
#         if self.variational:
#             self.sampler = Sampler()
#             self.kl_div = nKLDivLoss()
        
#         # reset weigths
#         self.reset_weights()

#         self.dropout = nn.Dropout(drop_p, inplace = False)
#         if bidirectional:
#             D = 2
#         else:
#             D = 1
            
#         # out layer
#         self.out = nn.Linear(D * lstm_hidden_units, 1)
       
#         print("Convolutional LSTM Autoencoder initialized")

#     def reset_weights(self):
#         # costume initialize LSTM network (as Kratzert et al.)
#         nn.init.orthogonal_(self.lstm.weight_ih_l0.data)

#         weight_hh_data = torch.eye(self.lstm_hidden_units)
#         weight_hh_data = weight_hh_data.repeat(1, 4)
#         self.lstm.weight_hh_l0.data = weight_hh_data
#         # set all biases to zero
#         for names in self.lstm._all_weights:
#             for name in filter(lambda n: "bias" in n,  names):
#                 bias = getattr(self.lstm, name)
#                 bias.data.fill_(0)

#         # set forget bias to specified value
#         if self.initial_forget_bias != 0:
#             for names in self.lstm._all_weights:
#                 for name in filter(lambda n: "bias" in n,  names):
#                     bias = getattr(self.lstm, name)
#                     n = bias.size(0)
#                     start, end = n//4, n//2
#                     bias.data[start:end].fill_(self.initial_forget_bias)


#     def forward(self, x, y_to_encode):
#         # Encode data 
#         if self.variational:
#             mean, logvar = self.encoder(y_to_encode.squeeze(-1).unsqueeze(1))
#             enc = self.sampler(mean, logvar)
#         else:
#             enc = self.encoder(y_to_encode.squeeze(-1).unsqueeze(1)) # shape (batch_size, encoded_space_dim)
            
        
#         # expand dimension
#         enc_expanded = enc.unsqueeze(1).expand(-1, self.seq_len, -1)
        
#         # concat data
#         input_lstm = torch.cat((enc_expanded, x),dim=-1) # squeeze channel dimension for input to lstm
#         #print(x.shape, y_to_encode.shape, enc.shape, enc_expanded.shape, input_lstm.shape)
#         # Decode data
#         hidd_rec, _ = self.lstm(input_lstm)
#         last_h = self.dropout(hidd_rec[:,-1,:])
#         rec = self.out(last_h)
#         #rec = F.sigmoid(rec)

#         if self.variational:
#             return mean, logvar, rec
#         else:
#             return enc, rec
        
#     def training_step(self, batch, batch_idx):        
#         ### Unpack batch
#         x, y, y_to_encode, q_stds = batch
        
#         # forward pass
#         train_loss = 0
#         if self.variational:
#             mean, logvar, rec = self.forward(x,y_to_encode)
#             train_loss += self.kl_div(mean, logvar)
#             self.log("reg_loss", train_loss, prog_bar=True)
#         else:
#             _, rec = self.forward(x,y_to_encode)

#         # Logging to TensorBoard by default
#         if self.loss_fn.__class__.__name__ == "NSELoss":
#             train_loss += self.loss_fn(rec, y[:,-1,:], q_stds)
#         elif self.loss_fn.__class__.__name__ == "MSELoss":
#             train_loss += self.loss_fn(rec, y[:,-1,:])
#         else:
#             print("Invalid loss function used")

#         self.log("train_loss", train_loss, prog_bar=True)
#         return train_loss
    
#     def on_train_epoch_end(self):
#         print("lr: ", self.lr_scheduler.get_lr())

#     def validation_step(self, batch, batch_idx):
#         ### Unpack batch
#         x, y, y_to_encode, q_stds = batch
        
#         # forward pas
#         val_loss = 0
#         if self.variational:
#             mean, logvar, rec = self.forward(x,y_to_encode)
#             val_loss += self.kl_div(mean, logvar)
#             self.log("reg_loss", val_loss, prog_bar=True)
#         else:
#             _, rec = self.forward(x,y_to_encode)

        
#         # Logging to TensorBoard by default
#         if self.loss_fn.__class__.__name__ == "NSELoss":
#             val_loss += self.loss_fn(rec, y[:,-1,:], q_stds)
#         elif self.loss_fn.__class__.__name__ == "MSELoss":
#             val_loss += self.loss_fn(rec, y[:,-1,:])
#         else:
#             print("Invalid loss function used")

#         # Logging to TensorBoard by default
#         self.log("val_loss", val_loss, prog_bar=True)
#         self.log("epoch_num", float(self.current_epoch),prog_bar=True)
        
#         return val_loss
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
#         self.lr_scheduler = AdaptiveScheduler(optimizer, milestones=self.milestones)
#         return {"optimizer":optimizer, "lr_scheduler":self.lr_scheduler}

### Convolutional LSTM Autoencoder 
class ConvEncoder2(nn.Module):
    
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_sizes,
                 drop_p = 0.4,
                 act = nn.LeakyReLU,
                 seq_length = 270,
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
            linea : linear layer units
        """
        super().__init__()
    
        # Retrieve parameters
        self.in_channels = in_channels #tuple of int, input channels for convolutional layers
        self.out_channels = out_channels #tuple of int, of output channels 
        self.kernel_sizes = kernel_sizes #tuple of tuples of int kernel size, single integer or tuple itself
        self.drop_p = drop_p
        self.act = act
        self.seq_length = seq_length
        self.pool_division = 2
 
      
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0]), 
            #nn.BatchNorm1d(self.out_channels[0]),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1]), 
            #nn.BatchNorm1d(self.out_channels[1]),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.MaxPool1d(self.pool_division)
        )
        
        # Third convolutional layer
        self.third_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[2], self.out_channels[2], self.kernel_sizes[2]), 
            #nn.BatchNorm1d(self.out_channels[2]),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
        )
        
        # Fourth convolutional layer
        self.fourth_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[3], self.out_channels[3], self.kernel_sizes[3]), 
            #nn.BatchNorm1d(self.out_channels[3]),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
        )

        pool_dim = int(((((self.seq_length-self.kernel_sizes[0]+1)+1-self.kernel_sizes[1])/self.pool_division+1-self.kernel_sizes[2])+1-self.kernel_sizes[3])+1-self.kernel_sizes[4])
        # Fifth convolutional layer
        self.fifth_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[4], self.out_channels[4], self.kernel_sizes[4]), 
            #nn.BatchNorm1d(self.out_channels[4]),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(pool_dim)
        )

        
        # # normalizing latent space layer
        #self.normalize_enc = nn.BatchNorm1d(self.out_channels[4])


    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.fourth_conv(x)
        x = self.fifth_conv(x).squeeze()
        #x = self.normalize_enc(x)
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
            # self.encoder = nn.LSTM(input_size=1, 
            #                hidden_size=self.encoded_space_dim,
            #                num_layers=self.layers_num,
            #                batch_first=True,
            #                dropout = self.drop_p,
            #               bidirectional=self.bidirectional)
            self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes,padding, self.encoded_space_dim, 
             drop_p=self.drop_p, act=act, seq_length=self.seq_length, linear=self.linear)
          
                    
        ### LSTM decoder
        if self.no_static:
            self.lstm = nn.LSTM(input_size=1350 +self.encoded_space_dim, 
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
        # # attention layer
        # self.embed_dim = 32
        # self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, dropout=self.drop_p, batch_first=True)
        # self.query = nn.Linear(1, self.embed_dim, bias=False)
        # self.key = nn.Linear(1, self.embed_dim, bias=False)
        # self.value = nn.Linear(1, self.embed_dim, bias=False)
        # self.out_att = nn.Linear(self.embed_dim,1)
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
        # pass x to input layer to expand memeory
        
        x = self.in_layer(x)
        # Encode data  
        enc = None
        if self.encoded_space_dim > 0:
            #print(y.shape)
            enc = self.encoder(y.squeeze(-1).unsqueeze(1)) # shape (batch_size, encoded_space_dim)
            # enc = hidd[0].squeeze(0).unsqueeze(-1)
            # query = self.query(enc)
            # key = self.key(enc)
            # value = self.value(enc)
            # multi headed attention
            # enc, _ = self.attention(query, key, value)
            # enc = self.out_att(enc).squeeze()
           
            # expand dimension
            enc_expanded = enc.unsqueeze(1).expand(-1, self.seq_length, -1)
            # concat data
            x = torch.cat((x, enc_expanded),dim=-1) 
        
    
        # possibily concat attributes after linear layer and decoer, but before LSTM
        if not self.no_static:
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
        #print(x.shape, torch.mean(x, dim=1), torch.std(x, dim=1))
        # forward pass
        enc, rec = self.forward(x,y, attr)
        
        #print(torch.mean(enc, dim=0), torch.std(enc, dim=0))
        # Logging to TensorBoard by default
        if self.loss_fn.__class__.__name__ == "MSELoss":
            train_loss = self.loss_fn(rec[:,self.warmup:,:], y[:,self.warmup:,:])
        else:
            raise SyntaxError("Invalid loss function used")
        # use first 13 encoded features as regressor of signatures
        #train_reg_loss = self.loss_fn(enc[:,:27], attr)

        self.log("train_loss", train_loss, on_step=True)
        #self.log("train_reg_loss", train_reg_loss, on_step=True)

        return train_loss #+ train_reg_loss
    
    def on_train_epoch_end(self):
        print("lr: ", self.lr_scheduler.get_lr())

    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, attr= batch
       
        # forward pas
        enc, rec = self.forward(x,y, attr)
       
        # Logging to TensorBoard by default
        if self.loss_fn.__class__.__name__ == "MSELoss":
            val_loss = self.loss_fn(rec[:,self.warmup:,:], y[:,self.warmup:,:])
        else:
             raise SyntaxError("Invalid loss function used")
        
        # compute nse
        val_nse = nse(y[:,self.warmup:,:].squeeze(), rec[:,self.warmup:,:].squeeze())
        # use first 13 encoded features as regressor of signatures
        #val_reg_loss = self.loss_fn(enc[:,:27], attr)

        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, on_step=True)
        self.log("val_nse", val_nse, on_step=True)
        #self.log("val_reg_loss", val_reg_loss, on_step=True)

        return val_loss #+ val_reg_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.lr_scheduler = AdaptiveScheduler(optimizer, milestones=self.milestones)
        return {"optimizer":optimizer, "lr_scheduler":self.lr_scheduler}


class Hydro_LSTM(pl.LightningModule):
    """
    LSTM decoder
    """
    def __init__(self,
                 input_size_dyn = 5,
                 lstm_hidden_units = 256, 
                 bidirectional = False,
                 initial_forget_bias = 5,
                 layers_num = 1,
                 loss_fn = NSELoss(),
                 drop_p = 0.4, 
                 seq_len = 270,
                 lr = 1e-3,
                 milestones = {0 : 1e-3},
                 weight_decay = 0.0,
                 encoder = None,
                ):
        
        """
        Args:
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
        self.input_size_dyn = input_size_dyn 
        self.lstm_hidden_units = lstm_hidden_units
        self.bidirectional = bidirectional 
        self.initial_forget_bias = initial_forget_bias
        self.layers_num = layers_num
        self.loss_fn = loss_fn
        self.drop_p = drop_p
        self.seq_len = seq_len
        self.lr = lr
        self.milestones = milestones
        self.weight_decay = weight_decay
        self.encoder = encoder
   
        self.lstm = nn.LSTM(input_size=self.input_size_dyn, 
                           hidden_size=self.lstm_hidden_units,
                           num_layers=self.layers_num,
                           batch_first=True,
                          bidirectional=self.bidirectional)
        
        # reset weights
        self.reset_weights()

        self.dropout = nn.Dropout(self.drop_p, inplace = False)
        if self.bidirectional:
            D = 2
        else:
            D = 1
        encoded_featues  = self.encoder.out_channels[-1] if encoder is not None else 0

        self.out = nn.Linear(D * self.lstm_hidden_units + encoded_featues, 1)

        print("LSTM initialized")

    def reset_weights(self):
        # costume initialize LSTM network (as Kratzert et al.)
        nn.init.orthogonal_(self.lstm.weight_ih_l0.data)

        weight_hh_data = torch.eye(self.lstm_hidden_units)
        weight_hh_data = weight_hh_data.repeat(1, 4)
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

    def forward(self, x, y): 
        x, _ = self.lstm(x[:,1:,:])
        x = self.dropout(x[:,-1,:])
        enc = None
        if self.encoder is not None:
            enc = self.encoder(y.squeeze().unsqueeze(1)[:,:,:-1])
            x = torch.cat((x,enc), -1)

        rec = self.out(x)
        
        return enc, rec
        
    def training_step(self, batch, batch_idx):        
        ### Unpack batch
        x, y, q_stds = batch
        #print(torch.mean(x[:,:,5:],dim=1), torch.std(x[:,:,5:], dim=1))
        # forward pass
        _, rec = self.forward(x, y)
        # Logging to TensorBoard by default
        if self.loss_fn.__class__.__name__ == "NSELoss":
            train_loss = self.loss_fn(rec, y[:,-1,:], q_stds)
        elif self.loss_fn.__class__.__name__ == "MSELoss":
            train_loss = self.loss_fn(rec, y[:,-1,:])
        else:
            print("Invalid loss function used")

        self.log("train_loss", train_loss, on_step=True)
        return train_loss
    
    def on_train_epoch_end(self):
        print("lr: ", self.lr_scheduler.get_lr())
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, q_stds = batch
        # forward pass
        _, rec = self.forward(x, y)
        # Logging to TensorBoard by default
        if self.loss_fn.__class__.__name__ == "NSELoss":
            val_loss = self.loss_fn(rec, y[:,-1,:], q_stds)
        elif self.loss_fn.__class__.__name__ == "MSELoss":
            val_loss = self.loss_fn(rec, y[:,-1,:])
        else:
            print("Invalid loss function used")

        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, on_step=True)
       
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.lr_scheduler = AdaptiveScheduler(optimizer, milestones = self.milestones)
        return {"optimizer":optimizer, "lr_scheduler":self.lr_scheduler}


