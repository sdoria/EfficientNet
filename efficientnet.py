import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial



__all__ = ['EfficientNet', 'efficientnetB0','efficientnetB1', 'efficientnetB2', 'efficientnetB3', 'efficientnetB4', 'efficientnetB5', 'efficientnetB6', 'efficientnetB7']

class Swish(nn.Module):
    def forward(self, x):
        x = x * torch.sigmoid(x)  #nn.functional.sigmoid is deprecated, use torch.sigmoid instead
        return x

act_fn = Swish() #nn.ReLU(inplace=True)


#added groups, needed for DWConv
#"The configuration when groups == in_channels and out_channels = K * in_channels where K is a positive integer is termed in literature as depthwise convolution."

def conv(ni, nf, ks=3, stride=1, groups=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, groups= groups, bias=bias)


#class noop(nn.Module):
  #  def __init__(self):
  #      super().__init__()
   # def forward(self,x): return x
    
def noop(x): return x

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)


# not compatible with fp16 training        
class Drop_Connect(nn.Module):
    """create a tensor mask and apply to inputs, for removing drop_ratio % of weights"""
    def __init__(self, drop_ratio=0):
        super().__init__()
        self.keep_percent = 1.0 - drop_ratio

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.size(0)
        random_tensor = self.keep_percent
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype,device=x.device)   #dtype is causing issues with fp16 training
        binary_tensor = torch.floor(random_tensor)
        output = x / self.keep_percent * binary_tensor

        return output
    
    
def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,device=inputs.device)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


#added groups, needed for DWConv
#fixed batch norm momentum = 1- Tensorflow value
def conv_layer(ni, nf, ks=3, stride=1,groups=1, zero_bn=False, act=True, eps=1e-03, momentum=0.01):
    bn = nn.BatchNorm2d(nf, eps=eps, momentum=momentum)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride, groups=groups), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)



class SqueezeEx(nn.Module):
    def __init__(self, ni, ns):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        ns = max(1, int(ns))
        
        layers = [nn.AdaptiveAvgPool2d(1),
                      conv(ni,ns,ks=1,bias=True),
                      act_fn,
                      conv(ns,ni,ks=1,bias=True),
                      nn.Sigmoid()]


        self.layers = nn.Sequential(*layers)

    def forward(self, x):
          
        return x * self.layers(x)




class MBConv(nn.Module):
    def __init__(self, ni, nf, expand_ratio, ks=3, stride=2, se = None, skip=True, drop_connect_rate=None):
        super().__init__()




        # Expansion (only if expand ratio>1)

        ne = ni*expand_ratio
        self.conv_exp = noop if ni==ne else conv_layer(ni, ne, ks=1)

        # Depthwise Convolution (implemented using 'groups')
        # This is where ks and stride get used
        #"The configuration when groups == in_channels and out_channels = K * in_channels 
        # where K is a positive integer is termed in literature as depthwise convolution."
        # depth_multiplier=1 is default in original TF code so we keep the same number of channels

        self.dw_conv = conv_layer(ne, ne, ks=ks, stride= stride, groups=ne)


        # Squeeze and Excitation (if se ratio is specified)
        # se ratio applies to ni and not ne


        self.se = SqueezeEx(ne, ni*se) if se else noop

        # Output Conv (no relu)

        self.conv_out = conv_layer(ne, nf, ks=1, act=False)

        

        # add skip connection or not
        self.skip = skip and stride==1 and ni==nf

        # Drop connect

        #self.dc = Drop_Connect(drop_connect_rate) if drop_connect_rate else noop
        self.dc = partial(drop_connect,p=drop_connect_rate, training=self.training) if drop_connect_rate else noop


    def forward(self, x): 
        
        out = self.conv_out(self.se(self.dw_conv(self.conv_exp(x))))
        if self.skip: out = self.dc(out) + x


        return out



class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class EfficientNet(nn.Sequential):
    def __init__(self, channels, repeat, ks, stride, expand, w_mult=1.0, d_mult=1.0, se = None, drop_connect_rate = None,dropout_rate= None, c_in=3, c_out=1000):

        
        repeat = [int(math.ceil(r*d_mult)) for r in repeat]
        channels = round_filters(channels, w_mult)
        
        
        stem = [conv_layer(c_in, channels[0], ks=3 ,stride=2)]

        blocks = []
        #The first block needs to take care of stride and filter size increase.

        for i in range(len(repeat)):
            blocks+= [MBConv(channels[i], channels[i+1], expand[i], ks=ks[i], stride=stride[i], se = se, drop_connect_rate=drop_connect_rate)]
            blocks+= [MBConv(channels[i+1], channels[i+1], expand[i], ks=ks[i], stride=1, se = se, drop_connect_rate=drop_connect_rate)] *(repeat[i]-1)

        dropout = nn.Dropout(p=dropout_rate) if dropout_rate else noop

        head = [conv_layer(channels[-2], channels[-1], ks=1 ,stride=1), nn.AdaptiveAvgPool2d(1), Flatten(), dropout, nn.Linear(channels[-1], c_out)]


        super().__init__(*stem,*blocks, *head)
                      
        init_cnn(self)


        
        
def round_filters(filters, d_mult, divisor=8, min_depth=None):
    """ Calculate and round number of filters based on depth multiplier. """
    
    if not d_mult:
        return filters
    
    filters = [f*d_mult for f in filters]
    min_depth = min_depth or divisor
    new_filters = [max(min_depth, int(f + divisor / 2) // divisor * divisor) for f in filters]
    # prevent rounding by more than 10%
    new_filters = [new_filters[i] + (new_filters[i] < 0.9 * filters[i])* divisor for i in range(len(new_filters))]
    new_filters = [int(f) for f in new_filters]
    return new_filters


me = sys.modules[__name__]
c = [32,16,24,40,80,112,192,320,1280]
r = [1,2,2,3,3,4,1]
ks = [3,3,5,3,5,5,3]
str = [1,2,2,2,1,2,1]
exp = [1,6,6,6,6,6,6]
se = 0.25
do = 0.2
dc=0.2


# base without multipliers and dropout
setattr(me, 'efficientnet', partial(EfficientNet, channels=c, repeat=r, ks=ks, stride=str, expand=exp, se=se, drop_connect_rate=dc))

# (number, width_coefficient, depth_coefficient, dropout_rate) 
for n, wm, dm, do in [
    [ 0, 1.0, 1.0, 0.2],
    [ 1, 1.0, 1.1, 0.2],
    [ 2, 1.1, 1.2, 0.3],
    [ 3, 1.2, 1.4, 0.3],
    [ 4, 1.4, 1.8, 0.4],
    [ 5, 1.6, 2.2, 0.4],
    [ 6, 1.8, 2.6, 0.5],
    [ 7, 2.0, 3.1, 0.5],
]:
    name = f'efficientnetB{n}'
    setattr(me, name, partial(efficientnet, d_mult=dm, w_mult=wm, dropout_rate=do))

