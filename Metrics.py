import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Provides various evaluation metrics and losses used to train and evaluate performance 

def dice_metric(y_pred,y_true):
    
    """
        Traditional dice
        
        Args:
            y_pred : the predicitied probability map
            y_true : the GT probability map
    """
    # cast to binary 1st
     
    smooth = 1                 # Smooth term is added to both numerator and denomerator to prevent division by 0
    num = y_true.size(0)
    m1 = y_pred.view(num, -1)
    m2 = y_true.view(num, -1)
    
        
    intersection = (m1* m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = score.sum() / num
        
    return score

class MultiClassBCE(nn.Module):
    
    """
        MultiClassBCE
        
        Applies weighted BCELoss on individual channels of the output
        
        Args:
            weights : (list) weights to apply to each channel
            
    """
        
        
    def __init__(self, weights=[0.5,0.5]):
        self.weights=weights
        super().__init__()

    def forward(self, logits, targets):
        
        """
        Args:
            logits : (tensor) Predictied probability map
            targets : (tensor) GT probability map
            
        """
        smooth = 1
        num = targets.size(0)
        cat_class=targets.size(1)
        m1 = logits.view(num,cat_class, -1)
        m2 = targets.view(num,cat_class, -1)
        final_loss=0
        
        
        for cat in range(cat_class):
            
            loss=nn.BCELoss()(m1[:,cat,:],m2[:,cat,:])
            final_loss+=self.weights[cat]*loss
            
            del loss

        return final_loss
    
    
class SoftDiceLoss(nn.Module):
    """
        SoftDiceLoss
        
        Implementation of Dice loss
        
    """
    
    
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        
        """
        Args:
            logits : (tensor) Predictied probability map
            targets : (tensor) GT probability map
            
        """
        
        
        smooth = 1
        num = targets.size(0)
        #probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    
    
    
class DE_loss(nn.Module):
    """
        SoftDiceLoss
        
        Implementation of Novel loss
        
        Args:
            dice_coeff :  weights to apply dice loss
            
    """
    def __init__(self,dice_coeff=0.02):
        self.dice_coeff=dice_coeff
        super().__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits : (tensor) Predictied probability map
            targets : (tensor) GT probability map
            
        """
        
        
        bce=nn.BCELoss()(logits,targets)
        smooth = 1
        num = targets.size(0)
        
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        dice_loss = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice_loss = 1 - dice_loss.sum() / num
        return dice_loss*self.dice_coeff + bce
    
    
    