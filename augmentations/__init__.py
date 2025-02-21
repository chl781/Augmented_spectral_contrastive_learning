from .simsiam_aug import SimSiamTransform, StandardTransform
from .eval_aug import Transform_single


def get_aug(model_cfg,epsilon,name='simsiam', image_size=224, train=True, train_classifier=None):
    if train==True:
        if name == 'standard':
            augmentation = StandardTransform(image_size)
        elif name == 'spectral':
            augmentation = SimSiamTransform(epsilon,image_size)
            # Change this step: the scaling should be scale down for the augmentation.
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








