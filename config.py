BEST_WEIGHTS = {
    'mn40': {
        1024: {
            'pointnet': 'pretrain/mn40/pointnet/best.pth',
            'pointnet2': 'pretrain/mn40/pointnet2/best.pth',
            'dgcnn': 'pretrain/mn40/dgcnn/best.pth',
            'pointmlp': 'pretrain/mn40/pointmlp/best.pth',
            'pointmlpelite': 'pretrain/mn40/pointmlpelite/best.pth',
            'pt2': 'pretrain/mn40/pt2/best.pth',
        },
    },
    'sonn': {
        1024: {
            'pointnet': 'pretrain/sonn/pointnet/best.pth',
            'pointnet2': 'pretrain/sonn/pointnet2/best.pth',
            'dgcnn': 'pretrain/sonn/dgcnn/best.pth',
            'pointmlp': 'pretrain/sonn/pointmlp/best.pth',
            'pointmlpelite': 'pretrain/sonn/pointmlpelite/best.pth',
            'pt2': 'pretrain/sonn/pt2/best.pth',
        },

    },
}

# PU-Net trained on Visionair with 1024 input point number, up rate 4
PU_NET_WEIGHT = 'defense_scripts/DUP_Net/pu-in_1024-up_4.pth'
