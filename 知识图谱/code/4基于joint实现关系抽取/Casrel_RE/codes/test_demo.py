import torch
import torch.nn as nn


def bce_loss_demo():
    y_pred = torch.tensor([-9, 1.0, ], requires_grad=True)
    y_true = torch.tensor([1, 1], dtype=torch.float32)
    # 2 实例化二分类交叉熵损失
    # 如果是none则保留所有损失结果
    criterion = nn.BCELoss(reduction='none')  # 默认reduction='mean'
    y_pred = torch.sigmoid(y_pred)
    print(f'y_pred_sigmoid-->{y_pred}')
    loss = criterion(y_pred, y_true)
    print(f'loss-->{loss}')


def repeat():
    ts1 = torch.randn(2, 3, 1)
    print(f'ts1-->{ts1}')

    new_tensor = ts1.repeat(1, 1, 18)
    print(f'new_tensor-->{new_tensor}')
    print(f'new_tensor-->{new_tensor.shape}')
    # x1 [batch_size, seq_len ] -> [batch_size, seq_len, 1 ].repeat(1, 1, 18) -> [batch_size, seq_len, 18]
    # x2 [batch_size, seq_len, 18]


def arrange_demo():
    tensor = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.long)
    result = torch.arange(0, 8)[tensor == 1]
    print(f'tensor-->{tensor}')
    print(f'result-->{result}')

if __name__ == '__main__':
    # bce_loss_demo()
    # repeat()
    arrange_demo()