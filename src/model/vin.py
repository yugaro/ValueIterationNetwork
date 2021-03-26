import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class VIN(nn.Module):
    def __init__(self, args):
        super(VIN, self).__init__()
        self.args = args
        self.ly_hidden = nn.Conv2d(in_channels=args.num_input, out_channels=args.num_hidden,
                                   kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.ly_reward = nn.Conv2d(in_channels=args.num_hidden, out_channels=1,
                                   kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.ly_q_value = nn.Conv2d(in_channels=1, out_channels=args.num_qlayer,
                                    kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.fc = nn.Linear(in_features=args.num_qlayer, out_features=8, bias=False)
        self.weight_v_value = Parameter(torch.zeros(args.num_qlayer, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        # params
        self.reward_image = None
        self.v_value_image = None

    def forward(self, input_view, state_x, state_y, num_vi):
        # intermediate output
        hidden = self.ly_hidden(input_view)

        # get reward
        reward = self.ly_reward(hidden)
        self.reward_image = reward.data.cpu().numpy().reshape(
            self.args.dom_size, self.args.dom_size)

        # get initial q value from reward
        q_value = self.ly_q_value(reward)

        # get v value
        v_value, _ = torch.max(q_value, dim=1, keepdim=True)
        self.v_value_image = v_value.data.cpu().numpy().reshape(
            1, self.args.dom_size, self.args.dom_size)

        def eval_q_value(r, v):
            return F.conv2d(
                # concatenate reward with most recent value
                input=torch.cat([r, v], 1),
                # convolve reward -> q value weights with reward and v value-> q value weights with v value.
                weight=torch.cat(
                    [self.ly_q_value.weight, self.weight_v_value], 1),
                stride=1,
                padding=1)

        # Update q and v values
        for i in range(num_vi - 1):
            q_value = eval_q_value(reward, v_value)
            v_value, _ = torch.max(q_value, dim=1, keepdim=True)
            self.v_value_image = np.append(self.v_value_image, v_value.data.cpu(
            ).numpy().reshape(1, self.args.dom_size, self.args.dom_size), axis=0)
        q_value = eval_q_value(reward, v_value)

        batch_size, l_q, _, _ = q_value.size()

        qvalue_state_xy = q_value[torch.arange(batch_size), :, state_x.long(),
                                  state_y.long()].view(batch_size, l_q)

        # transform qvalue corresponding to current state into actions
        logits = self.fc(qvalue_state_xy)

        return logits, self.sm(logits)
