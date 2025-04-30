import torch
import torch.nn.functional as F

class MAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.optimizer = torch.optim.Adam(mac.parameters(), lr=float(args.lr))

    def cuda(self):
        self.mac.agent.cuda()

    def train(self, batch):
        batch = batch.to(self.args.device)
        obs = batch['obs']
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        returns = batch['returns']
        advantages = batch['advantages']

        logits = self.mac.act(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_preds = self.mac.evaluate_value(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)

        entropy = dist.entropy().mean()

        total_loss = policy_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mac.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
