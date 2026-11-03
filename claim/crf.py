"""Linear chain CRF with exact forward partition, Viterbi decoding and mask support."""
import torch
from torch import nn


class LinearCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.normal_(self.transitions, std=0.02)

    def forward(self, emissions, tags, mask):
        # emissions [batch,seq,tags], tags [batch,seq], bool mask [batch,seq]
        mask = mask.bool()
        if not bool(mask[:, 0].all()):
            raise ValueError('The first sequence element must be valid')
        batch, length, _ = emissions.shape
        indices = torch.arange(batch, device=emissions.device)
        score = self.start[tags[:, 0]] + emissions[indices, 0, tags[:, 0]]
        alpha = self.start + emissions[:, 0]
        for step in range(1, length):
            prev, curr = tags[:, step-1], tags[:, step]
            transition = self.transitions[prev, curr]
            increment = transition + emissions[indices, step, curr]
            score += increment * mask[:, step]
            proposal = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
                                       + emissions[:, step].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, step].unsqueeze(1), proposal, alpha)
        final_tag = tags.gather(1, mask.long().sum(1).sub(1).unsqueeze(1)).squeeze(1)
        score += self.end[final_tag]
        log_partition = torch.logsumexp(alpha + self.end, dim=1)
        return (log_partition - score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        mask = mask.bool()
        dp = self.start + emissions[:, 0]
        backtrack = []
        for step in range(1, emissions.size(1)):
            potentials = dp.unsqueeze(2) + self.transitions.unsqueeze(0)
            best, previous = potentials.max(dim=1)
            proposal = best + emissions[:, step]
            dp = torch.where(mask[:, step].unsqueeze(1), proposal, dp)
            backtrack.append(previous)
        end_scores = dp + self.end
        last = end_scores.argmax(dim=1)
        results = []
        for sample in range(emissions.size(0)):
            n = int(mask[sample].sum())
            path = [int(last[sample])]
            for step in range(n-2, -1, -1):
                path.append(int(backtrack[step][sample, path[-1]]))
            results.append(path[::-1])
        return results
