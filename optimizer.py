import torch
import torch.optim

def create_LARS(opt_class, opt):
    class LARS(opt_class):
        def __init__(self, opt, trust_coef=0.02, eps=1e-8):
            self.param_groups = opt.param_groups
            self.opt = opt
            self.trust_coef = trust_coef
            self.eps = eps

        def step(self):
            with torch.no_grad():
                weight_decays = []
                for group in self.opt.param_groups:
                    if 'weight_decay' in group:
                        weight_decay = group['weight_decay']
                    else:
                        weight_decay = 0
                        group['weight_decay'] = 0
                    weight_decays.append(weight_decay)

                    for param in group['params']:
                        if param.grad is None:
                            continue
                        p_norm = torch.norm(param.data)
                        g_norm = torch.norm(param.grad.data)
                        if p_norm == 0 or g_norm == 0:
                            continue

                        adapt_lr = self.trust_coef * p_norm / (
                                    g_norm + p_norm * weight_decay + self.eps)
                        param.grad.data = (param.grad.data + weight_decay * param.data) * adapt_lr

            self.opt.step()

            for i, group in enumerate(self.opt.param_groups):
                group['weight_decay'] = weight_decays[i]

    return LARS(opt)


def start_opt(args, model):
    # Create Optimizer or Restart for final layer tuning
    if args.opt == 'Adam':
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        opt = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'LARS':
        opt = create_LARS(torch.optim.Adam,
                          torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    
    lr_decay = torch.optim.lr_scheduler.StepLR(opt, step_size= args.sch_step, gamma= args.sch_gamma)
    return opt, lr_decay