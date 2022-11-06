import torch
import torch.optim


def start_opt(args, model):
    # Create Optimizer or Restart for final layer tuning
    if args.opt == 'Adam':
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        opt = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_decay = torch.optim.lr_scheduler.StepLR(opt, step_size= args.sch_step, gamma= args.sch_gamma)
    return opt, lr_decay
