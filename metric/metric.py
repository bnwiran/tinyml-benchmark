import torch


@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)

    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)

    t2 = time.time()
    return (t2 - t1) / n_test  # average latency