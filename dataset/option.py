import argparse

parser = argparse.ArgumentParser()

# Model Options
parser.add_argument('--model', type = str, default = 'HPSR')
parser.add_argument('--scale', type = int, default = 5)
parser.add_argument('--channel1', type = int, default = 8)
parser.add_argument('--channel2', type = int, default = 16)
parser.add_argument('--channel3', type = int, default = 6)

# Training Options
parser.add_argument('--batch-size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 300)
parser.add_argument('--lr', type = float, default = 1e-3)
parser.add_argument('--lr-decay-steps', type = int, default = 200)
parser.add_argument('--lr-decay-gamma', type = float, default = 0.5)
