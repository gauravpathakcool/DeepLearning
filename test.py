import random

random.seed(1)
val_ratio = .25

while True:
    if random.random() < val_ratio:
        print("hi")
    else:
        break
