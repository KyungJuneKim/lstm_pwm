from random import randrange


def make_data_set(rate, period: int = 20, cycle: int = 5):
    data = []

    for i in range(period):
        if i < rate * period:
            data.append((60000+randrange(2000))/70000)      # normalization
        else:
            data.append(randrange(2000)/70000)              # normalization

    return data*cycle
