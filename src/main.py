from lstd_agent import LSTDAgent

def features(x, a):
    """Return a feature vector representing the given state and action"""
    output = [0] * (16*4)
    output[x+a*16] = 1
    return np.array([output])

def main():
    pass

if __name__ == "__main__":
    main()
