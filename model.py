from tensorflow import keras
from tensorflow.keras import layers

K1 = [None, 3, 3, None, 6]
K2 = [None, 3, 3, None, 5]
P = [None, 1, 1, None, 2]

class CNN:
    def __init__(self, tm, W):
        self.tm = tm
        self.W = W
        self.N = 7
        self.t1 = 2 if 1 > tm else 4
        self.t2 = 2 if 2 > tm else 4
        self.t3 = 2 if 3 > tm else 4
        self.t4 = 2 if 4 > tm else 4
        self.model = None
        self.initialize_layers()
    
    def initialize_layers(self):
        inputs = keras.Input(shape=(128, None, 1), name="audio")

        x = layers.BatchNormalization()(inputs)

        # block 0
        x = tfb_1(x, 1, 1, 8 * self.W)
        x = tfb(x, 1, 1, 8 * self.W)
        x = tfb(x, 1, 1, 8 * self.W)
        x = tfb(x, 1, 1, 8 * self.W)

        # block 1
        x = tfb(x, 2, self.t1, 16 * self.W)
        x = tfb(x, 1, 1, 16 * self.W)
        x = tfb(x, 1, 1, 16 * self.W)
        x = tfb(x, 1, 1, 16 * self.W)

        # block 2
        x = tfb(x, 2, self.t2, 32 * self.W)
        x = tfb(x, 1, 1, 32 * self.W)
        x = tfb(x, 1, 1, 32 * self.W)
        x = tfb(x, 1, 1, 32 * self.W)

        # block 3
        x = tfb(x, 2, self.t3, 64 * self.W)
        x = tfb(x, 1, 1, 64 * self.W)
        x = tfb(x, 1, 1, 64 * self.W)
        x = tfb(x, 1, 1, 64 * self.W)

        # block 4
        x = tfb(x, 2, self.t4, 128 * self.W)
        x = tfb(x, 1, 1, 128 * self.W)
        x = tfb(x, 1, 1, 128 * self.W)
        x = tfb(x, 1, 1, 128 * self.W)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128 * self.W, activation=layers.LeakyReLU(alpha=0.01))(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)

        outputs = layers.Dense(self.N, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs, name='toy_cnn')
        
        opt = keras.optimizers.Adam(learning_rate=0.0002)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(self.model.summary())


def tfb(input, x, y, c):
    print(input)

    fx = layers.BatchNormalization()(input)
    fx = layers.LeakyReLU(alpha=0.1)(fx)
    fx = layers.ZeroPadding2D(padding=(1, 1))(fx)
    fx = layers.Conv2D(c, (K1[x], K1[y]), (x, y), padding='valid')(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.LeakyReLU(alpha=0.1)(fx)
    fx = layers.ZeroPadding2D(padding=(P[x], P[y]))(fx)
    fx = layers.Conv2D(c, (K2[x], K2[y]), (1, 1), padding='valid')(fx)

    res = layers.ZeroPadding2D(padding=(0, 0))(input)
    res = layers.Conv2D(c, (1, 1), (x, y), padding='valid')(res)

    out_layers = []

    if c == input.shape[2] and (x == 1 and y == 1):
        out_layers.append(input)
    
    out_layers.append(fx)

    if c != input.shape[2] or x != 1 or y != 1:
        out_layers.append(res)

    out = layers.Add()(out_layers)

    return out

def tfb_1(input, x, y, c):
    fx = layers.ZeroPadding2D(padding=(1, 1))(input)
    fx = layers.Conv2D(c, (K1[x], K1[y]), (x, y), padding='valid')(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.LeakyReLU(alpha=0.1)(fx)
    fx = layers.ZeroPadding2D(padding=(P[x], P[y]))(fx)
    fx = layers.Conv2D(c, (K2[x], K2[y]), (1, 1), padding='valid')(fx)

    res = layers.ZeroPadding2D(padding=(0, 0))(input)
    res = layers.Conv2D(c, (1, 1), (x, y), padding='valid')(res)

    out_layers = []

    if c == input.shape[2] and (x == 1 and y == 1):
        out_layers.append(input)
    
    out_layers.append(fx)

    if c != input.shape[2] or x != 1 or y != 1:
        out_layers.append(res)

    out = layers.Add()(out_layers)

    return out