from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)

# Verisetini yükleme
dataset = numpy.loadtxt("CHDdata.csv", delimiter=",")

# Veri setini özellikler (features) ve etiketlere (labels) ayırma
datalar = dataset[:, 0:9]  # İlk 9 sütun özellikleri temsil eder
etiketler = dataset[:, 9]  # Son sütun etiketleri temsil eder

model = Sequential()
model.add(Dense(45, input_dim=9, activation='sigmoid'))  # H1
model.add(Dense(45, activation='tanh'))  # H2
model.add(Dense(45, activation='tanh'))  # H3
model.add(Dense(45, activation='tanh'))  # H4
model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(datalar, etiketler, epochs=1200, batch_size=9)

basarim = model.evaluate(datalar, etiketler)
print(basarim)
#model.save('CHDdataModel.h5')